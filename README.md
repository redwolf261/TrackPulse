# TrackPulse — Live Track Condition Detector

Built for the "Weather Whiplash" hackathon problem statement: given a track camera
image, estimate whether the surface is Dry, Damp, or Wet, show how that's trending
over a session, and give a plain-language suggestion.

## What it does

1. Upload a track photo (or click one of the two bundled sample images).
2. A fine-tuned image classifier (MobileNetV3-Small) predicts **DRY / DAMP / WET**
   with a confidence score.
3. The trend across your session's uploads is plotted (getting wetter / drying out /
   stable), computed deterministically from the label history — no LLM involved.
4. A simple, rule-based suggestion is shown (e.g. *"Track drying: slick tyre window
   approaching."*).

## Stack

- **Backend**: FastAPI + SQLite (`backend/`) — `/predict`, `/history/{session}`, `/health`.
- **Frontend**: Vite + React + TypeScript + Recharts (`frontend/`).
- **Model**: `torchvision.models.mobilenet_v3_small`, ImageNet-pretrained, fine-tuned
  for this task, exported to ONNX and served via ONNX Runtime.
- **Hugging Face Hub usage**: training data (`rezzzq/RSCD-1million`, a mirror of the
  Tsinghua Road Surface Condition Dataset) was downloaded from the HF Hub via the
  `datasets`/`huggingface_hub` libraries, and the classifier is initialized from
  torchvision's ImageNet-pretrained weights before fine-tuning.

## Why this is a fine-tune, not a from-scratch model or a single API call

The task explicitly asks for something in between those two extremes. We:

- Started from an ImageNet-pretrained backbone (not trained from random weights).
- Fine-tuned it ourselves on a filtered, relabeled, leakage-safe split of a public
  dataset, then again on a small hand-collected racing-domain set once we discovered
  the source-domain model didn't transfer well (see below) — real modeling work, not
  a wrapper around an existing wet/dry classifier.

## Methodology (short version)

We didn't just train a model and ship it — we ran it through an evidence-first
validation process, iterated three times, and re-measured after every change on the
same untouched evaluation set:

1. **Source-domain training (exp00)**: filtered ~50k RSCD images down to 3,266
   asphalt/concrete, dry/wet/water images, mapped to our DRY/DAMP/WET labels. Split by
   *capture-time group* (not random per-image) so near-duplicate frames from the same
   driving session never leak across train/val/test. Trained MobileNetV3-Small;
   **RSCD held-out test: 77.7% accuracy, 0.754 macro-F1, 83% WET recall.**
2. **Reality check**: that's a road-camera dataset, not a racing-track camera. We
   built an independent 49-image racing-domain evaluation set (Wikimedia Commons,
   CC-licensed, hand-labeled, multiple circuits/years to avoid single-clip bias) and
   ran the frozen model on it. Result: **accuracy collapsed to 34%**, with 29/30 dry
   racing images misclassified as WET — a real, measured domain gap, not a rounding
   error. The model had learned "dark, matte asphalt = wet," which doesn't hold on
   modern F1 pit-lane tarmac.
3. **Domain adaptation, round 1 (exp01)**: built a disjoint 115-image racing training
   set (zero overlap with the eval set — verified by SHA-256 hash), fine-tuned from
   the frozen checkpoint. Racing accuracy improved to 50%, DRY recall 3%→30%, at a
   cost of ~16 points of RSCD WET recall — a real trade-off, not a free win.
4. **Domain adaptation, round 2 (exp02)**: the biggest remaining gap was DAMP —
   zero examples anywhere in the project. Built a second, larger, disjoint training
   set (177 images: 117 DRY, 38 DAMP, 22 WET; genuinely hard to source — DAMP/
   transitional shots are rare even in rain-affected races, ~350 of ~2,100 candidate
   photos were viable after visual triage), fine-tuned from exp01's checkpoint.
   Re-evaluated on the *same untouched* 49-image set: **accuracy 50%→73%, DRY recall
   30%→63%, false-WET rate 60%→20%**, while WET recall held at 93%. RSCD performance
   stayed roughly flat (mild, stable trade-off, not a regression spiral).

| | RSCD test (n=305) | Racing eval (n=44, held out throughout) |
|---|---|---|
| exp00 (frozen baseline) | acc 77.7%, WET recall 83.1% | acc 34.1%, DRY recall 3.3%, false-WET 96.7% |
| exp01 (+115 racing imgs) | acc 75.4%, WET recall 67.5% | acc 50.0%, DRY recall 30.0%, false-WET 60.0% |
| **exp02 (+177 racing imgs, DAMP-focused)** | acc 74.8%, WET recall 75.3% | **acc 72.7%, DRY recall 63.3%, false-WET 20.0%** |

**exp02 is the model currently in production** (`models/trackpulse_classifier.onnx`).
Earlier versions are kept as backups (`models/trackpulse_classifier_v1_frozen_backup.onnx`,
`_v2_backup.onnx`) for comparison/rollback.

## Honest limitations

- **DAMP is still unvalidated on racing imagery** — our 49-image evaluation set has
  zero confirmed-damp racing photos (genuinely hard to find/label even in CC-licensed
  archives), so while we trained on 38 real racing DAMP examples in the latest round,
  we can't directly measure DAMP accuracy on racing footage the way we can for
  DRY/WET.
- **The model still misreads some dry racing images as wet** — down to a 20%
  false-WET rate on our racing eval set (from 97% pre-fine-tune, 60% after the first
  fine-tune round). Meaningfully better, not perfect. This is visible in the UI as a
  stated caveat, not hidden behind a false-confidence display.
- Racing-domain evaluation is a 44-image (49 minus 5 ambiguous) hand-labeled
  spot-check, not a statistically rigorous benchmark — useful for catching and
  tracking a large domain-gap failure across iterations, not for precise
  population-level accuracy claims.

We chose to surface this honestly (in the UI and here) rather than present the model
as more reliable than it's been measured to be.

## Running it locally

```bash
# backend
cd backend
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000

# frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Backend defaults to `http://127.0.0.1:8000`, frontend to `http://localhost:5173`
(configured via `frontend/.env`).

## Project structure

```text
backend/app/                FastAPI service, ONNX inference, deterministic trend/suggestion logic
frontend/src/                React UI
scripts/                     Data pipeline, training, ONNX export, evaluation scripts
experiments/                 Per-experiment configs, metrics, checkpoints:
                                exp00 = RSCD source-domain baseline
                                exp01 = racing fine-tune round 1 (+115 images)
                                exp02 = racing fine-tune round 2 (+177 images, DAMP-focused)
data/manifests/               Dataset splits and label manifests (RSCD)
data/racing_spotcheck/        Racing-domain evaluation set (49 images, ground truth, licenses) —
                                 held out from all training rounds, never touched
data/racing_train_pool/       exp01's racing training images
data/racing_train_pool_v2/    exp02's racing training images (disjoint from exp01's and from
                                 the eval set — verified by SHA-256 hash each round)
models/                      Trained checkpoints and ONNX exports (v1/v2/v3 = exp00/01/02;
                                 trackpulse_classifier.onnx = current production model, exp02)
```
