# TrackPulse — Live Track Condition Detector

Built for the "Weather Whiplash" hackathon problem statement: given photos or video
frames of the track, estimate whether the surface is Dry, Damp, or Wet, show how
that's trending over a session, and give a plain-language suggestion.

## What it does

1. Upload a track photo (or click one of the three bundled sample images), **or
   upload a short video clip** — the backend samples frames from it (up to 12, at a
   fixed interval across the clip) and classifies each one in sequence, exactly as
   if they'd been uploaded one at a time.
2. A fine-tuned image classifier (MobileNetV3-Small) predicts **DRY / DAMP / WET**
   with a confidence score, per frame.
3. The trend across your session's uploads is plotted (getting wetter / drying out /
   stable), computed deterministically from the label history — no LLM involved. A
   single video upload can populate the whole trend chart in one go.
4. A simple, rule-based suggestion is shown (e.g. *"Track drying: slick tyre window
   approaching."*).

## Stack

- **Backend**: FastAPI + SQLite (`backend/`) — `/predict` (image), `/predict/video`
  (video → sampled frames → same classification pipeline), `/history/{session}`,
  `/health`. Video frame extraction via OpenCV.
- **Frontend**: Vite + React + TypeScript + Recharts (`frontend/`).
- **Model**: `torchvision.models.mobilenet_v3_small`, ImageNet-pretrained, fine-tuned
  for this task, exported to ONNX and served via ONNX Runtime.
- **Hugging Face Hub usage**: training data (`rezzzq/RSCD-1million`, a mirror of the
  Tsinghua Road Surface Condition Dataset) was downloaded from the HF Hub via the
  `datasets`/`huggingface_hub` libraries, and the classifier is initialized from
  torchvision's ImageNet-pretrained weights before fine-tuning.

## Impact & scalability

**Who this actually helps.** The direct persona is a race/strategy engineer or spotter
who needs a fast, cheap second opinion on track condition from a camera feed —
trackside, onboard, or broadcast — without waiting on a human to eyeball footage or a
weather station that reports rain, not surface state. The gap those two miss is real:
weather stations describe what's falling from the sky, not what's actually on the
tarmac right now, and by the time a human calls it out over radio, conditions may have
already shifted. The classifier itself is fast enough not to be the bottleneck: p50
1.78ms / p95 2.76ms / p99 3.63ms per frame, CPU-only, ONNX Runtime, measured over 200
runs (`experiments/exp00_rscd_baseline/onnx_benchmark_results.json`) — the
observation-to-decision loop is limited by camera/network latency and human review
time, not by this model.

**What "scale" honestly looks like from here, in order of how close each is to real:**

1. **More cameras, same pipeline, no architecture change.** The system is already
   stateless per-frame and horizontally scalable — `/predict` and `/predict/video`
   don't share state between requests beyond the SQLite write, so running this
   behind a queue in front of N camera feeds (multiple corners of a track, multiple
   vehicles' onboard cameras) is an infrastructure change, not a rearchitecture.
2. **Beyond motorsport, to any domain where "does this surface look like X" from a
   camera has value cheaply**: municipal road-icing/flooding alerts, warehouse floor
   hazard detection, drone-based agricultural field-condition checks. The RSCD→racing
   domain-gap work in this repo is a direct demonstration of the actual risk in doing
   this — a model trained on one camera domain does not automatically transfer to
   another (we measured a 43-point accuracy drop moving from road-camera data to
   racing-camera data before domain adaptation), so "scaling to a new domain" is
   real, costed engineering work, not a checkbox. We'd tell a team evaluating this
   for a new surface/camera domain to expect the same three-round validate → find
   the gap → close it cycle documented below, not a drop-in reuse.
3. **What's NOT yet scalable, stated plainly**: no auth, no multi-tenant isolation
   between sessions beyond an unguessable session ID, free-tier hosting (cold starts
   on the deployed backend after idling). Rate limiting (per-IP, 30 req/min on image
   upload, 10/min on video) and a Postgres-backed persistence path (tested against a
   real instance, currently defaults to SQLite for local dev) are built and working
   as of this writing, closing two of the gaps from an earlier pass — the remaining
   ones are real and not yet closed, not a checklist we're claiming to have finished.

**Bottom line**: the interesting scaling story here isn't "add more GPUs," it's that
visual domain transfer for track-condition perception is a genuinely measured,
non-trivial cost — and this project is evidence for exactly how much that cost is
and how to pay it down, not a claim that it's already solved.

## Business model

We built this as a hackathon prototype, not a company — so this section is an
honest sketch of a plausible path, not a claim we've validated a market.

**Who would actually pay, and for what.** Not a per-consumer app — the buyer is a
team, series, or venue that already has camera infrastructure and a decision that
depends on track condition: a race/strategy engineer's tooling budget, a track-day
or club-racing operator managing driver safety calls, or a broadcast/production
team wanting an automated on-screen condition indicator. The honest constraint from
our own testing: professional F1-level teams already have humans and (likely)
better sensors for this; the more realistic near-term buyer is a level below that
— club racing, karting, track-day operators, driving schools — where a human
spotter watching a camera feed is a real cost and a $0-marginal-cost model call is
a genuine substitute, not a nice-to-have.

**Plausible model, in order of how validated each is:**

1. **Usage-based API** (most directly buildable from what exists): charge per
   inference or per session, aimed at teams/apps who want to embed this rather than
   use our UI. We already have the inference endpoint; this is a packaging question
   (auth + billing), not a new technology.
2. **SaaS seat/subscription** for a race-ops team: the dashboard + trend history +
   Evidence Trail as a tool a strategy engineer logs into during a session. Requires
   the auth/persistence work in the Impact & Scalability section above — a real,
   scoped next step, not hand-waving.
3. **Licensing the fine-tuned model weights** to a team building their own tooling —
   lowest-effort to ship, but the value we're actually selling in that case is the
   racing-domain adaptation work (see the Methodology section) more than the base
   model, and licensing terms would need to respect the RSCD data's non-commercial
   provenance (see Hugging Face Hub usage above) — a real constraint we'd have to
   resolve before this option is legally available, not just an execution detail.

**What we're explicitly not claiming**: revenue projections, market size, or that
we've spoken to a real customer. Those would be fabricated for a hackathon
submission. What we can defend is that the underlying capability (fast, cheap,
camera-based surface condition inference) has a real cost structure we've actually
measured — 1.78ms p50 inference, no GPU required — which is the input a real
unit-economics conversation would start from, not a guess.

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
5. **Closing the DAMP blind spot (exp03)**: exp01+exp02's training pools together
   contain 50 real racing-domain DAMP images — but every one of them had been used
   for training, none held out for evaluation, so DAMP accuracy on racing imagery had
   never actually been measured. We held out 18 of those 50 (spanning 6 different
   events, zero overlap with training — verified by SHA-256 hash), retrained
   excluding them, and evaluated on the held-out set. **First real result: DAMP
   recall 83.3% (15/18), precision 62.5%** — the model genuinely can recognize damp
   racing surfaces. But it came at a cost: DRY recall on the original 49-image set
   regressed 63.3%→50.0% and WET recall slipped 92.9%→85.7%, because the higher DAMP
   oversampling needed to force that gain also made the model over-call DAMP on
   ambiguous dry images (night races, tire smoke, pit-lane scenes). **We decided not
   to promote exp03** — exp02 remains in production because the regression on the
   larger, more established eval axis outweighs the new DAMP signal. This is a real,
   useful negative result: it tells us oversampling was the wrong lever, not that
   DAMP is unlearnable.

| | RSCD test (n=305) | Racing eval (n=44, held out throughout) |
|---|---|---|
| exp00 (frozen baseline) | acc 77.7%, WET recall 83.1% | acc 34.1%, DRY recall 3.3%, false-WET 96.7% |
| exp01 (+115 racing imgs) | acc 75.4%, WET recall 67.5% | acc 50.0%, DRY recall 30.0%, false-WET 60.0% |
| **exp02 (+177 racing imgs, DAMP-focused, production)** | acc 74.8%, WET recall 75.3% | **acc 72.7%, DRY recall 63.3%, false-WET 20.0%** |
| exp03 (DAMP holdout eval, not promoted) | acc 76.4%, WET recall (n/a, RSCD-scale) | acc 61.4%, DRY recall 50.0%, WET recall 85.7%; **DAMP recall 83.3%, precision 62.5% (n=18, first real measurement)** |

**exp02 is the model currently in production** (`models/trackpulse_classifier.onnx`),
**frozen** as the production candidate. **exp03 is retained as a research artifact,
not a candidate for deployment** (`models/trackpulse_classifier_v4_exp03.onnx`).
Earlier versions are kept as backups (`models/trackpulse_classifier_v1_frozen_backup.onnx`,
`_v2_backup.onnx`) for comparison/rollback.

### Class-balance ablation (exp03)

A controlled experiment increased DAMP oversampling from 8× (exp02) to 10× (exp03).
This substantially improved DAMP recall to 83.3% (15/18), establishing measurable
recognition of the previously underrepresented class. However, the intervention
increased false-DAMP predictions on difficult DRY samples, reducing DRY accuracy to
50.0%, while WET recall decreased from 92.9% to 85.7%. The experiment therefore
demonstrates that improving minority-class recall does not necessarily improve the
overall operational classifier. The 10× oversampling configuration was rejected, and
the 8× configuration (exp02) remains the production model.

| Model | DAMP Recall | DAMP Precision | DRY Performance | WET Recall | Decision |
|---|---:|---:|---:|---:|---|
| **exp02** | — | — | Better | **92.9%** | ✅ Production |
| **exp03** | **83.3%** | **62.5%** | Degraded to **50.0%** | **85.7%** | ❌ Rejected |

The result isn't simply that exp03 has higher DAMP recall — it demonstrates a
class-balance trade-off: increasing emphasis on the visually ambiguous DAMP class
improved DAMP detection but caused the classifier to absorb hard-negative DRY samples
into DAMP and slightly reduced WET recall. exp03 was evaluated as a class-imbalance
intervention and rejected for deployment on that evidence, while being retained as an
experimental artifact — not silently swapped in just because one metric improved.

**Production:** exp02. **Experimental/rejected:** exp03. We did not chase further
oversampling ratios or alternative rebalancing strategies (class weighting, focal
loss) after this — the experiment had already demonstrated the three things that
matter for this submission: the model learns the source task, domain shift produces
measurable failure, and correcting one weakness produces measurable trade-offs. That
is a more credible research narrative than tuning until every metric looks good.

## Honest limitations

- **DAMP recall is now measured (83.3%, n=18) but not yet reflected in production** —
  the model version that achieved this (exp03) traded away DRY/WET accuracy to get
  there, so it wasn't shipped. The production model (exp02) still has no direct DAMP
  measurement, only indirect evidence from training on real DAMP examples.
- **The model still misreads some dry racing images as wet** — down to a 20%
  false-WET rate on our racing eval set (from 97% pre-fine-tune, 60% after the first
  fine-tune round). Meaningfully better, not perfect. This is visible in the UI as a
  stated caveat, not hidden behind a false-confidence display.
- Racing-domain evaluation is a 44–62-image (depending on which eval set) hand-labeled
  spot-check, not a statistically rigorous benchmark — useful for catching and
  tracking a large domain-gap failure across iterations, not for precise
  population-level accuracy claims.

We chose to surface this honestly (in the UI and here) rather than present the model
as more reliable than it's been measured to be, and we chose not to ship a model
version just because it was newer — exp03 lost the promotion decision on its own
measured evidence.

### Where this leaves the project

We characterize this prototype as **conditionally validated, deployment-frozen, and
experimentally characterized** — not "production-ready," and not "racing-grade."

TrackPulse can provide lightweight visual track-condition estimates, but
racing-domain uncertainty and ambiguous DAMP/DRY conditions remain significant
limitations. The system therefore treats perception as evidence for decision support
rather than as an autonomous strategy authority — the UI states this explicitly (see
the model caveat shown alongside every prediction) rather than presenting a single
confident label as ground truth.

The production ONNX artifact is, and remains, exp02
(`models/trackpulse_classifier.onnx`) — the demo and submission are not running the
experimental (exp03) model.

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

## Deploying (Render + Netlify)

Config files (`render.yaml`, `netlify.toml`) are at the repo root, ready to connect.

1. **Backend → Render**: create a new Web Service from this repo. Render should
   auto-detect `render.yaml`; if not, set build command
   `pip install -r backend/requirements.txt`, start command
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT --app-dir backend`. Health
   check path `/health`. Note the assigned URL once live
   (`https://<service-name>.onrender.com`).
2. **Frontend → Netlify**: before connecting, update `frontend/.env.production`
   with the actual Render URL from step 1 (Vite bakes this in at build time, so
   it must be set before Netlify builds — there's no way to change it after the
   fact without a rebuild). Then create a new site from this repo; Netlify
   auto-detects `netlify.toml` (base `frontend/`, build `npm run build`, publish
   `frontend/dist`).
3. Free-tier notes: Render's free web services spin down after inactivity, so
   the first request after idling has a cold-start delay (10-60s) — expected,
   not a bug. SQLite (`backend/trackpulse.db`) lives on Render's ephemeral
   filesystem, so session history resets on redeploy/restart — acceptable for a
   demo, would need a real database for persistence beyond that.

`backend/app/inference.py` resolves the model path robustly (checks
`<repo_root>/models/`, then `<backend>/models/`, with an env var override
`TRACKPULSE_MODEL_PATH` available) rather than assuming a specific host
platform's checkout layout.

## Project structure

```text
backend/app/                FastAPI service, ONNX inference, deterministic trend/suggestion logic,
                               video frame extraction (video.py)
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
