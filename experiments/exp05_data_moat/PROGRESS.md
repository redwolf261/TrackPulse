# exp05 Data Moat — Progress

Status: STARTED
Started: 2026-08-15

## Step 1: Sourcing
Status: in progress

- Round 1 (broad full-text search, `search_commons.py`): 1246 candidates, mostly
  document/PDF/OCR noise from unfiltered full-text search. Abandoned this
  approach in favor of `filetype:bitmap` + mime-check.
- Round 2 (`search_commons2.py`, filetype:bitmap + mime filter, DAMP + DRY + WET
  queries/categories): 489 candidates. Downloaded/license-checked/deduped
  (`download_candidates2.py` -> `download_candidates3.py` after fixing a
  license-substring bug and adding retry/backoff for rate-limited downloads):
  450 accepted images staged (0 SHA-256 collisions against the 341-hash existing
  pool from racing_spotcheck + racing_spotcheck_v2 + racing_train_pool +
  racing_train_pool_v2).
- Visual triage of all 450 staged images via contact sheets
  (`scripts/exp05.../scratch/sheets3/`, plus a closer per-image pass on the
  2008 British GP rain cluster in `sheets_rain/`): most of round-2's yield was
  museum/showroom/pit-lane/non-track noise (car close-ups, trophies, historical
  document scans that slipped past title filters). Verified usable subset:
  132 DRY, 63 WET, only 6 DAMP (round 2 alone).
- DAMP is confirmed the hard case again, consistent with every prior round.
  Round 3 (`search_commons3_damp.py`) pulled 976 more candidates from known
  rain-affected race categories across many eras/series specifically to find
  more transitional/drying frames. Downloading now (`download_candidates4.py`).

## Step 1: Sourcing — FINAL RESULT
Status: complete

- Round 3 (`search_commons3_damp.py`, rain-affected-event categories across many
  eras/series): 976 candidates, downloaded/filtered -> 803 new accepted images
  (`download_candidates4.py`), combined with round 2's 450 = 1253 total staged.
- Full visual triage of all 1253 staged images via contact sheets (sheets3/,
  sheets4/, sheets_rain/, sheets_verify/) plus individual full-resolution
  spot-checks for DAMP candidates and blur/non-track quality issues.
- Found additional rain-affected events beyond 2008 British GP: 2016 Monaco GP
  (wet start, dried out mid-race), 2011-2012 Canadian GP (best DAMP source),
  2007/2008/2014/2019 Japanese GP (Fuji/Suzuka, mixed wet/dry).
- FINAL curated + quality-filtered set: 10 DAMP, 106 WET, 481 DRY = 597 images
  (dropped 21 more for extreme motion-blur/non-track content found during a
  targeted re-check of the broad-sweep DRY range).
- DAMP confirmed hard again: of ~1700 total candidates scanned across both
  sourcing rounds, only 10 were judged genuinely damp/transitional (matte-grey,
  no spray, no standing water) after individual full-resolution verification.
  Consistent with exp01-exp04's repeated finding. Not forcing weak labels to
  hit an arbitrary target.

## Step 2: Training pool build
Status: complete

- Split curated set: 36 new eval images (4 DAMP, 16 WET, 16 DRY) to
  `data/racing_spotcheck_v3/images_new/` + unchanged v2 67-image set copied to
  `data/racing_spotcheck_v3/{images,images2}/` = 103-image v3 eval set
  (`data/racing_spotcheck_v3/ground_truth_manifest.json`).
- Remaining 561 images (6 DAMP, 90 WET, 465 DRY) to
  `data/racing_train_pool_v3/images/`, with 85/15 stratified train/val split
  matching v2's protocol (`racing_train_split.json` / `racing_val_split.json`).
- LEAKAGE VERIFICATION (method): SHA-256 hashed every file in all 8 image
  directories (spotcheck v1 images+images2, spotcheck v2 images+images2,
  train_pool v1, train_pool v2, train_pool v3, spotcheck_v3 new-additions) and
  did pairwise set-intersection checks between the two NEW pools and all other
  pools including each other. Result: **zero overlap confirmed**
  (`experiments/exp05_data_moat/scratch/final_leakage_reverify.py` output).
- **Total unique racing-domain images project-wide: 938** (up from exp02's
  ~292 train + 67 eval baseline mentioned in the task -- exact count depends on
  what's included; raw sum of all 8 dirs' unique hashes = 938).

## Step 3: Fine-tune
Status: complete

- Two strategies trained (`scripts/s_exp05_finetune.py optionA|optionB`), both
  initialized from exp02's checkpoint (production model), same fine-tune scope
  (last 3 MobileNetV3 feature blocks + classifier head) and LR/schedule as
  exp01-exp04:
  - optionA: aggressive per-class oversample (DRY 2x, DAMP 15x, WET 4x),
    recalculated for v3's DRY-heavy raw balance. Best val macro-F1 = 0.7516
    (epoch 8/12).
  - optionB: light oversample (DRY 1x, DAMP 4x, WET 2x) + inverse-frequency
    class-weighted CrossEntropyLoss, following exp04's "class weighting
    instead of heavy oversampling" direction. Best val macro-F1 = 0.7459
    (epoch 1/12, early-stopped at epoch 5).
- Checkpoints: `experiments/exp05_data_moat/checkpoints_optionA/best_model.pth`,
  `checkpoints_optionB/best_model.pth`.

## Step 4: Evaluation
Status: complete

- Full 4-eval-set protocol run for exp02 (re-baselined on the new v3 set),
  optionA, and optionB via `scripts/s_exp05_eval_all.py`.
- optionA selected as the winner: best macro-F1 on both the v2 67-image set
  (0.766) and the new v3 103-image set (0.708), meaningfully better DRY
  recall (82.6% vs exp02's 67.4% on v3) at a real but smaller cost to WET
  recall (60% vs exp02's 60% on v3 -- actually flat) and DAMP recall (68.2%
  vs exp02's 72.7% on v3, a real 4.5-point give-back but not a collapse).
  Full numbers in REPORT.md.
- Found and fixed a real bug during evaluation: v2's ground_truth_manifest.json
  references an 18-image `damp_holdout/` subdirectory (exp03's held-out DAMP
  set) that wasn't copied when building v3 -- caused 18 missing-file warnings
  on the first eval run. Fixed by copying `damp_holdout/` into
  `data/racing_spotcheck_v3/` unchanged; re-ran eval afterward. v3's DAMP
  support is correctly n=22 (18 exp03 holdout + 4 new exp05 additions).

## Step 5: ONNX export
Status: complete

- Exported optionA's checkpoint to `models/trackpulse_classifier_v6_exp05.onnx`
  via `scripts/s_exp05_export_onnx.py`. PyTorch vs ONNX equivalence verified
  on 20 random inputs: max abs logit diff = 8.11e-06 (well under 1e-3
  tolerance). Additional sanity check: ONNX Runtime inference (manual
  preprocessing, no torchvision) on a subset of the v2 eval set produces
  results consistent with the PyTorch-side numbers.
- Production `models/trackpulse_classifier.onnx` (exp02) was NOT touched.

## Step 6: Report
Status: complete -- see REPORT.md
