# exp05 -- Data Moat: Substantial Racing-Domain Dataset Expansion

## Summary

exp05 expanded the racing-domain training pool with 561 new images (sourced and hand-labeled from Wikimedia Commons, zero SHA-256 overlap with any prior pool) and a new 36-image eval addition (part of a combined 103-image v3 eval set), then fine-tuned two variants from exp02's production checkpoint. The best variant (optionA, aggressive per-class oversampling) modestly improves DRY recall and overall macro-F1 on the two larger/more established eval axes (v2 67-image set and the new v3 103-image set) at a real, measured cost to WET recall and a small cost to DAMP recall, relative to exp02. This is a similar shape of trade-off to exp03's rejected result, but smaller in magnitude and with a genuinely larger, more diverse racing-domain data moat behind it. Recommendation: do not promote to production as-is -- the trade-off is real and not a clear net win against exp02 on the metric the project has consistently protected (WET recall). exp02 remains production. Full reasoning below.

## Dataset moat: what was added

Sourcing method: Wikimedia Commons API (action=query, list=search with filetype:bitmap + list=categorymembers for known racing-event categories), harvested candidate metadata (title, imageinfo, extmetadata for license/artist), downloaded + resized (max 1024px, JPEG q85) + SHA-256 hash-deduplicated against all 4 existing pools before accepting, license filtered to CC-BY/CC0/Public-Domain-only. Two sourcing rounds:

1. Round 2 (search_commons2.py + download_candidates2/3.py): broad DAMP/DRY/WET queries + categories (fixed after round 1's unfiltered full-text search returned mostly PDF/document-scan noise) -- 489 candidates, 450 accepted after license/dedup.
2. Round 3 (search_commons3_damp.py + download_candidates4.py): rain-affected-event categories across eras/series specifically to find more DAMP/transitional frames (2016 Monaco GP, 2011-2012 Canadian GP, 2007/2008/2014/2019 Japanese GP, plus the 2008 British GP already found in round 2) -- 976 candidates, 803 accepted after license/dedup.

Visual triage: all 1253 staged candidates were reviewed via contact sheets (25-image grids) plus individual full-resolution spot-checks for every DAMP candidate and for a targeted quality re-check of the broad DRY sweep (caught 21 unusable frames -- extreme motion-blur panning shots with no identifiable track texture, and non-track content like trophies/portraits/museum displays that had slipped past title-keyword filtering).

Final curated, quality-checked counts: 10 DAMP, 106 WET, 481 DRY = 597 images.

DAMP sourcing -- reported honestly: of roughly 1700 total candidate titles scanned across both sourcing rounds (1246 + 976, with heavy overlap in motorsport-adjacent categories), only 10 were judged genuinely damp/transitional after individual full-resolution verification (matte-grey track, visible moisture, no heavy spray, no standing-water glare -- distinct from both the bright dry-sun look and the dark saturated-spray wet look). This is consistent with every prior round's finding (exp02: ~350/2100 viable after triage for a broader DRY+DAMP+WET pool; exp03: only 50 DAMP images existed total across the whole project before this round). We did not force weak or ambiguous labels to hit an arbitrary target -- DAMP remains the hardest class to source, full stop.

### Split into eval additions vs. training pool

| | DAMP | WET | DRY | Total |
|---|---:|---:|---:|---:|
| New eval additions (racing_spotcheck_v3/images_new/) | 4 | 16 | 16 | 36 |
| New training pool (racing_train_pool_v3/images/) | 6 | 90 | 465 | 561 |

data/racing_spotcheck_v3/ = the unchanged 67-image v2 set (including its damp_holdout/ 18-image exp03 subset, copied verbatim) plus the 36 new additions = 103-image v3 eval set, ground truth in data/racing_spotcheck_v3/ground_truth_manifest.json.

### Zero-leakage verification (exact method)

SHA-256 hashed every file in all 8 relevant image directories (spotcheck v1 images/+images2/, spotcheck v2 images/+images2/, train_pool v1, train_pool v2, train_pool v3, spotcheck_v3 images_new/) and computed pairwise set-intersections between both NEW pools (train_pool_v3, spotcheck_v3 new-additions) and every other pool, including each other. Result: zero overlapping hashes in every pairwise check (experiments/exp05_data_moat/scratch/final_leakage_reverify.py, run against the final on-disk file layout, not just staging). Total unique racing-domain images across the whole project after exp05: 938 (up from 341 pre-exp05).

## Training

Initialized from experiments/exp02_racing_v2/checkpoints/best_model.pth (current production model), not exp03 or exp04 (rejected/experimental). Combined dataset = RSCD train (2582) + racing_train_pool v1 (115) + racing_train_pool_v2 (177) + new racing_train_pool_v3 (561). Same fine-tune scope (classifier head + last 3 MobileNetV3-Small feature blocks unfrozen), same LR (3e-4 AdamW, cosine schedule, 12 epochs max, patience 4) as exp01-exp04, for direct comparability.

v3's raw class balance is heavily DRY-skewed (465/561 = 83% DRY, 6 DAMP/90 WET in the train split) -- the opposite skew from exp02/exp03's pools, because DRY diversity (new circuits/eras/series) was comparatively easy to source this round while DAMP stayed hard. This required recalculating oversample factors from scratch rather than reusing exp02/exp03's.

Two strategies trained and compared on the frozen eval sets (following exp04's precedent of comparing oversampling vs. class-weighting rather than assuming one wins):

- optionA (oversample): DRY 2x, DAMP 15x, WET 4x -- DAMP kept aggressively oversampled since it's scarcest/highest-priority; DRY given a lighter factor than exp02 used, specifically to avoid swamping batches with easy DRY examples (part of what caused exp03's documented DRY-regression failure mode). Best val macro-F1 = 0.7516 (epoch 8/12).
- optionB (class-weighted loss + light oversample): DRY 1x, DAMP 4x, WET 2x + inverse-frequency class weights in CrossEntropyLoss, testing exp03's unexplored "class weighting instead of heavy oversampling" direction. Best val macro-F1 = 0.7459 (epoch 1/12, early-stopped at epoch 5).

optionA had the higher validation macro-F1 and, per full eval below, the stronger overall result on the eval sets -- selected as exp05's headline checkpoint.

## Evaluation -- full comparison table

All numbers are real, measured, single frozen-checkpoint runs (no tuning against any eval set). exp02 was re-evaluated on the new v3 set for a fair comparison (v3 didn't exist when exp02 was originally scored).

### Original 49-image eval set (n=44 quantitative, 5 AMBIGUOUS excluded; DAMP has 0 support here -- this set predates DAMP labeling)

| Model | Accuracy | Macro-F1 | DRY recall | WET recall | False-WET rate | False-DRY rate |
|---|---:|---:|---:|---:|---:|---:|
| exp02 (production) | 72.7% | 0.516 | 63.3% | 92.9% | 20.0% | 7.1% |
| exp05 optionA | 75.0% | 0.493 | 80.0% | 64.3% | 13.3% | 35.7% |
| exp05 optionB | 65.9% | 0.469 | 63.3% | 71.4% | 16.7% | 28.6% |

### v2 67-image eval set (n=62 quantitative, 5 AMBIGUOUS; DAMP support n=18, the exp03 holdout)

| Model | Accuracy | Macro-F1 | DRY recall | DAMP recall | DAMP precision | WET recall |
|---|---:|---:|---:|---:|---:|---:|
| exp02 (production) | 75.8% | 0.762 | 63.3% | 83.3% | 75.0% | 92.9% |
| exp05 optionA | 77.4% | 0.766 | 80.0% | 83.3% | 88.2% | 64.3% |
| exp05 optionB | 72.6% | 0.723 | 63.3% | 88.9% | 72.7% | 71.4% |

### v3 103-image eval set (NEW, exp05; n=98 quantitative, 5 AMBIGUOUS; DAMP support n=22)

| Model | Accuracy | Macro-F1 | DRY recall | DAMP recall | DAMP precision | WET recall | False-WET | False-DRY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp02 (production) | 66.3% | 0.655 | 67.4% | 72.7% | 57.1% | 60.0% | 17.4% | 23.3% |
| exp05 optionA | 72.4% | 0.708 | 82.6% | 68.2% | 68.2% | 60.0% | 10.9% | 26.7% |
| exp05 optionB | 70.4% | 0.698 | 71.7% | 77.3% | 63.0% | 63.3% | 13.0% | 26.7% |

### RSCD test set (n=305, regression check)

| Model | Accuracy | Macro-F1 | DRY recall | DAMP recall | WET recall |
|---|---:|---:|---:|---:|---:|
| exp02 (production) | 74.8% | 0.724 | 81.8% | 61.3% | 75.3% |
| exp05 optionA | 73.8% | 0.711 | 82.4% | 55.0% | 76.6% |
| exp05 optionB | 76.4% | 0.739 | 85.8% | 58.8% | 76.6% |

## Interpretation

optionA is a real, measured improvement on DRY performance and overall macro-F1 on both the more-established v2 set and the new, larger, more diverse v3 set -- DRY recall goes from 63-67% (exp02) to 80-83% (optionA), and macro-F1 improves on v2 (0.762 to 0.766) and more substantially on v3 (0.655 to 0.708). This is the direct, expected payoff of the new DRY-diversity data (465 new DRY images spanning Spa-Francorchamps, additional Monaco/Canadian/Japanese GP eras, 1980s historic F1, more camera angles).

But it comes at a real cost to WET recall: 92.9% to 64.3% on the v2 set -- a 28.6-point drop, larger than exp03's regression on the axis it did protect. On the new v3 set the WET-recall picture is flatter (60.0% to 60.0%, i.e. no change), but the v2-set drop alone is a serious regression on the metric this project has treated as load-bearing since exp00 (WET recall was 83-93% in every prior promoted/production round). DAMP recall also slips modestly on v3 (72.7% to 68.2%), though DAMP precision improves (57.1% to 68.2%) -- a genuinely mixed DAMP picture, not a clean loss.

This is structurally the same shape of trade-off exp03 was rejected for (chasing one axis at the expense of WET recall), just with the axis being pushed being DRY/overall-accuracy this time instead of DAMP. optionB (class-weighting) does better at preserving WET recall (71.4%/63.3% instead of 64.3%/60.0%) and has the best DAMP recall on v3 (77.3%), and even edges out both other models on the RSCD regression check (macro-F1 0.739) -- but its overall v2/v3 macro-F1 and DRY gains are smaller than optionA's, and its own WET recall is still below exp02's on v2 (71.4% vs 92.9%).

## Recommendation: do not promote exp05 to production

exp02 remains production (models/trackpulse_classifier.onnx). Neither exp05 variant clears the bar the project has consistently applied since exp03: a new checkpoint needs to win or hold steady on the established WET-recall axis to be promoted, not just win on a new metric or a new eval set. Both optionA and optionB show a real WET-recall cost on the v2 eval set specifically (the longest-standing, most-scrutinized eval axis in the project). This is a legitimate, well-characterized negative/mixed result -- consistent with exp03's precedent -- not a failure to hide.

What this round did establish, and why it's still valuable:

1. The racing-domain data moat is genuinely larger and more diverse: 938 unique images project-wide (up from 341), spanning meaningfully more circuits/eras/series (Spa-Francorchamps, additional Monaco/Canadian/ Japanese GP years, 1980s historic F1) -- a real asset for a future round that specifically targets the WET-recall trade-off rather than accepting it.
2. DAMP sourcing difficulty is now confirmed across a fourth independent round (exp01 to exp02 to exp03 to exp05), each time landing on the same conclusion: genuine damp/transitional racing imagery is intrinsically rare in publicly available CC-licensed sources, not a search-effort problem. Future rounds should budget for this rather than assume more search time closes the gap.
3. The oversampling-vs-class-weighting question (raised as an open direction in exp03) now has a second data point: neither cleanly dominates, and both show a version of the "improve axis X at cost to WET recall" pattern -- suggesting the fix for this project's remaining trade-off is more likely to be architectural (e.g., a WET-recall-weighted loss term, or an oversample schedule that never lets WET's relative share of the combined train set shrink) rather than "try another rebalancing ratio."

## Comparison across all experiments (exp00-exp05)

| | RSCD test (n=305) | Racing eval (varies by round, held out throughout) |
|---|---|---|
| exp00 (frozen baseline) | acc 77.7%, WET recall 83.1% | acc 34.1% (n=49), DRY recall 3.3%, false-WET 96.7% |
| exp01 (+115 racing imgs) | acc 75.4%, WET recall 67.5% | acc 50.0% (n=49), DRY recall 30.0%, false-WET 60.0% |
| exp02 (+177 racing imgs, production) | acc 74.8%, WET recall 75.3% | acc 72.7% (n=49) / 75.8% (n=67 v2) / 66.3% (n=103 v3, re-measured this round), DRY recall 63.3-67.4% |
| exp03 (DAMP holdout, not promoted) | acc 76.4% | acc 61.4% (n=44 v2 subset); DAMP recall 83.3% (n=18, first measurement) |
| exp04 (rebalance variants, research artifacts) | not re-measured here (see exp04's own writeup) | -- |
| exp05 optionA (data moat, not promoted) | acc 73.8%, WET recall 76.6% | acc 75.0% (n=49) / 77.4% (n=67 v2) / 72.4% (n=103 v3, new), DRY recall 80.0-82.6%, WET recall drops to 64.3% (v2) |
| exp05 optionB (data moat, not promoted) | acc 76.4%, WET recall 76.6% | acc 65.9% (n=49) / 72.6% (n=67 v2) / 70.4% (n=103 v3), DAMP recall 77.3-88.9% (best of exp05), WET recall 71.4% (v2) |

## Key artifacts

- Training pool: data/racing_train_pool_v3/images/ (561 images), data/racing_train_pool_v3/training_manifest_raw.json, racing_train_split.json / racing_val_split.json
- Eval set: data/racing_spotcheck_v3/ (103 images total: unchanged v2's 67 in images/+images2/+damp_holdout/, plus 36 new in images_new/), data/racing_spotcheck_v3/ground_truth_manifest.json
- Leakage verification: experiments/exp05_data_moat/scratch/final_leakage_reverify.py (script + confirmed-zero-overlap console output)
- Training scripts: scripts/s_exp05_finetune.py (optionA/optionB), scripts/s_exp05_eval_all.py, scripts/s_exp05_export_onnx.py
- Checkpoints: experiments/exp05_data_moat/checkpoints_optionA/best_model.pth, checkpoints_optionB/best_model.pth
- Eval results (raw JSON, full protocol per set): experiments/exp05_data_moat/eval_results_optionA.json, eval_results_optionB.json, eval_results_exp02_baseline.json
- ONNX export: models/trackpulse_classifier_v6_exp05.onnx (optionA, NOT production -- models/trackpulse_classifier.onnx remains exp02)
- This report: experiments/exp05_data_moat/REPORT.md
- Progress log with running numbers: experiments/exp05_data_moat/PROGRESS.md
