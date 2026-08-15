# exp06: WET-recall-safe DRY improvement search — PROGRESS

## Status: IN PROGRESS

## Step 1 — Bugfix verification: DONE
Read `scripts/s_exp05_finetune.py` in full. Confirmed the fix is real and correctly applied:
- Line 204: `loss_weights = weights if USE_EXTRA_CLASS_WEIGHT else None`
- Line 225: `criterion = nn.CrossEntropyLoss(weight=loss_weights)`
optionA (USE_EXTRA_CLASS_WEIGHT=False) now genuinely receives `weight=None`. No other bugs found on close read of data loading, oversampling, checkpoint init, scheduler, or logging.

`scripts/s_exp05_eval_all.py` also read in full — logic looks correct (recall/precision computed via sklearn `precision_recall_fscore_support` with fixed label order `[0,1,2]` = `[DRY,DAMP,WET]`, matches `CLASSES` list). Reused as-is.

## Step 2 — Grid design: DONE
Built `scripts/s_exp06_finetune.py`, a parameterized version of the exp05 script (same data pipeline: RSCD train + racing v1+v2+v3 pools = 938 unique racing images, init from exp02 checkpoint, same LR/epochs/early-stop). Usage:
`python s_exp06_finetune.py <variant> <dry_f> <damp_f> <wet_f> <use_class_weight:0|1>`

Planned variants:
| variant | DRY | DAMP | WET | class-weight | status |
|---|---|---|---|---|---|
| pure_oversample | 2 | 15 | 4 | No | pending |
| light_damp | 2 | 7 | 4 | No | pending |
| weight_only | 1 | 1 | 1 | Yes | pending |
| wet_protective (optional) | 2 | 6 | 8 | No | pending |

## Step 3 — Gate eval (v2 67-image set): IN PROGRESS

Training done so far:
- pure_oversample (DRY2/DAMP15/WET4, no class weight): best_val_macro_f1=0.7469, epochs_run=10 (early stop)

Gate eval results (v2 67-image set, n=62 quantitative):
| variant | DRY recall | DAMP recall | WET recall | notes |
|---|---|---|---|---|
| exp02 (production baseline) | 0.633 | 0.833 | 0.929 | reference |
| exp05-optionA (buggy, rejected) | 0.800 | — | 0.643 | from original report, had extra class-weight bug stacked |
| **pure_oversample (fixed, DRY2/DAMP15/WET4, no weight)** | **0.800** | **0.778** | **0.714** | GATE FAIL — WET recall still ~21pt below exp02 baseline (threshold: stay above ~0.83) |

**Key finding so far**: pure_oversample gets the SAME DRY recall (80%) as the buggy exp05-optionA, but WET recall is only modestly better (71.4% vs 64.3%) — still a severe regression, well outside the 10-point gate tolerance. This suggests the WET collapse in exp05-optionA was NOT primarily caused by the class-weighting bug — it looks like the oversampling itself (esp. DAMP 15x, or general DRY2/WET4/DAMP15 recipe) is the main driver of WET recall collapse, likely via the same DRY/DAMP/WET cross-class-confusion mechanism exp03 first flagged (just manifesting on WET here instead of DRY).

pure_oversample FAILS the Step 3 gate.

| light_damp (DRY2/DAMP7/WET4, no weight) | 0.700 | 0.833 | 0.571 | GATE FAIL — WET recall even worse than pure_oversample, and DRY gain is smaller. Lighter DAMP oversampling did NOT rescue WET recall; if anything it's worse. |

light_damp FAILS the Step 3 gate, more severely than pure_oversample. This weakens the hypothesis that DAMP-oversampling-intensity alone drives the WET collapse — reducing it made things worse, not better, on this run.

| weight_only (DRY1/DAMP1/WET1, class-weighted, no oversample) | 0.733 | 0.722 | 0.643 | GATE FAIL — WET recall matches the buggy exp05-optionA exactly (0.643), DRY gain smallest of all 3 variants tried so far. |

weight_only FAILS the Step 3 gate too. All three tried strategies (pure oversample, light-DAMP oversample, pure class-weighting) show the SAME qualitative failure: WET recall collapses from exp02's 92.9% baseline to somewhere in the 57-71% range, while DRY recall improves modestly (70-80%). This is a consistent, strategy-independent pattern — strong evidence the WET/DRY trade-off is not an artifact of any one oversample/weighting recipe.

Launching optional 4th variant: wet_protective (DRY2/DAMP6/WET8, no weight) to test whether directly boosting WET's own oversample factor (rather than relying on DAMP factor tuning) can rescue WET recall. Result pending.

**No variant has passed the gate yet.** If wet_protective also fails, Step 4 (full eval) will be skipped and the honest conclusion is that the WET/DRY trade-off persists across all training-strategy variants tried, at least with this fine-tune recipe (last-3-blocks unfrozen, same LR/epochs/schedule as exp01-05) — pointing toward a more fundamental limitation (architecture, frozen-backbone scope, or underlying data/feature overlap between WET and DRY/DAMP classes) rather than a training-strategy fix.

## Step 4 — Full eval: NOT STARTED (pending a variant that passes the gate)

## Step 5 — ONNX export: NOT STARTED

## Step 6 — Final report: NOT STARTED
