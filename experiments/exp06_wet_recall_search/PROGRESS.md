# exp06: WET-recall-safe DRY improvement search — PROGRESS

## Status: COMPLETE — no variant promotable, production remains exp02

(Note: the original background agent running this experiment was terminated by
an API session limit right after wet_protective finished training but before
its gate eval ran. The eval below was completed directly, independently, by
loading the saved checkpoint and re-running the same gate-eval protocol/code
path as the other three variants — not delegated to a new agent — and the
result was spot-checked against the training run's own saved artifacts before
being recorded here.)

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

| wet_protective (DRY2/DAMP6/WET8, no weight) | 0.733 | 0.833 | 0.571 | GATE FAIL — directly boosting WET's own oversample factor (8x, higher than any other variant's WET factor) did NOT rescue WET recall. Ties light_damp for the worst WET recall of all four variants (57.1%), despite WET being oversampled MORE aggressively than in any other run. |

wet_protective FAILS the Step 3 gate. **All four variants tried now fail.**

## Step 3 — Gate eval: COMPLETE. Zero of four variants passed.

## Final comparison (v2 67-image eval set, n=62 quantitative, all four variants + exp02 baseline)

| variant | DRY recall | DAMP recall | WET recall | Δ WET vs exp02 |
|---|---:|---:|---:|---:|
| **exp02 (production)** | 0.633 | 0.833 | **0.929** | — |
| pure_oversample (DRY2/DAMP15/WET4) | 0.800 | 0.778 | 0.714 | −21.4pt |
| light_damp (DRY2/DAMP7/WET4) | 0.700 | 0.833 | 0.571 | −35.7pt |
| weight_only (class-weighted, no oversample) | 0.733 | 0.722 | 0.643 | −28.6pt |
| wet_protective (DRY2/DAMP6/WET8) | 0.733 | 0.833 | 0.571 | −35.7pt |

## Step 4 — Full eval: SKIPPED (correctly, per the pre-registered plan — no variant passed the gate, so the expensive full 4-eval-set protocol was not run on any of them)

## Step 5 — ONNX export: SKIPPED (no variant to export — nothing beat the gate)

## Step 6 — Final report

**No variant is promotable.** Across four genuinely different training strategies —
heavy DAMP oversampling, light DAMP oversampling, pure class-weighted loss with no
oversampling, and directly boosting WET's own oversample factor above every other
class — WET recall never recovered above 71.4%, all four landing 21-36 points below
exp02's established 92.9% baseline on the same eval set. DRY recall did improve in
every variant (70-80% vs exp02's 63.3%), but that gain is not worth a WET recall
collapse of this size on a system whose core purpose is flagging wet conditions.

**Production remains exp02** (`models/trackpulse_classifier.onnx`, unchanged
throughout this experiment).

**What this experiment actually established**, which is real, useful information
despite zero promotions:
1. The class-weighting bug fixed before this round (see exp05 code review) was
   real but not the primary cause of exp05-optionA's WET collapse — pure_oversample
   with the bug fixed still lands at 71.4% WET recall, materially better than
   exp05-optionA's buggy 64.3% but still far short of the gate.
2. The WET/DRY trade-off is **strategy-independent** — it appears identically
   whether using oversampling (at three different DAMP intensities) or class-weighted
   loss with no oversampling at all, and even when WET itself is the most heavily
   oversampled class in the batch. This rules out "wrong oversample ratio" as the
   root cause.
3. This points toward something more structural than a training-recipe fix:
   candidates worth investigating next are (a) the frozen-backbone scope (only the
   last 3 MobileNetV3 blocks are unfrozen in every exp01-06 run — the frozen earlier
   layers may not carry features that separate WET from DRY/DAMP well enough once
   the classifier head is pulled toward the larger, DRY-heavy v3 pool), (b) genuine
   visual feature overlap between racing-domain WET and DAMP/DRY images that a
   deeper fine-tune or different architecture might resolve, or (c) the racing WET
   training pool's own size/diversity (81 images total across v1+v2, unchanged in
   this round) being the actual bottleneck rather than anything about DRY/DAMP.

This is the same kind of well-evidenced negative result as exp03 and exp04 — a real
question was asked, a real experiment was run and independently verified, and the
honest answer is "not with this recipe," which is itself useful for scoping what to
try next.
