"""
Step 3: confusion matrix + metrics on the racing contrast set (excluding AMBIGUOUS).
"""
import json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score

OUT_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'
CLASSES = ['DRY', 'DAMP', 'WET']

with open(f'{OUT_DIR}/contrast_set_inference_results.json', encoding='utf-8') as f:
    results = json.load(f)

labeled = [r for r in results if r['ground_truth'] != 'AMBIGUOUS']
ambiguous = [r for r in results if r['ground_truth'] == 'AMBIGUOUS']

print(f"total images: {len(results)}  labeled (non-ambiguous): {len(labeled)}  ambiguous: {len(ambiguous)}")

y_true = [CLASSES.index(r['ground_truth']) for r in labeled]
y_pred = [CLASSES.index(r['predicted_class']) for r in labeled]

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0,1,2], zero_division=0)
prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)

print("\nconfusion matrix (rows=true, cols=pred), order", CLASSES)
print(cm)
print(f"\naccuracy: {acc:.4f}")
print(f"macro F1: {macro_f1:.4f}")
for i, c in enumerate(CLASSES):
    print(f"{c}: precision={prec[i]:.4f} recall={rec[i]:.4f} f1={f1[i]:.4f} support={support[i]}")

# false-wet rate: dry or damp ground truth predicted as WET
dry_damp_gt = [r for r in labeled if r['ground_truth'] in ('DRY', 'DAMP')]
false_wet = [r for r in dry_damp_gt if r['predicted_class'] == 'WET']
false_wet_rate = len(false_wet) / len(dry_damp_gt) if dry_damp_gt else None
print(f"\nfalse-WET rate (dry/damp GT predicted WET): {len(false_wet)}/{len(dry_damp_gt)} = {false_wet_rate:.4f}" if false_wet_rate is not None else "N/A")

# false-dry rate: wet or damp ground truth predicted as DRY
wet_damp_gt = [r for r in labeled if r['ground_truth'] in ('WET', 'DAMP')]
false_dry = [r for r in wet_damp_gt if r['predicted_class'] == 'DRY']
false_dry_rate = len(false_dry) / len(wet_damp_gt) if wet_damp_gt else None
print(f"false-DRY rate (wet/damp GT predicted DRY): {len(false_dry)}/{len(wet_damp_gt)} = {false_dry_rate:.4f}" if false_dry_rate is not None else "N/A (no wet/damp ground truth samples predicted dry to divide by, or zero denominator)")

# confidence stats
all_conf = np.array([r['confidence'] for r in labeled])
correct_mask = np.array([r['predicted_class'] == r['ground_truth'] for r in labeled])
mean_conf_all = all_conf.mean()
mean_conf_correct = all_conf[correct_mask].mean() if correct_mask.sum() > 0 else None
mean_conf_incorrect = all_conf[~correct_mask].mean() if (~correct_mask).sum() > 0 else None

print(f"\nmean confidence (all labeled): {mean_conf_all:.4f}")
print(f"mean confidence (correct predictions, n={correct_mask.sum()}): {mean_conf_correct}")
print(f"mean confidence (incorrect predictions, n={(~correct_mask).sum()}): {mean_conf_incorrect}")

# ambiguous descriptive stats
print(f"\n=== AMBIGUOUS images (n={len(ambiguous)}) - descriptive only, not in metrics ===")
for r in ambiguous:
    print(f"  {r['image_id']}: pred={r['predicted_class']} conf={r['confidence']:.3f} P=[{r['P_DRY']:.3f},{r['P_DAMP']:.3f},{r['P_WET']:.3f}]  note={r['note']}")

metrics_out = {
    'n_total': len(results),
    'n_labeled': len(labeled),
    'n_ambiguous': len(ambiguous),
    'confusion_matrix': cm.tolist(),
    'classes_order': CLASSES,
    'accuracy': float(acc),
    'macro_f1': float(macro_f1),
    'per_class': {CLASSES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)},
    'false_wet_rate': false_wet_rate,
    'false_wet_count': len(false_wet),
    'false_wet_denominator': len(dry_damp_gt),
    'false_dry_rate': false_dry_rate,
    'false_dry_count': len(false_dry),
    'false_dry_denominator': len(wet_damp_gt),
    'mean_confidence_all': float(mean_conf_all),
    'mean_confidence_correct': float(mean_conf_correct) if mean_conf_correct is not None else None,
    'mean_confidence_incorrect': float(mean_conf_incorrect) if mean_conf_incorrect is not None else None,
    'ambiguous_descriptive': [{'image_id': r['image_id'], 'predicted_class': r['predicted_class'], 'confidence': r['confidence'],
                                 'P_DRY': r['P_DRY'], 'P_DAMP': r['P_DAMP'], 'P_WET': r['P_WET'], 'note': r['note']} for r in ambiguous],
}
with open(f'{OUT_DIR}/contrast_set_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, indent=2)
print(f"\nsaved {OUT_DIR}/contrast_set_metrics.json")
