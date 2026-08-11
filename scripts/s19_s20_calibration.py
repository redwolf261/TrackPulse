"""
Section 19: probability calibration (Brier score, ECE, reliability diagram).
Section 20: confusion matrix PNG + remaining artifact bookkeeping.
Uses the saved test_predictions_with_probs.csv from training (test set touched only once, at eval time).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'
CLASSES = ['DRY', 'DAMP', 'WET']

df = pd.read_csv(f'{EXP_DIR}/test_predictions_with_probs.csv')
print("loaded", len(df), "test predictions")

# ---- multiclass Brier score (mean squared error between one-hot true and predicted probs) ----
y_true_onehot = np.zeros((len(df), 3))
for i, lbl in enumerate(df['true_label']):
    y_true_onehot[i, CLASSES.index(lbl)] = 1.0
probs = df[[f'prob_{c}' for c in CLASSES]].values

brier_score = np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))
print(f"\nmulticlass Brier score (mean sum-squared-error over 3 classes, lower=better, 0=perfect): {brier_score:.4f}")

# ---- ECE (expected calibration error), using max predicted prob (confidence) vs correctness, 10 bins ----
confidences = probs.max(axis=1)
predictions = probs.argmax(axis=1)
true_idx = np.array([CLASSES.index(l) for l in df['true_label']])
correct = (predictions == true_idx).astype(float)

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
ece = 0.0
bin_stats = []
for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i+1]
    if i == n_bins - 1:
        mask = (confidences >= lo) & (confidences <= hi)
    else:
        mask = (confidences >= lo) & (confidences < hi)
    n_in_bin = mask.sum()
    if n_in_bin > 0:
        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (n_in_bin / len(df)) * abs(avg_conf - avg_acc)
        bin_stats.append({'bin_lo': float(lo), 'bin_hi': float(hi), 'n': int(n_in_bin), 'avg_confidence': float(avg_conf), 'avg_accuracy': float(avg_acc)})
    else:
        bin_stats.append({'bin_lo': float(lo), 'bin_hi': float(hi), 'n': 0, 'avg_confidence': None, 'avg_accuracy': None})

print(f"\nECE (10-bin, |confidence - accuracy| weighted by bin size): {ece:.4f}")
print("overall test accuracy (for reference):", correct.mean())

for b in bin_stats:
    if b['n'] > 0:
        print(f"  bin [{b['bin_lo']:.1f},{b['bin_hi']:.1f}): n={b['n']} avg_conf={b['avg_confidence']:.3f} avg_acc={b['avg_accuracy']:.3f}")

# ---- reliability diagram ----
fig, ax = plt.subplots(figsize=(6, 6))
xs = [b['bin_lo'] + 0.05 for b in bin_stats if b['n'] > 0]
accs = [b['avg_accuracy'] for b in bin_stats if b['n'] > 0]
confs = [b['avg_confidence'] for b in bin_stats if b['n'] > 0]
ns = [b['n'] for b in bin_stats if b['n'] > 0]
ax.plot([0, 1], [0, 1], 'k--', label='perfect calibration')
ax.bar(xs, accs, width=0.08, alpha=0.7, label='observed accuracy', color='steelblue', edgecolor='black')
ax.scatter(confs, accs, color='red', zorder=5, label='(avg confidence, avg accuracy) per bin')
ax.set_xlabel('Confidence (max predicted probability)')
ax.set_ylabel('Accuracy')
ax.set_title(f'Reliability Diagram (ECE={ece:.4f}, Brier={brier_score:.4f})\nTest set n={len(df)}, MobileNetV3-Small on RSCD 3-class')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig(f'{EXP_DIR}/calibration.png', dpi=120)
print(f"\nsaved {EXP_DIR}/calibration.png")

# ---- confusion matrix PNG (Section 20) ----
with open(f'{EXP_DIR}/metrics.json') as f:
    metrics = json.load(f)
cm = np.array(metrics['confusion_matrix'])
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix - Test Set (n={cm.sum()})\naccuracy={metrics["test_accuracy"]:.3f} macroF1={metrics["test_macro_f1"]:.3f}')
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{EXP_DIR}/confusion_matrix.png', dpi=120)
print(f"saved {EXP_DIR}/confusion_matrix.png")

calib_results = {
    'brier_score_multiclass': float(brier_score),
    'ece_10bin': float(ece),
    'n_test_samples': len(df),
    'bin_stats': bin_stats,
    'note': 'Probability outputs are model confidence scores, NOT physical measurements. '
            'P(WET)=0.73 means the model assigns 0.73 probability to class WET, not that 73% of the road surface is wet.',
}
with open(f'{EXP_DIR}/calibration_metrics.json', 'w') as f:
    json.dump(calib_results, f, indent=2)
print(f"saved {EXP_DIR}/calibration_metrics.json")
