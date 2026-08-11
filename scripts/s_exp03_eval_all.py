"""
exp03 evaluation: run the fine-tuned checkpoint (PyTorch) on:
  (a) the untouched 49-image racing eval set (data/racing_spotcheck/ground_truth_manifest.json)
  (b) the NEW expanded 67-image eval set = 49 + 18 held-out DAMP
      (data/racing_spotcheck_v2/ground_truth_manifest.json)
  (c) the original RSCD 305-image held-out test set (split_manifest_test.csv)
Same protocol as exp00/exp01/exp02's evaluation for direct comparability.
Single run, no tuning on eval sets.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

ROOT = 'c:/Users/Rivan/Projects/AI_Grand_Prix'
CKPT = f'{ROOT}/experiments/exp03_damp_holdout/checkpoints/best_model.pth'
EXP_DIR = f'{ROOT}/experiments/exp03_damp_holdout'
CLASSES = ['DRY', 'DAMP', 'WET']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

model = mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
state = torch.load(CKPT, map_location='cpu')
model.load_state_dict(state)
model = model.to(device)
model.eval()
print("loaded fine-tuned checkpoint:", CKPT)


def infer_image(path):
    img = Image.open(path).convert('RGB')
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out.float(), dim=1).cpu().numpy()[0]
    return probs


def run_racing_eval(gt_path, out_json_name, label):
    with open(gt_path, encoding='utf-8') as f:
        gt = json.load(f)

    results = []
    for e in gt:
        probs = infer_image(e['filepath'])
        pred_idx = int(probs.argmax())
        results.append({
            'image_id': e['image_id'],
            'ground_truth': e['ground_truth'],
            'predicted_class': CLASSES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probs': {CLASSES[i]: float(probs[i]) for i in range(3)},
            'source_event': e.get('source_event', ''),
        })

    quant = [r for r in results if r['ground_truth'] != 'AMBIGUOUS']
    y_true = [CLASS_TO_IDX[r['ground_truth']] for r in quant]
    y_pred = [CLASS_TO_IDX[r['predicted_class']] for r in quant]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)

    print(f"\n=== {label} ({len(gt)} images, quantitative on non-AMBIGUOUS, n={len(quant)}) ===")
    print("accuracy:", acc)
    print("macro F1:", macro_f1)
    print(report)
    print("confusion matrix (rows=true, cols=pred), order", CLASSES)
    print(cm)

    n_true_dry = sum(1 for r in quant if r['ground_truth'] == 'DRY')
    n_true_wet = sum(1 for r in quant if r['ground_truth'] == 'WET')
    false_wet = sum(1 for r in quant if r['ground_truth'] == 'DRY' and r['predicted_class'] == 'WET')
    false_dry = sum(1 for r in quant if r['ground_truth'] == 'WET' and r['predicted_class'] == 'DRY')
    false_wet_rate = false_wet / n_true_dry if n_true_dry else None
    false_dry_rate = false_dry / n_true_wet if n_true_wet else None

    correct_confs = [r['confidence'] for r in quant if r['predicted_class'] == r['ground_truth']]
    incorrect_confs = [r['confidence'] for r in quant if r['predicted_class'] != r['ground_truth']]
    mean_conf_correct = float(np.mean(correct_confs)) if correct_confs else None
    mean_conf_incorrect = float(np.mean(incorrect_confs)) if incorrect_confs else None

    ambiguous = [r for r in results if r['ground_truth'] == 'AMBIGUOUS']
    if ambiguous:
        print(f"\nAMBIGUOUS ({len(ambiguous)}, descriptive only):")
        for r in ambiguous:
            print(f"  {r['image_id']}: pred={r['predicted_class']} conf={r['confidence']:.3f}")

    metrics = {
        'n_total': len(results), 'n_ambiguous_excluded': len(ambiguous), 'n_quantitative': len(quant),
        'accuracy': float(acc), 'macro_f1': float(macro_f1),
        'per_class': {CLASSES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)},
        'confusion_matrix': cm.tolist(), 'classes_order': CLASSES,
        'wet_recall': float(rec[2]), 'dry_recall': float(rec[0]), 'dry_precision': float(prec[0]),
        'damp_recall': float(rec[1]), 'damp_precision': float(prec[1]), 'damp_f1': float(f1[1]), 'damp_support': int(support[1]),
        'false_wet_rate': false_wet_rate, 'false_dry_rate': false_dry_rate,
        'mean_confidence_correct': mean_conf_correct, 'mean_confidence_incorrect': mean_conf_incorrect,
        'ambiguous_predictions': [{'image_id': r['image_id'], 'predicted_class': r['predicted_class'], 'confidence': r['confidence']} for r in ambiguous],
    }
    with open(f'{EXP_DIR}/{out_json_name}', 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'all_predictions': results}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {EXP_DIR}/{out_json_name}")
    return metrics


# ============ (a) original 49-image eval set (unchanged) ============
racing_metrics_49 = run_racing_eval(
    f'{ROOT}/data/racing_spotcheck/ground_truth_manifest.json',
    'racing_eval_49_results.json',
    'ORIGINAL 49-IMAGE RACING EVAL SET'
)

# ============ (b) NEW expanded 67-image eval set (49 + 18 DAMP holdout) ============
racing_metrics_67 = run_racing_eval(
    f'{ROOT}/data/racing_spotcheck_v2/ground_truth_manifest.json',
    'racing_eval_67_results.json',
    'EXPANDED 67-IMAGE EVAL SET (49 + 18 DAMP holdout)'
)

# ============ (c) RSCD 305-image held-out test set ============
test_df = pd.read_csv(f'{ROOT}/data/manifests/split_manifest_test.csv')
all_preds, all_labels, all_probs = [], [], []
for _, row in test_df.iterrows():
    probs = infer_image(row['filepath'])
    all_probs.append(probs.tolist())
    all_preds.append(int(probs.argmax()))
    all_labels.append(CLASS_TO_IDX[row['label']])

rscd_acc = accuracy_score(all_labels, all_preds)
rscd_macro_f1 = f1_score(all_labels, all_preds, average='macro')
r_prec, r_rec, r_f1, r_support = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1, 2], zero_division=0)
rscd_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
rscd_report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4, zero_division=0)

print("\n=== RSCD 305-IMAGE TEST SET (exp03 fine-tuned model) ===")
print("accuracy:", rscd_acc)
print("macro F1:", rscd_macro_f1)
print(rscd_report)
print("confusion matrix (rows=true, cols=pred), order", CLASSES)
print(rscd_cm)
print(f"WET recall: {r_rec[2]:.4f}")

rscd_metrics = {
    'test_accuracy': float(rscd_acc), 'test_macro_f1': float(rscd_macro_f1),
    'per_class': {CLASSES[i]: {'precision': float(r_prec[i]), 'recall': float(r_rec[i]), 'f1': float(r_f1[i]), 'support': int(r_support[i])} for i in range(3)},
    'confusion_matrix': rscd_cm.tolist(), 'classes_order': CLASSES,
    'wet_recall': float(r_rec[2]), 'dry_precision': float(r_prec[0]),
}
with open(f'{EXP_DIR}/rscd_test_eval_results.json', 'w') as f:
    json.dump(rscd_metrics, f, indent=2)
print(f"\nsaved {EXP_DIR}/rscd_test_eval_results.json")

print("\n\n=== SUMMARY ===")
print("49-img eval: acc=%.4f macroF1=%.4f DRY_recall=%.4f WET_recall=%.4f" % (
    racing_metrics_49['accuracy'], racing_metrics_49['macro_f1'], racing_metrics_49['dry_recall'], racing_metrics_49['wet_recall']))
print("67-img eval: acc=%.4f macroF1=%.4f DRY_recall=%.4f WET_recall=%.4f DAMP_recall=%.4f DAMP_precision=%.4f DAMP_f1=%.4f (support=%d)" % (
    racing_metrics_67['accuracy'], racing_metrics_67['macro_f1'], racing_metrics_67['dry_recall'], racing_metrics_67['wet_recall'],
    racing_metrics_67['damp_recall'], racing_metrics_67['damp_precision'], racing_metrics_67['damp_f1'], racing_metrics_67['damp_support']))
print("RSCD test: acc=%.4f macroF1=%.4f" % (rscd_acc, rscd_macro_f1))
