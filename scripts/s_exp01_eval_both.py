"""
exp01 evaluation: run the fine-tuned checkpoint (PyTorch, GPU/CPU) on:
  (a) the untouched 49-image racing eval set (ground_truth_manifest.json)
  (b) the original RSCD 305-image held-out test set (split_manifest_test.csv)
Same protocol as exp00's frozen-baseline evaluation for direct comparability.
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
CKPT = f'{ROOT}/experiments/exp01_racing_finetune/checkpoints/best_model.pth'
EXP_DIR = f'{ROOT}/experiments/exp01_racing_finetune'
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


# ============ (a) 49-image racing eval set ============
gt_path = f'{ROOT}/data/racing_spotcheck/ground_truth_manifest.json'
with open(gt_path, encoding='utf-8') as f:
    gt = json.load(f)

racing_results = []
for e in gt:
    probs = infer_image(e['filepath'])
    pred_idx = int(probs.argmax())
    racing_results.append({
        'image_id': e['image_id'],
        'ground_truth': e['ground_truth'],
        'predicted_class': CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'probs': {CLASSES[i]: float(probs[i]) for i in range(3)},
        'source_event': e['source_event'],
    })

# quantitative metrics: exclude AMBIGUOUS
quant = [r for r in racing_results if r['ground_truth'] != 'AMBIGUOUS']
y_true = [CLASS_TO_IDX[r['ground_truth']] for r in quant]
y_pred = [CLASS_TO_IDX[r['predicted_class']] for r in quant]

racing_acc = accuracy_score(y_true, y_pred)
racing_macro_f1 = f1_score(y_true, y_pred, average='macro')
prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0)

print("\n=== RACING EVAL SET (49 images, quantitative on 44 non-AMBIGUOUS) ===")
print("accuracy:", racing_acc)
print("macro F1:", racing_macro_f1)
print(report)
print("confusion matrix (rows=true, cols=pred), order", CLASSES)
print(cm)

# false-wet rate: true DRY predicted WET / total true DRY
# false-dry rate: true WET predicted DRY / total true WET
dry_idx, damp_idx, wet_idx = 0, 1, 2
n_true_dry = sum(1 for r in quant if r['ground_truth']=='DRY')
n_true_wet = sum(1 for r in quant if r['ground_truth']=='WET')
false_wet = sum(1 for r in quant if r['ground_truth']=='DRY' and r['predicted_class']=='WET')
false_dry = sum(1 for r in quant if r['ground_truth']=='WET' and r['predicted_class']=='DRY')
false_wet_rate = false_wet / n_true_dry if n_true_dry else None
false_dry_rate = false_dry / n_true_wet if n_true_wet else None

correct_confs = [r['confidence'] for r in quant if r['predicted_class']==r['ground_truth']]
incorrect_confs = [r['confidence'] for r in quant if r['predicted_class']!=r['ground_truth']]
mean_conf_correct = float(np.mean(correct_confs)) if correct_confs else None
mean_conf_incorrect = float(np.mean(incorrect_confs)) if incorrect_confs else None

ambiguous = [r for r in racing_results if r['ground_truth']=='AMBIGUOUS']
print("\nAMBIGUOUS (5, descriptive only):")
for r in ambiguous:
    print(f"  {r['image_id']}: pred={r['predicted_class']} conf={r['confidence']:.3f}")

racing_metrics = {
    'n_total': len(racing_results), 'n_ambiguous_excluded': len(ambiguous), 'n_quantitative': len(quant),
    'accuracy': float(racing_acc), 'macro_f1': float(racing_macro_f1),
    'per_class': {CLASSES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)},
    'confusion_matrix': cm.tolist(), 'classes_order': CLASSES,
    'wet_recall': float(rec[2]), 'dry_recall': float(rec[0]), 'dry_precision': float(prec[0]),
    'false_wet_rate': false_wet_rate, 'false_dry_rate': false_dry_rate,
    'mean_confidence_correct': mean_conf_correct, 'mean_confidence_incorrect': mean_conf_incorrect,
    'ambiguous_predictions': [{'image_id': r['image_id'], 'predicted_class': r['predicted_class'], 'confidence': r['confidence']} for r in ambiguous],
}
with open(f'{EXP_DIR}/racing_eval_results.json', 'w', encoding='utf-8') as f:
    json.dump({'metrics': racing_metrics, 'all_predictions': racing_results}, f, indent=2, ensure_ascii=False)
print(f"\nsaved {EXP_DIR}/racing_eval_results.json")


# ============ (b) RSCD 305-image held-out test set ============
test_df = pd.read_csv(f'{ROOT}/data/manifests/split_manifest_test.csv')
all_preds, all_labels, all_probs = [], [], []
for _, row in test_df.iterrows():
    probs = infer_image(row['filepath'])
    all_probs.append(probs.tolist())
    all_preds.append(int(probs.argmax()))
    all_labels.append(CLASS_TO_IDX[row['label']])

rscd_acc = accuracy_score(all_labels, all_preds)
rscd_macro_f1 = f1_score(all_labels, all_preds, average='macro')
r_prec, r_rec, r_f1, r_support = precision_recall_fscore_support(all_labels, all_preds, labels=[0,1,2], zero_division=0)
rscd_cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
rscd_report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4, zero_division=0)

print("\n=== RSCD 305-IMAGE TEST SET (fine-tuned model) ===")
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
