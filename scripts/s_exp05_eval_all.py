"""
exp05 evaluation: run a given checkpoint (PyTorch) on all 4 eval sets, full
protocol (confusion matrix, accuracy, macro-F1, per-class P/R/F1, WET recall,
DRY recall/precision, false-wet rate, false-dry rate, mean confidence
correct/incorrect):
  1. original 49-image eval set (data/racing_spotcheck/ground_truth_manifest.json)
  2. 67-image v2 eval set (data/racing_spotcheck_v2/ground_truth_manifest.json)
  3. NEW v3 eval set (data/racing_spotcheck_v3/ground_truth_manifest.json, 103 imgs)
  4. RSCD test (305 images, data/manifests/split_manifest_test.csv)

Usage: python s_exp05_eval_all.py <checkpoint_path> <output_json_path> [--label NAME]
"""
import os, sys, json
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
CKPT = sys.argv[1]
OUT_PATH = sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(CKPT)

CLASSES = ['DRY', 'DAMP', 'WET']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device, "| checkpoint:", CKPT)

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


def infer_image(path):
    img = Image.open(path).convert('RGB')
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out.float(), dim=1).cpu().numpy()[0]
    return probs


def eval_racing_set(gt_path, image_id_key='image_id', gt_key='ground_truth', filepath_key='filepath'):
    with open(gt_path, encoding='utf-8') as f:
        gt = json.load(f)
    results = []
    for e in gt:
        fp = e[filepath_key]
        if not os.path.isabs(fp):
            fp = os.path.join(ROOT, fp)
        if not os.path.exists(fp):
            print(f"  WARNING missing file: {fp}")
            continue
        probs = infer_image(fp)
        pred_idx = int(probs.argmax())
        results.append({
            'image_id': e.get(image_id_key, e.get('filename', '?')),
            'ground_truth': e[gt_key],
            'predicted_class': CLASSES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probs': {CLASSES[i]: float(probs[i]) for i in range(3)},
        })
    quant = [r for r in results if r['ground_truth'] in CLASSES]
    ambiguous = [r for r in results if r['ground_truth'] not in CLASSES]
    y_true = [CLASS_TO_IDX[r['ground_truth']] for r in quant]
    y_pred = [CLASS_TO_IDX[r['predicted_class']] for r in quant]

    acc = accuracy_score(y_true, y_pred) if quant else None
    macro_f1 = f1_score(y_true, y_pred, average='macro') if quant else None
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0) if quant else (None,)*4
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2]) if quant else None

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

    metrics = {
        'n_total': len(results), 'n_ambiguous_excluded': len(ambiguous), 'n_quantitative': len(quant),
        'accuracy': float(acc) if acc is not None else None,
        'macro_f1': float(macro_f1) if macro_f1 is not None else None,
        'per_class': ({CLASSES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)} if quant else None),
        'confusion_matrix': cm.tolist() if cm is not None else None, 'classes_order': CLASSES,
        'wet_recall': float(rec[2]) if quant else None, 'dry_recall': float(rec[0]) if quant else None,
        'dry_precision': float(prec[0]) if quant else None,
        'false_wet_rate': false_wet_rate, 'false_dry_rate': false_dry_rate,
        'mean_confidence_correct': mean_conf_correct, 'mean_confidence_incorrect': mean_conf_incorrect,
    }
    return metrics, results


all_results = {'checkpoint': CKPT, 'label': LABEL}

print("\n=== (1) Original 49-image eval set ===")
m1, r1 = eval_racing_set(f'{ROOT}/data/racing_spotcheck/ground_truth_manifest.json')
print(json.dumps({k: v for k, v in m1.items() if k not in ('per_class','confusion_matrix')}, indent=2))
all_results['eval_49img'] = m1

print("\n=== (2) v2 67-image eval set ===")
m2, r2 = eval_racing_set(f'{ROOT}/data/racing_spotcheck_v2/ground_truth_manifest.json')
print(json.dumps({k: v for k, v in m2.items() if k not in ('per_class','confusion_matrix')}, indent=2))
all_results['eval_v2_67img'] = m2

print("\n=== (3) v3 103-image eval set (NEW, exp05) ===")
m3, r3 = eval_racing_set(f'{ROOT}/data/racing_spotcheck_v3/ground_truth_manifest.json')
print(json.dumps({k: v for k, v in m3.items() if k not in ('per_class','confusion_matrix')}, indent=2))
all_results['eval_v3_103img'] = m3

print("\n=== (4) RSCD 305-image test set ===")
test_df = pd.read_csv(f'{ROOT}/data/manifests/split_manifest_test.csv')
all_preds, all_labels = [], []
for _, row in test_df.iterrows():
    probs = infer_image(row['filepath'])
    all_preds.append(int(probs.argmax()))
    all_labels.append(CLASS_TO_IDX[row['label']])
rscd_acc = accuracy_score(all_labels, all_preds)
rscd_macro_f1 = f1_score(all_labels, all_preds, average='macro')
r_prec, r_rec, r_f1, r_support = precision_recall_fscore_support(all_labels, all_preds, labels=[0,1,2], zero_division=0)
rscd_cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
print("accuracy:", rscd_acc, "macro_f1:", rscd_macro_f1, "WET recall:", r_rec[2])
rscd_metrics = {
    'accuracy': float(rscd_acc), 'macro_f1': float(rscd_macro_f1),
    'per_class': {CLASSES[i]: {'precision': float(r_prec[i]), 'recall': float(r_rec[i]), 'f1': float(r_f1[i]), 'support': int(r_support[i])} for i in range(3)},
    'confusion_matrix': rscd_cm.tolist(), 'classes_order': CLASSES,
    'wet_recall': float(r_rec[2]), 'dry_precision': float(r_prec[0]),
}
all_results['rscd_test_305img'] = rscd_metrics

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2)
print(f"\nsaved {OUT_PATH}")
