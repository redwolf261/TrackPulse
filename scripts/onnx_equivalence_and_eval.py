"""
Step 2: numerical equivalence test PyTorch vs ONNX on all 305 held-out test images.
Step 3: independent re-evaluation via ONNX Runtime (accuracy/macroF1/per-class/confusion/WET recall/ECE).
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as T
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report

CKPT = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline/checkpoints/best_model.pth'
ONNX_PATH = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models/trackpulse_classifier.onnx'
TEST_MANIFEST = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/manifests/split_manifest_test.csv'
EXP_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'
CLASSES = ['DRY', 'DAMP', 'WET']

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

# load PyTorch model
model = mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
model.load_state_dict(torch.load(CKPT, map_location='cpu'))
model.eval()
print("loaded PyTorch checkpoint:", CKPT)

# load ONNX session (CPU)
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
print("loaded ONNX model:", ONNX_PATH, "input:", input_name)
print("ONNX providers active:", sess.get_providers())

df = pd.read_csv(TEST_MANIFEST)
print(f"\nrunning equivalence test on {len(df)} test images")

torch_logits_all, onnx_logits_all = [], []
torch_preds_all, onnx_preds_all = [], []
true_labels = []

with torch.no_grad():
    for i, row in df.iterrows():
        img = Image.open(row['filepath']).convert('RGB')
        x = eval_tf(img).unsqueeze(0)  # (1,3,224,224) float32

        torch_out = model(x).numpy()  # (1,3)
        onnx_out = sess.run(None, {input_name: x.numpy().astype(np.float32)})[0]  # (1,3)

        torch_logits_all.append(torch_out[0])
        onnx_logits_all.append(onnx_out[0])
        torch_preds_all.append(int(torch_out[0].argmax()))
        onnx_preds_all.append(int(onnx_out[0].argmax()))
        true_labels.append(CLASSES.index(row['label']))

        if i % 50 == 0:
            print(i, "done")

torch_logits_all = np.array(torch_logits_all)
onnx_logits_all = np.array(onnx_logits_all)
torch_preds_all = np.array(torch_preds_all)
onnx_preds_all = np.array(onnx_preds_all)
true_labels = np.array(true_labels)

# softmax probs
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

torch_probs = softmax(torch_logits_all)
onnx_probs = softmax(onnx_logits_all)

max_abs_logit_diff = np.abs(torch_logits_all - onnx_logits_all).max()
max_abs_prob_diff = np.abs(torch_probs - onnx_probs).max()
n_pred_mismatch = int((torch_preds_all != onnx_preds_all).sum())

TOL_LOGITS = 1e-4
pass_fail = "PASS" if max_abs_logit_diff <= TOL_LOGITS else "FAIL"

print("\n=== STEP 2: NUMERICAL EQUIVALENCE RESULTS ===")
print(f"n images compared: {len(df)}")
print(f"max abs logit diff (PyTorch vs ONNX): {max_abs_logit_diff:.8f}")
print(f"max abs softmax prob diff: {max_abs_prob_diff:.8f}")
print(f"prediction mismatches (argmax differs): {n_pred_mismatch} / {len(df)}")
print(f"tolerance set: atol={TOL_LOGITS} on logits -> {pass_fail}")

if n_pred_mismatch > 0:
    mismatch_idx = np.where(torch_preds_all != onnx_preds_all)[0]
    print("\nMISMATCHED PREDICTIONS (investigate):")
    for idx in mismatch_idx:
        print(f"  row {idx}: file={df.iloc[idx]['filename']} torch_pred={CLASSES[torch_preds_all[idx]]} onnx_pred={CLASSES[onnx_preds_all[idx]]}")
        print(f"    torch_logits={torch_logits_all[idx]} onnx_logits={onnx_logits_all[idx]}")

equiv_results = {
    'n_images': len(df),
    'max_abs_logit_diff': float(max_abs_logit_diff),
    'max_abs_softmax_prob_diff': float(max_abs_prob_diff),
    'n_prediction_mismatches': n_pred_mismatch,
    'tolerance_atol_logits': TOL_LOGITS,
    'pass_fail': pass_fail,
}
with open(f'{EXP_DIR}/onnx_equivalence_results.json', 'w') as f:
    json.dump(equiv_results, f, indent=2)
print(f"\nsaved {EXP_DIR}/onnx_equivalence_results.json")

# ---- Step 3: independent full evaluation via ONNX ----
print("\n=== STEP 3: INDEPENDENT ONNX EVALUATION (305-image test set) ===")
onnx_acc = accuracy_score(true_labels, onnx_preds_all)
onnx_macro_f1 = f1_score(true_labels, onnx_preds_all, average='macro')
prec, rec, f1, support = precision_recall_fscore_support(true_labels, onnx_preds_all, labels=[0,1,2])
cm = confusion_matrix(true_labels, onnx_preds_all, labels=[0,1,2])
report = classification_report(true_labels, onnx_preds_all, target_names=CLASSES, digits=4)

print("ONNX accuracy:", onnx_acc)
print("ONNX macro F1:", onnx_macro_f1)
print(report)
print("confusion matrix (rows=true, cols=pred), order", CLASSES)
print(cm)
print(f"ONNX WET recall: {rec[2]:.4f}")

# ECE via ONNX probs, 10-bin
confidences = onnx_probs.max(axis=1)
correct = (onnx_preds_all == true_labels).astype(float)
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
ece = 0.0
for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i+1]
    mask = (confidences >= lo) & (confidences < hi) if i < n_bins-1 else (confidences >= lo) & (confidences <= hi)
    n_in_bin = mask.sum()
    if n_in_bin > 0:
        ece += (n_in_bin / len(df)) * abs(confidences[mask].mean() - correct[mask].mean())

# brier
y_true_onehot = np.zeros((len(df), 3))
for i, l in enumerate(true_labels):
    y_true_onehot[i, l] = 1.0
brier = np.mean(np.sum((onnx_probs - y_true_onehot) ** 2, axis=1))

print(f"\nONNX Brier score: {brier:.4f}")
print(f"ONNX ECE (10-bin): {ece:.4f}")

# compare vs PyTorch reference
ref = {
    'accuracy': 0.7770491803278688,
    'macro_f1': 0.7540313972839924,
    'wet_recall': 0.8311688311688312,
    'wet_f1': 0.7757575757575758,
    'ece': 0.0819,
}
print("\n=== DRIFT vs PYTORCH REFERENCE ===")
print(f"accuracy: onnx={onnx_acc:.6f} ref={ref['accuracy']:.6f} diff={onnx_acc-ref['accuracy']:+.6f}")
print(f"macro_f1: onnx={onnx_macro_f1:.6f} ref={ref['macro_f1']:.6f} diff={onnx_macro_f1-ref['macro_f1']:+.6f}")
print(f"wet_recall: onnx={rec[2]:.6f} ref={ref['wet_recall']:.6f} diff={rec[2]-ref['wet_recall']:+.6f}")
print(f"wet_f1: onnx={f1[2]:.6f} ref={ref['wet_f1']:.6f} diff={f1[2]-ref['wet_f1']:+.6f}")
print(f"ece: onnx={ece:.6f} ref={ref['ece']:.6f} diff={ece-ref['ece']:+.6f}")

onnx_eval_results = {
    'test_accuracy': float(onnx_acc),
    'test_macro_f1': float(onnx_macro_f1),
    'per_class': {CLASSES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)},
    'confusion_matrix': cm.tolist(),
    'wet_recall': float(rec[2]),
    'wet_f1': float(f1[2]),
    'brier_score': float(brier),
    'ece_10bin': float(ece),
    'pytorch_reference': ref,
    'drift': {
        'accuracy_diff': float(onnx_acc - ref['accuracy']),
        'macro_f1_diff': float(onnx_macro_f1 - ref['macro_f1']),
        'wet_recall_diff': float(rec[2] - ref['wet_recall']),
        'wet_f1_diff': float(f1[2] - ref['wet_f1']),
        'ece_diff': float(ece - ref['ece']),
    },
}
with open(f'{EXP_DIR}/onnx_eval_results.json', 'w') as f:
    json.dump(onnx_eval_results, f, indent=2)
print(f"\nsaved {EXP_DIR}/onnx_eval_results.json")
