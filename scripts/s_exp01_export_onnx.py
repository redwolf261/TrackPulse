"""
Export exp01 fine-tuned checkpoint to ONNX (static 1x3x224x224), saved as
models/trackpulse_classifier_v2_finetuned.onnx (does NOT touch the production
models/trackpulse_classifier.onnx). Then verify numerical equivalence vs PyTorch
on both the 49-image racing eval set and RSCD 305-image test set.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import onnx
import onnxruntime as ort

ROOT = 'c:/Users/Rivan/Projects/AI_Grand_Prix'
CKPT = f'{ROOT}/experiments/exp01_racing_finetune/checkpoints/best_model.pth'
ONNX_OUT = f'{ROOT}/models/trackpulse_classifier_v2_finetuned.onnx'
EXP_DIR = f'{ROOT}/experiments/exp01_racing_finetune'
CLASSES = ['DRY', 'DAMP', 'WET']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

model = mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
state = torch.load(CKPT, map_location='cpu')
model.load_state_dict(state)
model.eval()
print("loaded checkpoint:", CKPT)

dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
torch.onnx.export(
    model, dummy, ONNX_OUT,
    input_names=['input'], output_names=['logits'],
    opset_version=17,
    dynamic_axes=None,
    dynamo=False,
)
print("exported static-shape ONNX model to", ONNX_OUT)

m = onnx.load(ONNX_OUT)
onnx.checker.check_model(m)
print("ONNX model check passed")
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print("input:", inp.name, dims)
for out in m.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print("output:", out.name, dims)

# ==================== equivalence check ====================
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

sess = ort.InferenceSession(ONNX_OUT, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

def compare_on_image(path):
    img = Image.open(path).convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        pt_logits = model(x).numpy()[0]
    onnx_logits = sess.run(None, {input_name: x.numpy().astype(np.float32)})[0][0]
    logit_diff = float(np.abs(pt_logits - onnx_logits).max())
    pt_probs = np.exp(pt_logits - pt_logits.max()); pt_probs /= pt_probs.sum()
    onnx_probs = np.exp(onnx_logits - onnx_logits.max()); onnx_probs /= onnx_probs.sum()
    prob_diff = float(np.abs(pt_probs - onnx_probs).max())
    pt_pred = int(pt_probs.argmax()); onnx_pred = int(onnx_probs.argmax())
    return logit_diff, prob_diff, pt_pred, onnx_pred

results = {}
for name, get_paths in [
    ('racing_eval_49', None),
    ('rscd_test_305', None),
]:
    pass

# racing eval set
with open(f'{ROOT}/data/racing_spotcheck/ground_truth_manifest.json', encoding='utf-8') as f:
    gt = json.load(f)
racing_paths = [e['filepath'] for e in gt]

# rscd test set
test_df = pd.read_csv(f'{ROOT}/data/manifests/split_manifest_test.csv')
rscd_paths = test_df['filepath'].tolist()

for name, paths in [('racing_eval_49', racing_paths), ('rscd_test_305', rscd_paths)]:
    logit_diffs, prob_diffs, mismatches = [], [], 0
    for p in paths:
        ld, pd_, ptp, onp = compare_on_image(p)
        logit_diffs.append(ld); prob_diffs.append(pd_)
        if ptp != onp:
            mismatches += 1
    results[name] = {
        'n_images': len(paths),
        'max_logit_diff': float(max(logit_diffs)),
        'mean_logit_diff': float(np.mean(logit_diffs)),
        'max_prob_diff': float(max(prob_diffs)),
        'mean_prob_diff': float(np.mean(prob_diffs)),
        'prediction_mismatches': mismatches,
    }
    print(f"\n{name}: n={len(paths)} max_logit_diff={results[name]['max_logit_diff']:.2e} "
          f"max_prob_diff={results[name]['max_prob_diff']:.2e} mismatches={mismatches}/{len(paths)}")

with open(f'{EXP_DIR}/onnx_equivalence_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {EXP_DIR}/onnx_equivalence_results.json")
