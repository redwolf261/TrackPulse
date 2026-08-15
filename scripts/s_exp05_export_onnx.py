"""
Export exp05's winning checkpoint (optionA) to ONNX, verify PyTorch/ONNX
equivalence, same protocol as prior rounds. Saves to
models/trackpulse_classifier_v6_exp05.onnx (does NOT overwrite production
models/trackpulse_classifier.onnx, which remains exp02).
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import numpy as np

CKPT = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp05_data_moat/checkpoints_optionA/best_model.pth'
ONNX_OUT = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models/trackpulse_classifier_v6_exp05.onnx'
CLASSES = ['DRY', 'DAMP', 'WET']

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

import onnx
m = onnx.load(ONNX_OUT)
onnx.checker.check_model(m)
print("ONNX model check passed")
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print("input:", inp.name, dims)
for out in m.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print("output:", out.name, dims)

# ---- PyTorch vs ONNX equivalence check ----
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_OUT, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

torch.manual_seed(123)
max_abs_diff = 0.0
n_checks = 20
with torch.no_grad():
    for i in range(n_checks):
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        torch_out = model(x).numpy()
        onnx_out = sess.run(None, {input_name: x.numpy().astype(np.float32)})[0]
        diff = np.abs(torch_out - onnx_out).max()
        max_abs_diff = max(max_abs_diff, diff)
print(f"\nPyTorch vs ONNX equivalence check ({n_checks} random inputs): max abs diff = {max_abs_diff:.8f}")
assert max_abs_diff < 1e-3, "ONNX output diverges from PyTorch beyond tolerance!"
print("EQUIVALENCE CONFIRMED (max abs diff < 1e-3)")
