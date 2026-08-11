"""
Step 1: Export the frozen best_model.pth checkpoint to ONNX.
Static input shape (1,3,224,224), FP32, matching backend/app/inference.py's contract exactly.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

CKPT = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline/checkpoints/best_model.pth'
ONNX_OUT = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models/trackpulse_classifier.onnx'
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
    dynamic_axes=None,  # STATIC shape 1x3x224x224 per spec - batch=1 fixed
    dynamo=False,  # use legacy TorchScript-based exporter (stable, avoids dynamo/onnxscript console-encoding issue)
)
print("exported static-shape ONNX model to", ONNX_OUT)

# quick shape sanity check
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
