"""
Step 6: Export trained model to ONNX and verify against PyTorch outputs.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import mobilenet_v3_small
import onnxruntime as ort
import pandas as pd
from PIL import Image
import torchvision.transforms as T

MODELS_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models'
DATA_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data'
CLASSES = ['DRY', 'DAMP', 'WET']

device = torch.device('cpu')  # export/verify on CPU for determinism/simplicity

model = mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 3)
state = torch.load(os.path.join(MODELS_DIR, 'trackpulse_classifier.pt'), map_location='cpu')
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
onnx_path = os.path.join(MODELS_DIR, 'trackpulse_classifier.onnx')
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['input'], output_names=['logits'],
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
)
print("exported to", onnx_path)

# verify with a handful of real val images
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

val_df = pd.read_csv(os.path.join(DATA_DIR, 'manifest_val.csv')).sample(n=8, random_state=0)

sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

max_abs_diff = 0.0
mismatches = 0
with torch.no_grad():
    for _, row in val_df.iterrows():
        img = Image.open(row['filepath']).convert('RGB')
        x = tf(img).unsqueeze(0)
        torch_out = model(x).numpy()
        onnx_out = sess.run(None, {'input': x.numpy()})[0]
        diff = np.abs(torch_out - onnx_out).max()
        max_abs_diff = max(max_abs_diff, diff)
        torch_pred = torch_out.argmax(axis=1)[0]
        onnx_pred = onnx_out.argmax(axis=1)[0]
        if torch_pred != onnx_pred:
            mismatches += 1
        print(f"{row['filename']}: true={row['label']} torch_pred={CLASSES[torch_pred]} onnx_pred={CLASSES[onnx_pred]} max_logit_diff={diff:.6f}")

print(f"\nmax abs logit diff across samples: {max_abs_diff:.6f}")
print(f"prediction mismatches (torch vs onnx): {mismatches}/{len(val_df)}")
