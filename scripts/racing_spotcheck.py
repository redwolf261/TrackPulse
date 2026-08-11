"""
Step 5: racing-domain stress test using the frozen ONNX model.
Lightweight spot-check only - NOT a statistical benchmark.
"""
import json, os
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as T

ONNX_PATH = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models/trackpulse_classifier.onnx'
IMG_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/racing_spotcheck/images'
MANIFEST = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/racing_spotcheck/source_manifest.json'
EXP_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'
CLASSES = ['DRY', 'DAMP', 'WET']

with open(MANIFEST, encoding='utf-8') as f:
    manifest = json.load(f)

IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

results = []
for m in manifest:
    path = os.path.join(IMG_DIR, m['filename'])
    img = Image.open(path).convert('RGB')
    x = tf(img).unsqueeze(0).numpy().astype(np.float32)
    logits = sess.run(None, {input_name: x})[0][0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    pred_idx = int(probs.argmax())
    results.append({
        'filename': m['filename'],
        'source_title': m['source_title'],
        'license': m['license'],
        'predicted_class': CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'probs': {CLASSES[i]: float(probs[i]) for i in range(3)},
    })
    print(f"{m['filename']:20s} pred={CLASSES[pred_idx]:5s} conf={probs[pred_idx]:.3f}  "
          f"probs=DRY:{probs[0]:.3f} DAMP:{probs[1]:.3f} WET:{probs[2]:.3f}  src={m['source_title']}")

pred_counts = {}
for r in results:
    pred_counts[r['predicted_class']] = pred_counts.get(r['predicted_class'], 0) + 1
print("\nprediction distribution across", len(results), "racing images:", pred_counts)

with open(f'{EXP_DIR}/racing_spotcheck_results.json', 'w', encoding='utf-8') as f:
    json.dump({'n_images': len(results), 'prediction_distribution': pred_counts, 'results': results}, f, indent=2, ensure_ascii=False)
print(f"\nsaved {EXP_DIR}/racing_spotcheck_results.json")
