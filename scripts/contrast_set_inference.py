"""
Step 2: Run frozen ONNX model on the racing-domain contrast set, record full per-image detail.
"""
import json, time
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as T

ONNX_PATH = 'c:/Users/Rivan/Projects/AI_Grand_Prix/models/trackpulse_classifier.onnx'
MANIFEST = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/racing_spotcheck/ground_truth_manifest.json'
OUT_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'
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
for r in manifest:
    img = Image.open(r['filepath']).convert('RGB')
    x = tf(img).unsqueeze(0).numpy().astype(np.float32)

    t0 = time.perf_counter()
    logits = sess.run(None, {input_name: x})[0][0]
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    pred_idx = int(probs.argmax())

    rec = {
        'image_id': r['image_id'],
        'source_event': r['source_event'],
        'clip_id': r['clip_id'],
        'camera_type': r['camera_type'],
        'ground_truth': r['ground_truth'],
        'predicted_class': CLASSES[pred_idx],
        'P_DRY': float(probs[0]),
        'P_DAMP': float(probs[1]),
        'P_WET': float(probs[2]),
        'confidence': float(probs[pred_idx]),
        'latency_ms': latency_ms,
        'license': r['license'],
        'note': r.get('note', ''),
    }
    results.append(rec)
    print(f"{r['image_id']:20s} GT={r['ground_truth']:10s} pred={CLASSES[pred_idx]:5s} conf={probs[pred_idx]:.3f}  "
          f"P=[{probs[0]:.3f},{probs[1]:.3f},{probs[2]:.3f}] lat={latency_ms:.2f}ms")

with open(f'{OUT_DIR}/contrast_set_inference_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nsaved {OUT_DIR}/contrast_set_inference_results.json ({len(results)} records)")

# also CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(f'{OUT_DIR}/contrast_set_inference_results.csv', index=False)
print(f"saved {OUT_DIR}/contrast_set_inference_results.csv")
