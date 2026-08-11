"""
Section 7: image quality audit on the filtered (asphalt/concrete, dry/wet/water) set.
Resolution, aspect ratio, corrupt files, brightness, blur (variance of Laplacian).
"""
import json, os
import numpy as np
from PIL import Image
import cv2
import collections

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records.json') as f:
    records = json.load(f)

print("auditing", len(records), "filtered images")

resolutions = collections.Counter()
aspect_ratios = []
corrupt = []
brightness_vals = []
blur_vals = []
filesizes = []

for i, r in enumerate(records):
    path = r['filepath']
    try:
        filesizes.append(os.path.getsize(path))
        img = Image.open(path)
        img.verify()  # check corruption
        img = Image.open(path).convert('RGB')  # reopen after verify
        w, h = img.size
        resolutions[(w, h)] += 1
        aspect_ratios.append(w / h)

        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        brightness = gray.mean()
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness_vals.append(brightness)
        blur_vals.append(blur)
        r['width'] = w
        r['height'] = h
        r['brightness'] = float(brightness)
        r['blur_laplacian_var'] = float(blur)
        r['filesize_bytes'] = os.path.getsize(path)
    except Exception as e:
        corrupt.append((r['filename'], str(e)))
        r['corrupt'] = True
    if i % 500 == 0:
        print(i, "processed")

print("\n=== CORRUPT FILES ===")
print("count:", len(corrupt))
for f, e in corrupt[:10]:
    print(" ", f, e)

print("\n=== RESOLUTION DISTRIBUTION ===")
for (w, h), cnt in resolutions.most_common(15):
    print(f"  {w}x{h}: {cnt}")
print(f"  ... {len(resolutions)} distinct resolutions total")

ar = np.array(aspect_ratios)
print(f"\n=== ASPECT RATIO ===  min={ar.min():.3f} max={ar.max():.3f} mean={ar.mean():.3f} std={ar.std():.3f}")

fs = np.array(filesizes)
print(f"\n=== FILE SIZE (bytes) === min={fs.min()} max={fs.max()} mean={fs.mean():.0f} median={np.median(fs):.0f}")

bv = np.array(brightness_vals)
print(f"\n=== BRIGHTNESS (0-255 grayscale mean) === min={bv.min():.1f} max={bv.max():.1f} mean={bv.mean():.1f} median={np.median(bv):.1f}")
low_brightness_thresh = 20
high_brightness_thresh = 235
n_dark = (bv < low_brightness_thresh).sum()
n_bright = (bv > high_brightness_thresh).sum()
print(f"images with brightness < {low_brightness_thresh} (near-black): {n_dark}")
print(f"images with brightness > {high_brightness_thresh} (near-white/blown out): {n_bright}")

blv = np.array(blur_vals)
print(f"\n=== BLUR (variance of Laplacian) === min={blv.min():.1f} max={blv.max():.1f} mean={blv.mean():.1f} median={np.median(blv):.1f}")
blur_thresh = 50  # commonly used heuristic threshold for "very blurry" - reported not enforced blindly
n_blurry = (blv < blur_thresh).sum()
print(f"images with blur variance < {blur_thresh} (candidate 'very blurry', heuristic threshold, NOT auto-excluded): {n_blurry}")

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records_with_quality.json', 'w') as f:
    json.dump(records, f)

summary = {
    'n_audited': len(records),
    'n_corrupt': len(corrupt),
    'corrupt_files': corrupt,
    'n_distinct_resolutions': len(resolutions),
    'top_resolutions': [{'w': w, 'h': h, 'count': c} for (w,h), c in resolutions.most_common(15)],
    'aspect_ratio': {'min': float(ar.min()), 'max': float(ar.max()), 'mean': float(ar.mean()), 'std': float(ar.std())},
    'filesize_bytes': {'min': int(fs.min()), 'max': int(fs.max()), 'mean': float(fs.mean()), 'median': float(np.median(fs))},
    'brightness': {'min': float(bv.min()), 'max': float(bv.max()), 'mean': float(bv.mean()), 'median': float(np.median(bv)),
                   'n_near_black_lt20': int(n_dark), 'n_near_white_gt235': int(n_bright)},
    'blur_laplacian_var': {'min': float(blv.min()), 'max': float(blv.max()), 'mean': float(blv.mean()), 'median': float(np.median(blv)),
                            'n_below_heuristic_thresh_50': int(n_blurry)},
}
with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/s7_quality_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nsaved filtered_records_with_quality.json and s7_quality_summary.json")
