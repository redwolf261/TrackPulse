"""
Remove unusable images from the curated set: images where the track surface
is not identifiable (extreme motion-panning blur reduces the whole frame to
horizontal streaks with no discernible tarmac texture). Uses a cheap Laplacian
variance blur-detection heuristic on a center crop, plus removes a
manually-identified bad index found during spot-check (1179).
"""
import json, os
import numpy as np
from PIL import Image, ImageFilter

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"

with open(f"{EXP}/scratch/curated_all_final.json", encoding="utf-8") as f:
    curated = json.load(f)

MANUAL_DROP = {
    1179,  # extreme motion blur, no track texture
    1163, 1165, 1170, 1173, 1174, 1175, 1176, 1177,  # extreme pan-blur/off-track
    1164, 1185, 1186, 1189, 1233, 1184,  # non-track (plaque/portrait/tire/museum/map)
    1238, 1239, 1243, 1244, 1246, 1247,  # non-track (fence-obscured, tent, museum display)
}  # confirmed unusable via full-res spot-check

def laplacian_var(img):
    # cheap edge-energy proxy without cv2/scipy: Laplacian-like via PIL filter
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float32)
    return arr.var()

kept = []
dropped_blur = []
for item in curated:
    if item["idx"] in MANUAL_DROP:
        continue
    path = os.path.join(STAGING, item["staged_filename"])
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        continue
    # center-crop lower half (where track surface usually is) for the metric
    w, h = img.size
    crop = img.crop((int(w*0.1), int(h*0.4), int(w*0.9), int(h*0.95)))
    v = laplacian_var(crop)
    if v < 80:  # very low edge energy = likely uniform blur streak / no texture
        dropped_blur.append((item["idx"], item["staged_filename"], round(v, 1)))
        continue
    kept.append(item)

print(f"curated input: {len(curated)}  manual_drop: {len(MANUAL_DROP)}  "
      f"blur_dropped: {len(dropped_blur)}  final_kept: {len(kept)}")
if dropped_blur:
    print("some blur-dropped examples:", dropped_blur[:15])

from collections import Counter
counts = Counter(it["label"] for it in kept)
print("final label counts:", dict(counts))

with open(f"{EXP}/scratch/curated_final_clean.json", "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=1, ensure_ascii=False)
print("saved curated_final_clean.json")
