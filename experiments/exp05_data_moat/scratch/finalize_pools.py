import json, os, shutil, random
random.seed(42)

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"

# ---- 1. racing_train_pool_v3: build stratified 85/15 train/val split (matches v2's protocol) ----
TRAIN_DIR = f"{ROOT}/data/racing_train_pool_v3"
with open(f"{TRAIN_DIR}/training_manifest_raw.json", encoding="utf-8") as f:
    manifest = json.load(f)

by_label = {}
for m in manifest:
    by_label.setdefault(m["label"], []).append(m)

train_split, val_split = [], []
for label, items in by_label.items():
    items = items[:]
    random.shuffle(items)
    n_val = max(1, round(len(items) * 0.15))
    val_items = items[:n_val]
    train_items = items[n_val:]
    for it in train_items:
        rec = dict(it)
        rec["filepath"] = f"data/racing_train_pool_v3/images/{it['filename']}"
        train_split.append(rec)
    for it in val_items:
        rec = dict(it)
        rec["filepath"] = f"data/racing_train_pool_v3/images/{it['filename']}"
        val_split.append(rec)

with open(f"{TRAIN_DIR}/racing_train_split.json", "w", encoding="utf-8") as f:
    json.dump(train_split, f, indent=1, ensure_ascii=False)
with open(f"{TRAIN_DIR}/racing_val_split.json", "w", encoding="utf-8") as f:
    json.dump(val_split, f, indent=1, ensure_ascii=False)

from collections import Counter
print("train_pool_v3 train split:", len(train_split), dict(Counter(r["label"] for r in train_split)))
print("train_pool_v3 val split:", len(val_split), dict(Counter(r["label"] for r in val_split)))

# ---- 2. racing_spotcheck_v3: copy v2's 67 images UNCHANGED + new 36, combined manifest ----
V2_DIR = f"{ROOT}/data/racing_spotcheck_v2"
V3_DIR = f"{ROOT}/data/racing_spotcheck_v3"
os.makedirs(f"{V3_DIR}/images", exist_ok=True)
os.makedirs(f"{V3_DIR}/images2", exist_ok=True)

# copy v2 images unchanged (both images/ and images2/ subdirs) into v3, same relative layout
for sub in ("images", "images2"):
    src_dir = os.path.join(V2_DIR, sub)
    dst_dir = os.path.join(V3_DIR, sub)
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(dst_dir, fname))

with open(f"{V2_DIR}/ground_truth_manifest.json", encoding="utf-8") as f:
    v2_manifest = json.load(f)
# rewrite filepaths to point at v3 (same relative sub-layout, just new root)
v3_manifest_v2part = []
for m in v2_manifest:
    rec = dict(m)
    rel = m["filepath"].split("racing_spotcheck_v2/")[-1]
    rec["filepath"] = f"{ROOT}/data/racing_spotcheck_v3/{rel}"
    v3_manifest_v2part.append(rec)

with open(f"{V3_DIR}/ground_truth_manifest_new_additions.json", encoding="utf-8") as f:
    new_additions = json.load(f)

full_v3_manifest = v3_manifest_v2part + new_additions
with open(f"{V3_DIR}/ground_truth_manifest.json", "w", encoding="utf-8") as f:
    json.dump(full_v3_manifest, f, indent=1, ensure_ascii=False)

print(f"\nracing_spotcheck_v3: {len(v3_manifest_v2part)} (unchanged from v2) + "
      f"{len(new_additions)} (new) = {len(full_v3_manifest)} total")
print("v3 label distribution:", dict(Counter(m["ground_truth"] for m in full_v3_manifest)))
