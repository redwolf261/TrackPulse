"""
Split curated_final_clean.json into:
- eval additions (data/racing_spotcheck_v3/): NEW held-out eval images, 40 total,
  DAMP-prioritized (take most/all DAMP), diverse DRY/WET sample.
- training pool (data/racing_train_pool_v3/): everything else.
Copies+resizes images to their final destinations (already resized at staging
time to ~1024px/q85, so this is effectively a straight copy+rename), builds
manifests in the EXACT schema of racing_train_pool_v2 / racing_spotcheck_v2,
and verifies zero SHA-256 overlap between the new eval additions and the new
training pool (and re-verifies against the original 4-source existing pool).
"""
import json, os, shutil, hashlib, random

random.seed(42)
ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"

with open(f"{EXP}/scratch/curated_final_clean.json", encoding="utf-8") as f:
    curated = json.load(f)

by_label = {"DAMP": [], "WET": [], "DRY": []}
for it in curated:
    by_label[it["label"]].append(it)

for k in by_label:
    random.shuffle(by_label[k])

# ---- eval additions: DAMP-prioritized, target ~40 total ----
EVAL_TARGET = 40
eval_damp = by_label["DAMP"][:4]           # split DAMP roughly evenly: eval needs
                                            # signal but training needs it more,
                                            # given only 10 total genuine examples
eval_wet = by_label["WET"][:16]
eval_dry = by_label["DRY"][:16]
eval_items = eval_damp + eval_wet + eval_dry
print("eval additions:", len(eval_items),
      {"DAMP": len(eval_damp), "WET": len(eval_wet), "DRY": len(eval_dry)})

eval_titles = {it["title"] for it in eval_items}
train_items = [it for it in curated if it["title"] not in eval_titles]
train_counts = {}
for it in train_items:
    train_counts[it["label"]] = train_counts.get(it["label"], 0) + 1
print("training pool:", len(train_items), train_counts)

# ---- write eval additions to data/racing_spotcheck_v3/images_new/ ----
EVAL_DIR = f"{ROOT}/data/racing_spotcheck_v3"
EVAL_IMG_DIR = f"{EVAL_DIR}/images_new"
os.makedirs(EVAL_IMG_DIR, exist_ok=True)

eval_manifest_new = []
for i, it in enumerate(sorted(eval_items, key=lambda x: x["label"])):
    fname = f"racing_v3new_{i:03d}.jpg"
    src = os.path.join(STAGING, it["staged_filename"])
    dst = os.path.join(EVAL_IMG_DIR, fname)
    shutil.copyfile(src, dst)
    eval_manifest_new.append({
        "image_id": fname,
        "filepath": f"{ROOT}/data/racing_spotcheck_v3/images_new/{fname}".replace("\\", "/"),
        "source_event": it["title"].replace("File:", ""),
        "clip_id": "UNKNOWN",
        "camera_type": "trackside",
        "ground_truth": it["label"],
        "license": it.get("license_short") or "unknown",
        "source_title": it["title"],
        "note": f"exp05 new eval addition; source_url={it.get('source_url','')}",
    })

with open(f"{EVAL_DIR}/ground_truth_manifest_new_additions.json", "w", encoding="utf-8") as f:
    json.dump(eval_manifest_new, f, indent=1, ensure_ascii=False)
print(f"saved {len(eval_manifest_new)} new eval images to {EVAL_IMG_DIR}")

# ---- write training pool to data/racing_train_pool_v3/images/ ----
TRAIN_DIR = f"{ROOT}/data/racing_train_pool_v3"
TRAIN_IMG_DIR = f"{TRAIN_DIR}/images"
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)

train_manifest_raw = []
for i, it in enumerate(sorted(train_items, key=lambda x: x["label"])):
    fname = f"racetrainv3_{i:03d}_{it['label']}_{os.path.splitext(it['staged_filename'])[0]}.jpg"
    # keep filename reasonably short/clean
    fname = f"racetrainv3_{i:03d}_{it['label']}.jpg"
    src = os.path.join(STAGING, it["staged_filename"])
    dst = os.path.join(TRAIN_IMG_DIR, fname)
    shutil.copyfile(src, dst)
    train_manifest_raw.append({
        "filename": fname,
        "label": it["label"],
        "note": f"source queries: {', '.join(it.get('queries', [])[:2])}",
        "source_title": it["title"],
        "license": it.get("license_short") or "unknown",
        "artist": it.get("artist") or "unknown",
        "source_url": it.get("source_url", ""),
    })

with open(f"{TRAIN_DIR}/training_manifest_raw.json", "w", encoding="utf-8") as f:
    json.dump(train_manifest_raw, f, indent=1, ensure_ascii=False)
print(f"saved {len(train_manifest_raw)} training images to {TRAIN_IMG_DIR}")

# ---- verify zero SHA-256 overlap: eval additions vs training pool vs existing 4-source pool ----
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

with open(f"{EXP}/scratch/existing_hash_pool.json", encoding="utf-8") as f:
    existing_hashes = set(json.load(f))

eval_hashes = {sha256_file(os.path.join(EVAL_IMG_DIR, m["image_id"])) for m in eval_manifest_new}
train_hashes = {sha256_file(os.path.join(TRAIN_IMG_DIR, m["filename"])) for m in train_manifest_raw}

overlap_eval_train = eval_hashes & train_hashes
overlap_eval_existing = eval_hashes & existing_hashes
overlap_train_existing = train_hashes & existing_hashes

print(f"\nLEAKAGE CHECK:")
print(f"  eval additions ({len(eval_hashes)} unique hashes) vs training pool: overlap={len(overlap_eval_train)}")
print(f"  eval additions vs existing 4-source pool ({len(existing_hashes)} hashes): overlap={len(overlap_eval_existing)}")
print(f"  training pool ({len(train_hashes)} unique hashes) vs existing 4-source pool: overlap={len(overlap_train_existing)}")
assert len(overlap_eval_train) == 0, "LEAKAGE: eval additions overlap training pool!"
assert len(overlap_eval_existing) == 0, "LEAKAGE: eval additions overlap existing pool!"
assert len(overlap_train_existing) == 0, "LEAKAGE: training pool overlaps existing pool!"
print("ALL CLEAR: zero SHA-256 overlap across all pools.")

with open(f"{EXP}/scratch/final_leakage_check.json", "w") as f:
    json.dump({
        "eval_additions_n": len(eval_hashes), "train_pool_n": len(train_hashes),
        "existing_pool_n": len(existing_hashes),
        "overlap_eval_train": len(overlap_eval_train),
        "overlap_eval_existing": len(overlap_eval_existing),
        "overlap_train_existing": len(overlap_train_existing),
    }, f, indent=2)
