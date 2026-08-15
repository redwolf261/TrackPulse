import hashlib, json, os, glob

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

POOLS = {
    "spotcheck_v1_images": f"{ROOT}/data/racing_spotcheck/images",
    "spotcheck_v1_images2": f"{ROOT}/data/racing_spotcheck/images2",
    "spotcheck_v2_images": f"{ROOT}/data/racing_spotcheck_v2/images",
    "spotcheck_v2_images2": f"{ROOT}/data/racing_spotcheck_v2/images2",
    "train_pool_v1": f"{ROOT}/data/racing_train_pool/images",
    "train_pool_v2": f"{ROOT}/data/racing_train_pool_v2/images",
    "train_pool_v3": f"{ROOT}/data/racing_train_pool_v3/images",
    "spotcheck_v3_new": f"{ROOT}/data/racing_spotcheck_v3/images_new",
}

hashes = {}
for name, d in POOLS.items():
    files = [p for p in glob.glob(os.path.join(d, "*")) if os.path.isfile(p)]
    hs = set()
    for p in files:
        hs.add(sha256_file(p))
    hashes[name] = hs
    print(f"{name}: {len(files)} files, {len(hs)} unique hashes")

print("\nPairwise overlap checks (new pools vs everything, including each other):")
new_pools = ["train_pool_v3", "spotcheck_v3_new"]
all_pools = list(POOLS.keys())
any_leak = False
for new in new_pools:
    for other in all_pools:
        if other == new:
            continue
        overlap = hashes[new] & hashes[other]
        if overlap:
            any_leak = True
            print(f"  LEAK: {new} vs {other}: {len(overlap)} overlapping hashes")
if not any_leak:
    print("  NONE. Zero overlap confirmed for both new pools against all other pools "
          "(including each other).")

total_unique_project_wide = set()
for hs in hashes.values():
    total_unique_project_wide |= hs
print(f"\ntotal unique racing-domain images project-wide (all 8 dirs, dedup by hash): {len(total_unique_project_wide)}")
