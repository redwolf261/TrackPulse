import hashlib, json, os, glob

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
DIRS = [
    f"{ROOT}/data/racing_spotcheck/images",
    f"{ROOT}/data/racing_spotcheck/images2",
    f"{ROOT}/data/racing_spotcheck_v2/images",
    f"{ROOT}/data/racing_spotcheck_v2/images2",
    f"{ROOT}/data/racing_train_pool/images",
    f"{ROOT}/data/racing_train_pool_v2/images",
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

pool = {}
for d in DIRS:
    if not os.path.isdir(d):
        print("MISSING DIR:", d)
        continue
    files = [p for p in glob.glob(os.path.join(d, "*")) if os.path.isfile(p)]
    for p in files:
        try:
            h = sha256_file(p)
        except Exception as e:
            print("hash fail", p, e)
            continue
        pool[h] = pool.get(h, []) + [p]
    print(d, "->", len(files), "files")

print("total unique hashes across all 4 sources:", len(pool))
dupes = {h: v for h, v in pool.items() if len(v) > 1}
print("internal duplicate hashes (across existing pools):", len(dupes))
for h, v in list(dupes.items())[:10]:
    print(" dup:", v)

with open(f"{ROOT}/experiments/exp05_data_moat/scratch/existing_hash_pool.json", "w") as f:
    json.dump(list(pool.keys()), f)
print("saved existing_hash_pool.json with", len(pool), "hashes")
