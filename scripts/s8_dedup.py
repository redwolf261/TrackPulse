"""
Section 8: duplicate / near-duplicate audit.
- Exact duplicates: SHA-256 over raw file bytes.
- Near-duplicates: perceptual hash (phash) via imagehash, bucketed by minute-group
  (from timestamp) to keep comparisons tractable - only compare within same minute-group,
  since RSCD images are dashcam frames and near-dupes will be temporally close.
"""
import json, hashlib, os
import imagehash
from PIL import Image
import collections

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records_with_quality.json') as f:
    records = json.load(f)

print("deduping", len(records), "filtered images")

# ---- exact duplicates via SHA-256 ----
sha_map = collections.defaultdict(list)
for r in records:
    with open(r['filepath'], 'rb') as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    r['sha256'] = h
    sha_map[h].append(r['filename'])

exact_dup_groups = {h: fs for h, fs in sha_map.items() if len(fs) > 1}
n_exact_dupes_extra = sum(len(fs) - 1 for fs in exact_dup_groups.values())
print(f"\n=== EXACT DUPLICATES (SHA-256) ===")
print(f"distinct hash collision groups: {len(exact_dup_groups)}")
print(f"total extra duplicate files (beyond first copy): {n_exact_dupes_extra}")
for h, fs in list(exact_dup_groups.items())[:5]:
    print(f"  group {h[:12]}...: {fs}")

# mark which filenames are "drop as exact dup" (keep first occurrence in list order)
drop_exact = set()
for h, fs in exact_dup_groups.items():
    for extra in fs[1:]:
        drop_exact.add(extra)

# ---- near-duplicates via perceptual hash, bucketed by minute-group ----
for r in records:
    ts = r['timestamp_raw']
    r['minute_group'] = ts[:12]  # YYYYMMDDHHMM

buckets = collections.defaultdict(list)
for r in records:
    buckets[r['minute_group']].append(r)

print(f"\n=== NEAR-DUPLICATE SEARCH (bucketed by minute-group timestamp prefix) ===")
print(f"number of minute-groups: {len(buckets)}")
multi_buckets = {k: v for k, v in buckets.items() if len(v) > 1}
print(f"minute-groups with >1 image (candidates for near-dup comparison): {len(multi_buckets)}")
total_pairs_compared = sum(len(v)*(len(v)-1)//2 for v in multi_buckets.values())
print(f"total pairwise comparisons needed (only within-bucket): {total_pairs_compared}  (vs {len(records)*(len(records)-1)//2} for naive all-pairs)")

# compute phash for images in multi-image buckets only
phash_cache = {}
def get_phash(r):
    if r['filename'] not in phash_cache:
        img = Image.open(r['filepath']).convert('RGB')
        phash_cache[r['filename']] = imagehash.phash(img)
    return phash_cache[r['filename']]

NEAR_DUP_HAMMING_THRESH = 5  # phash hamming distance threshold (64-bit hash), commonly used conservative threshold
near_dup_pairs = []
for bucket_key, recs in multi_buckets.items():
    hashes = [(r, get_phash(r)) for r in recs]
    for i in range(len(hashes)):
        for j in range(i+1, len(hashes)):
            r1, h1 = hashes[i]
            r2, h2 = hashes[j]
            dist = h1 - h2
            if dist <= NEAR_DUP_HAMMING_THRESH:
                near_dup_pairs.append({'a': r1['filename'], 'b': r2['filename'], 'hamming_dist': int(dist),
                                        'label_a': r1['label_3class'], 'label_b': r2['label_3class']})

print(f"\nnear-duplicate pairs found (phash hamming distance <= {NEAR_DUP_HAMMING_THRESH}): {len(near_dup_pairs)}")
label_mismatch_near_dups = [p for p in near_dup_pairs if p['label_a'] != p['label_b']]
print(f"near-duplicate pairs with DIFFERENT labels (potential label noise / ambiguous transition frames): {len(label_mismatch_near_dups)}")
for p in near_dup_pairs[:10]:
    print(" ", p)

# union-find style clustering of near-dup pairs into clusters
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

involved = set()
for p in near_dup_pairs:
    union(p['a'], p['b'])
    involved.add(p['a']); involved.add(p['b'])

clusters = collections.defaultdict(list)
for fn in involved:
    clusters[find(fn)].append(fn)
print(f"\nnear-duplicate clusters (connected components): {len(clusters)}")
cluster_sizes = collections.Counter(len(v) for v in clusters.values())
print("cluster size distribution:", dict(cluster_sizes))

results = {
    'n_filtered_images': len(records),
    'exact_dup_groups': len(exact_dup_groups),
    'exact_dup_extra_files': n_exact_dupes_extra,
    'exact_dup_filenames_to_drop': sorted(drop_exact),
    'n_minute_groups': len(buckets),
    'n_multi_image_minute_groups': len(multi_buckets),
    'pairwise_comparisons_done': total_pairs_compared,
    'near_dup_hamming_threshold': NEAR_DUP_HAMMING_THRESH,
    'near_dup_pairs_found': len(near_dup_pairs),
    'near_dup_pairs_with_label_mismatch': len(label_mismatch_near_dups),
    'near_dup_clusters': len(clusters),
    'near_dup_cluster_size_distribution': {str(k): v for k, v in cluster_sizes.items()},
    'near_dup_pairs_sample': near_dup_pairs[:50],
}
with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/s8_dedup_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

# save records with sha256 + minute_group + near_dup flag for downstream use
near_dup_filenames = involved
for r in records:
    r['near_dup_flag'] = r['filename'] in near_dup_filenames
    r['exact_dup_drop'] = r['filename'] in drop_exact

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records_deduped.json', 'w') as f:
    json.dump(records, f)

print("\nsaved s8_dedup_summary.json and filtered_records_deduped.json")
