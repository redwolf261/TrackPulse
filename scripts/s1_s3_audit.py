"""
Sections 1-3: dataset structure, exact counts, label ontology reconstruction.
Parses ONLY from actual local filenames on disk - no assumptions.
"""
import os, re, json, collections

ROOT = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/_hf_snapshot/test_50k'
OUT = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data'

files = sorted(os.listdir(ROOT))
files = [f for f in files if f.lower().endswith('.jpg')]
print("TOTAL FILES ON DISK:", len(files))

pattern = re.compile(r'^(\d+)-(.+)\.jpg$', re.IGNORECASE)
records = []
unparsed = []
for f in files:
    m = pattern.match(f)
    if not m:
        unparsed.append(f)
        continue
    ts, rest = m.groups()
    parts = rest.split('-')
    if len(parts) == 3:
        condition, material, surface = parts
    elif len(parts) == 2:
        condition, material = parts
        surface = None
    elif len(parts) == 1:
        condition, material, surface = parts[0], None, None
    else:
        # more than 3 dash-separated fields - flag it
        condition, material, surface = parts[0], parts[1] if len(parts)>1 else None, '-'.join(parts[2:])
    records.append({
        'filename': f,
        'filepath': os.path.join(ROOT, f).replace('\\','/'),
        'timestamp_raw': ts,
        'condition': condition,
        'material': material,
        'surface': surface,
        'n_dash_fields': len(parts),
    })

print("PARSED RECORDS:", len(records))
print("UNPARSED FILENAMES:", len(unparsed))
for u in unparsed[:20]:
    print("  UNPARSED:", u)

# distinct field vocab
cond_vocab = collections.Counter(r['condition'] for r in records)
mat_vocab = collections.Counter(r['material'] for r in records)
surf_vocab = collections.Counter(r['surface'] for r in records)
ndash_dist = collections.Counter(r['n_dash_fields'] for r in records)

print("\n=== CONDITION VOCAB (exact strings found) ===")
for k, v in cond_vocab.most_common():
    print(f"  {k}: {v}")

print("\n=== MATERIAL VOCAB (exact strings found, including None) ===")
for k, v in mat_vocab.most_common():
    print(f"  {k}: {v}")

print("\n=== SURFACE VOCAB (exact strings found, including None) ===")
for k, v in surf_vocab.most_common():
    print(f"  {k}: {v}")

print("\n=== FIELD COUNT DISTRIBUTION (dash-separated segments after timestamp) ===")
for k, v in sorted(ndash_dist.items()):
    print(f"  {k} fields: {v} files")

# full label string distribution
label_full = collections.Counter(
    f"{r['condition']}" + (f"-{r['material']}" if r['material'] else "") + (f"-{r['surface']}" if r['surface'] else "")
    for r in records
)
print(f"\n=== FULL LABEL STRING DISTRIBUTION ({len(label_full)} distinct) ===")
for k, v in sorted(label_full.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# timestamp length variability check
ts_lens = collections.Counter(len(r['timestamp_raw']) for r in records)
print("\n=== TIMESTAMP STRING LENGTH DISTRIBUTION ===")
for k, v in sorted(ts_lens.items()):
    print(f"  length {k}: {v} files")

with open(os.path.join(OUT, 'raw_parsed_records.json'), 'w') as f:
    json.dump(records, f)
print("\nsaved raw_parsed_records.json with", len(records), "records")

summary = {
    'total_files_on_disk': len(files),
    'parsed_records': len(records),
    'unparsed_filenames': unparsed,
    'condition_vocab': dict(cond_vocab),
    'material_vocab': {str(k): v for k, v in mat_vocab.items()},
    'surface_vocab': {str(k): v for k, v in surf_vocab.items()},
    'full_label_distribution': dict(label_full),
    'timestamp_length_distribution': {str(k): v for k, v in ts_lens.items()},
}
with open(os.path.join(OUT, 's1_s3_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("saved s1_s3_summary.json")
