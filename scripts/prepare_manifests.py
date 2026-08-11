"""
Step 2-4: Dedup, group-aware split, filter/relabel, save manifests.
"""
import os, re, hashlib, json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

RAW_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/_hf_snapshot/test_50k'
DATA_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data'

pattern = re.compile(r'^(\d+)-(.+)\.jpg$')

files = [f for f in os.listdir(RAW_DIR) if f.endswith('.jpg')]
print("total downloaded files:", len(files))

records = []
for f in files:
    m = pattern.match(f)
    if not m:
        continue
    ts, rest = m.groups()
    parts = rest.split('-')
    if len(parts) == 3:
        condition, material, texture = parts
    elif len(parts) == 1:
        condition, material, texture = parts[0], None, None
    elif len(parts) == 2:
        condition, material = parts
        texture = None
    else:
        continue
    records.append({
        'filename': f,
        'filepath': os.path.join(RAW_DIR, f),
        'timestamp': ts,
        'condition': condition,
        'material': material,
        'texture': texture,
        'minute_group': ts[:12],  # YYYYMMDDHHMM
    })

df = pd.DataFrame(records)
print("parsed records:", len(df))

# ---- Step 2: exact duplicate check via MD5 ----
print("\n--- Step 2: dedup ---")
hashes = {}
dupe_count = 0
dupe_groups = Counter()
for idx, row in df.iterrows():
    with open(row['filepath'], 'rb') as fh:
        h = hashlib.md5(fh.read()).hexdigest()
    hashes.setdefault(h, []).append(row['filename'])

dup_files_to_drop = set()
for h, flist in hashes.items():
    if len(flist) > 1:
        dupe_count += len(flist) - 1
        # keep first, drop rest
        for extra in flist[1:]:
            dup_files_to_drop.add(extra)

print("exact duplicate files found (extras beyond first copy):", dupe_count)
print("distinct hash collisions groups:", sum(1 for v in hashes.values() if len(v)>1))

df = df[~df['filename'].isin(dup_files_to_drop)].reset_index(drop=True)
print("records after dedup:", len(df))

# ---- Step 4: filter to asphalt/concrete, map condition ----
print("\n--- Step 4: filter & relabel (before filtering) ---")
before_counts = df.groupby(['condition','material']).size()
print(before_counts)

df_f = df[df['material'].isin(['asphalt', 'concrete'])].copy()
cond_map = {'dry': 'DRY', 'wet': 'DAMP', 'water': 'WET'}
df_f = df_f[df_f['condition'].isin(cond_map.keys())].copy()
df_f['label'] = df_f['condition'].map(cond_map)

print("\nafter filtering to asphalt/concrete + dry/wet/water condition:")
print(df_f['label'].value_counts())
print("\nby condition x material:")
print(df_f.groupby(['condition','material']).size())

# ---- Step 3: group-aware split using minute_group ----
print("\n--- Step 3: group-aware split ---")
n_groups = df_f['minute_group'].nunique()
print("distinct minute-groups in filtered set:", n_groups)
group_sizes = df_f.groupby('minute_group').size()
print("max images per group:", group_sizes.max(), "median:", group_sizes.median())

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_temp = next(gss1.split(df_f, groups=df_f['minute_group']))
df_train = df_f.iloc[idx_train].reset_index(drop=True)
df_temp = df_f.iloc[idx_temp].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
idx_val, idx_test = next(gss2.split(df_temp, groups=df_temp['minute_group']))
df_val = df_temp.iloc[idx_val].reset_index(drop=True)
df_test = df_temp.iloc[idx_test].reset_index(drop=True)

# verify no group leakage
train_groups = set(df_train['minute_group'])
val_groups = set(df_val['minute_group'])
test_groups = set(df_test['minute_group'])
assert not (train_groups & val_groups)
assert not (train_groups & test_groups)
assert not (val_groups & test_groups)
print("\nNo group leakage confirmed across train/val/test.")

print(f"\ntrain: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)}")
print("train label dist:\n", df_train['label'].value_counts())
print("val label dist:\n", df_val['label'].value_counts())
print("test label dist:\n", df_test['label'].value_counts())

# save manifests
for name, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
    out = d[['filepath', 'label', 'minute_group', 'condition', 'material', 'texture', 'filename']].rename(columns={'minute_group':'group_id'})
    out.to_csv(os.path.join(DATA_DIR, f'manifest_{name}.csv'), index=False)
    print("saved", f'manifest_{name}.csv', len(out))

# save audit summary json
summary = {
    'total_downloaded': len(files),
    'parsed_records': len(records),
    'exact_duplicates_removed': dupe_count,
    'records_after_dedup': len(df),
    'before_filter_counts': {f"{k[0]}-{k[1]}": int(v) for k, v in before_counts.items()},
    'after_filter_label_counts': df_f['label'].value_counts().to_dict(),
    'n_minute_groups_filtered': int(n_groups),
    'train_size': len(df_train),
    'val_size': len(df_val),
    'test_size': len(df_test),
    'train_label_dist': df_train['label'].value_counts().to_dict(),
    'val_label_dist': df_val['label'].value_counts().to_dict(),
    'test_label_dist': df_test['label'].value_counts().to_dict(),
}
with open(os.path.join(DATA_DIR, 'audit_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("\nsaved audit_summary.json")
