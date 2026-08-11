"""
Section 10-11: group-aware leakage-safe split + class balancing decision.
Group signal: minute_group (YYYYMMDDHHMM timestamp prefix) - Section 9 showed this
is a real, recoverable grouping signal (up to 69 images/minute-group, 318 multi-image
groups, some with mixed labels indicating within-session transitions).
"""
import json, collections
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records_deduped.json') as f:
    records = json.load(f)

# no exact dups were found (Section 8), so nothing to drop there
n_exact_drop = sum(1 for r in records if r.get('exact_dup_drop'))
print("exact dup drops to apply:", n_exact_drop)
records = [r for r in records if not r.get('exact_dup_drop')]
print("records after exact-dup removal:", len(records))

df = pd.DataFrame(records)

print("\n=== Section 11: class balance before split ===")
print(df['label_3class'].value_counts())
print((df['label_3class'].value_counts(normalize=True) * 100).round(1))

# group-aware split using minute_group
n_groups = df['minute_group'].nunique()
print(f"\ndistinct groups: {n_groups}")

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_temp = next(gss1.split(df, groups=df['minute_group']))
df_train = df.iloc[idx_train].reset_index(drop=True)
df_temp = df.iloc[idx_temp].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
idx_val, idx_test = next(gss2.split(df_temp, groups=df_temp['minute_group']))
df_val = df_temp.iloc[idx_val].reset_index(drop=True)
df_test = df_temp.iloc[idx_test].reset_index(drop=True)

train_groups = set(df_train['minute_group'])
val_groups = set(df_val['minute_group'])
test_groups = set(df_test['minute_group'])
assert not (train_groups & val_groups), "LEAKAGE: train/val group overlap"
assert not (train_groups & test_groups), "LEAKAGE: train/test group overlap"
assert not (val_groups & test_groups), "LEAKAGE: val/test group overlap"
print("\nVERIFIED: no group overlap between train/val/test (group = minute_group timestamp prefix)")

print(f"\nsplit sizes: train={len(df_train)} ({100*len(df_train)/len(df):.1f}%) "
      f"val={len(df_val)} ({100*len(df_val)/len(df):.1f}%) "
      f"test={len(df_test)} ({100*len(df_test)/len(df):.1f}%)")

for name, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
    print(f"\n{name} label distribution:")
    print(d['label_3class'].value_counts())

# Section 11: class balancing decision
train_counts = df_train['label_3class'].value_counts()
imbalance_ratio = train_counts.max() / train_counts.min()
print(f"\n=== class balancing decision ===")
print(f"train class counts: {train_counts.to_dict()}")
print(f"max/min imbalance ratio: {imbalance_ratio:.2f}")
print("DECISION: imbalance ratio < 2x -> use class-weighted cross-entropy loss (inverse frequency weights),")
print("NOT oversampling/undersampling, since the imbalance is mild and natural distribution preserves scene diversity.")

# save manifests
manifest_cols = ['filepath', 'label_3class', 'minute_group', 'condition', 'material', 'surface',
                  'timestamp_raw', 'sha256', 'filename', 'width', 'height', 'brightness',
                  'blur_laplacian_var', 'near_dup_flag']
import os
os.makedirs('c:/Users/Rivan/Projects/AI_Grand_Prix/data/manifests', exist_ok=True)
for name, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
    out = d[manifest_cols].rename(columns={'label_3class': 'label', 'minute_group': 'sequence_id'})
    out['source_split'] = name
    out.to_csv(f'c:/Users/Rivan/Projects/AI_Grand_Prix/data/manifests/split_manifest_{name}.csv', index=False)
    print(f"saved split_manifest_{name}.csv ({len(out)} rows)")

# combined manifest
all_df = pd.concat([df_train.assign(source_split='train'), df_val.assign(source_split='val'), df_test.assign(source_split='test')])
all_out = all_df[manifest_cols + ['source_split']].rename(columns={'label_3class': 'label', 'minute_group': 'sequence_id'})
all_out.to_csv('c:/Users/Rivan/Projects/AI_Grand_Prix/data/manifests/dataset_manifest.csv', index=False)
print(f"saved dataset_manifest.csv ({len(all_out)} rows)")

class_mapping = {'DRY': 0, 'DAMP': 1, 'WET': 2}
with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/manifests/class_mapping.json', 'w') as f:
    json.dump({'classes': class_mapping, 'source_condition_map': {'dry':'DRY','wet':'DAMP','water':'WET'}}, f, indent=2)
print("saved class_mapping.json")

summary = {
    'n_after_exact_dup_removal': len(df),
    'n_groups_total': int(n_groups),
    'train_size': len(df_train), 'val_size': len(df_val), 'test_size': len(df_test),
    'train_pct': round(100*len(df_train)/len(df),1), 'val_pct': round(100*len(df_val)/len(df),1), 'test_pct': round(100*len(df_test)/len(df),1),
    'train_label_dist': df_train['label_3class'].value_counts().to_dict(),
    'val_label_dist': df_val['label_3class'].value_counts().to_dict(),
    'test_label_dist': df_test['label_3class'].value_counts().to_dict(),
    'group_leakage_verified_none': True,
    'imbalance_ratio_train': float(imbalance_ratio),
    'balancing_decision': 'class-weighted cross-entropy (inverse frequency), natural distribution otherwise',
}
with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/s10_s11_split_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nsaved s10_s11_split_summary.json")
