import json, collections

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/raw_parsed_records.json') as f:
    records = json.load(f)

print("=== Section 4/5: task definition + material filtering ===\n")

# Before filtering: full breakdown
cond_mat = collections.Counter((r['condition'], r['material']) for r in records)
print("condition x material breakdown (material=None means no material field, e.g. ice/snow/mud/gravel-as-condition):")
for k, v in sorted(cond_mat.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Filter to asphalt/concrete material AND condition in dry/wet/water
relevant = [r for r in records if r['material'] in ('asphalt', 'concrete') and r['condition'] in ('dry', 'wet', 'water')]
print(f"\nAfter filtering material in {{asphalt, concrete}} AND condition in {{dry, wet, water}}: {len(relevant)} images remain (of {len(records)} total, {len(records)-len(relevant)} excluded)")

excl_reason = collections.Counter()
for r in records:
    if r['material'] not in ('asphalt', 'concrete'):
        excl_reason[f"material={r['material']}"] += 1
    elif r['condition'] not in ('dry','wet','water'):
        excl_reason[f"condition={r['condition']} (material was asphalt/concrete)"] += 1
print("\nexclusion reasons:")
for k, v in excl_reason.most_common():
    print(f"  {k}: {v}")

label_map = {'dry': 'DRY', 'wet': 'DAMP', 'water': 'WET'}
for r in relevant:
    r['label_3class'] = label_map[r['condition']]

after_counts = collections.Counter(r['label_3class'] for r in relevant)
print("\n3-class label distribution AFTER filtering (Option B: dry/wet/water -> DRY/DAMP/WET):")
for k, v in sorted(after_counts.items()):
    print(f"  {k}: {v} ({100*v/len(relevant):.1f}%)")

# also material breakdown within relevant
mat_counts = collections.Counter(r['material'] for r in relevant)
print("\nmaterial distribution within filtered set:")
for k, v in mat_counts.items():
    print(f"  {k}: {v}")

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records.json', 'w') as f:
    json.dump(relevant, f)
print(f"\nsaved filtered_records.json with {len(relevant)} records")
