import json, collections
from datetime import datetime

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records_deduped.json') as f:
    records = json.load(f)

print("=== Section 9: timestamp/sequence analysis on filtered set (n=%d) ===" % len(records))

def parse_dt(ts):
    base = ts[:14]
    try:
        return datetime.strptime(base, "%Y%m%d%H%M%S")
    except ValueError:
        return None

for r in records:
    r['parsed_dt'] = parse_dt(r['timestamp_raw'])

parseable = [r for r in records if r['parsed_dt'] is not None]
print(f"timestamps parseable as YYYYMMDDHHMMSS: {len(parseable)} / {len(records)}")

dts = sorted([r['parsed_dt'] for r in parseable])
print(f"date range: {dts[0]} to {dts[-1]}")

gaps = [(dts[i]-dts[i-1]).total_seconds() for i in range(1, len(dts))]
import statistics
print(f"consecutive-sorted-timestamp gaps: n={len(gaps)} median={statistics.median(gaps)}s min={min(gaps)}s max={max(gaps)}s")
print(f"gaps <=5s: {sum(1 for g in gaps if g<=5)}  <=60s: {sum(1 for g in gaps if g<=60)}")

minute_groups = collections.Counter(r['minute_group'] for r in records)
print(f"\ndistinct minute-groups in filtered set: {len(minute_groups)}")
multi = {k:v for k,v in minute_groups.items() if v>1}
print(f"minute-groups with >1 image: {len(multi)}  (max images in one group: {max(minute_groups.values())})")

# does a minute-group ever contain mixed labels? (would indicate label changes within a short session - expected at wet/dry transitions)
label_by_group = collections.defaultdict(set)
for r in records:
    label_by_group[r['minute_group']].add(r['label_3class'])
mixed_label_groups = {k:v for k,v in label_by_group.items() if len(v)>1}
print(f"\nminute-groups containing MORE THAN ONE label (mixed DRY/DAMP/WET within same minute): {len(mixed_label_groups)}")
for k,v in list(mixed_label_groups.items())[:5]:
    print(f"  {k}: {v}")

summary = {
    'n_filtered': len(records),
    'n_timestamp_parseable': len(parseable),
    'date_range': [str(dts[0]), str(dts[-1])],
    'gap_stats_seconds': {'median': statistics.median(gaps), 'min': min(gaps), 'max': max(gaps),
                           'n_le_5s': sum(1 for g in gaps if g<=5), 'n_le_60s': sum(1 for g in gaps if g<=60)},
    'n_minute_groups': len(minute_groups),
    'n_multi_image_minute_groups': len(multi),
    'max_images_per_minute_group': max(minute_groups.values()),
    'n_mixed_label_minute_groups': len(mixed_label_groups),
}
with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/s9_timestamp_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("\nsaved s9_timestamp_summary.json")
