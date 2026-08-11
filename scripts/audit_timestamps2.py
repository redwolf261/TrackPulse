import json, re
from datetime import datetime
import collections

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/all_filenames.json') as f:
    files = json.load(f)

pattern = re.compile(r'^test_50k/(\d+)-(.+)\.jpg$')
recs = []
for f in files:
    m = pattern.match(f)
    ts, rest = m.groups()
    recs.append((ts, rest, f))

recs.sort(key=lambda x: x[0])

def parse_dt(ts):
    # first 14 digits = YYYYMMDDHHMMSS
    base = ts[:14]
    try:
        return datetime.strptime(base, "%Y%m%d%H%M%S")
    except ValueError:
        return None

dts = [parse_dt(r[0]) for r in recs]
gaps_sec = []
for i in range(1, len(dts)):
    if dts[i] and dts[i-1]:
        gaps_sec.append((dts[i]-dts[i-1]).total_seconds())

import statistics
print("num consecutive gaps:", len(gaps_sec))
print("gaps <=1s:", sum(1 for g in gaps_sec if g<=1))
print("gaps <=5s:", sum(1 for g in gaps_sec if g<=5))
print("gaps <=60s:", sum(1 for g in gaps_sec if g<=60))
print("median gap (s):", statistics.median(gaps_sec))
print("min gap:", min(gaps_sec), "max gap:", max(gaps_sec))

# group by same-minute (YYYYMMDDHHMM) to define "burst" groups
minute_groups = collections.Counter(r[0][:12] for r in recs)
multi = [k for k,v in minute_groups.items() if v>1]
print("\nnum distinct minute-groups:", len(minute_groups))
print("minute-groups with >1 image:", len(multi))
print("max images in one minute-group:", max(minute_groups.values()))

# same 14-digit second bucket
sec_groups = collections.Counter(r[0][:14] for r in recs)
multi_sec = [k for k,v in sec_groups.items() if v>1]
print("\nnum distinct second-groups:", len(sec_groups))
print("second-groups with >1 image:", len(multi_sec))
