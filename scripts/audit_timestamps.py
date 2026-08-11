import json, re
from datetime import datetime

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/all_filenames.json') as f:
    files = json.load(f)

pattern = re.compile(r'^test_50k/(\d+)-(.+)\.jpg$')
recs = []
for f in files:
    m = pattern.match(f)
    ts, rest = m.groups()
    recs.append((ts, rest, f))

recs.sort(key=lambda x: x[0])
print("earliest:", recs[0])
print("latest:", recs[-1])

# check consecutive timestamp gaps to see if bursts cluster
gaps = []
for i in range(1, len(recs)):
    t0 = recs[i-1][0][:14]  # yyyymmddhhmmss prefix common length
    t1 = recs[i][0][:14]
    gaps.append((recs[i][0], recs[i][1]))

# print a slice to see patterns
for r in recs[:40]:
    print(r)
