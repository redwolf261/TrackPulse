from huggingface_hub import HfApi
import re, json, collections

api = HfApi()
info = api.dataset_info('rezzzq/RSCD-1million')
files = [s.rfilename for s in info.siblings if s.rfilename.startswith('test_50k/') and s.rfilename.endswith('.jpg')]
print("total files in test_50k:", len(files))

pattern = re.compile(r'^test_50k/(\d+)-(.+)\.jpg$')
label_counts = collections.Counter()
unparsed = []
timestamps = []
condition_set = set()
material_set = set()
texture_set = set()

for f in files:
    m = pattern.match(f)
    if not m:
        unparsed.append(f)
        continue
    ts, rest = m.groups()
    timestamps.append(ts)
    parts = rest.split('-')
    label_counts[rest] += 1
    if len(parts) == 3:
        condition_set.add(parts[0])
        material_set.add(parts[1])
        texture_set.add(parts[2])
    elif len(parts) == 1:
        condition_set.add(parts[0])
    elif len(parts) == 2:
        condition_set.add(parts[0])
        material_set.add(parts[1])

print("\nunparsed filenames:", len(unparsed))
for u in unparsed[:10]:
    print(" ", u)

print("\ndistinct condition-material-texture label strings:", len(label_counts))
for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {lbl}: {cnt}")

print("\ndistinct condition tokens:", sorted(condition_set))
print("distinct material tokens:", sorted(material_set))
print("distinct texture tokens:", sorted(texture_set))

# save full listing for later use
with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/all_filenames.json', 'w') as fh:
    json.dump(files, fh)
