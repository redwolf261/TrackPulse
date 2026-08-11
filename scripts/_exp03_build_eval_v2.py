import json, os, shutil, hashlib

ROOT = 'c:/Users/Rivan/Projects/AI_Grand_Prix'
sp = r'C:\Users\Rivan\AppData\Local\Temp\claude\c--Users-Rivan-Projects-AI-Grand-Prix\4410f2bc-e013-48f8-a62f-ddc3499ff722\scratchpad\damp_holdout_filenames_resolved.json'
with open(sp, encoding='utf-8') as f:
    resolved = json.load(f)

dest_dir = f'{ROOT}/data/racing_spotcheck_v2/damp_holdout'
os.makedirs(dest_dir, exist_ok=True)

with open(f'{ROOT}/data/racing_spotcheck/ground_truth_manifest.json', encoding='utf-8') as f:
    orig49 = json.load(f)

new_entries = []
for i, item in enumerate(resolved):
    fn = item['filename']
    src = item['path']
    e = item['entry']
    dst_name = f"damp_holdout_{i:02d}_{fn}"
    dst_path = os.path.join(dest_dir, dst_name)
    shutil.copy2(src, dst_path)
    new_entries.append({
        "image_id": dst_name,
        "filepath": dst_path.replace(chr(92), '/'),
        "source_event": e.get('note', '') or e.get('source_title', ''),
        "clip_id": "UNKNOWN",
        "camera_type": "trackside",
        "ground_truth": "DAMP",
        "license": e.get('license', 'UNKNOWN'),
        "source_title": e.get('source_title', ''),
        "note": "held out from " + ('racing_train_pool' if fn.startswith('racetrain_') else 'racing_train_pool_v2') + " for exp03 DAMP eval; orig train note: " + e.get('note', ''),
        "orig_train_pool_filename": fn,
    })

orig49_v2 = []
for e in orig49:
    e2 = dict(e)
    e2['filepath'] = e['filepath'].replace('racing_spotcheck', 'racing_spotcheck_v2')
    orig49_v2.append(e2)

combined = orig49_v2 + new_entries
with open(f'{ROOT}/data/racing_spotcheck_v2/ground_truth_manifest.json', 'w', encoding='utf-8') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print("combined manifest entries:", len(combined), "= 49 orig +", len(new_entries), "damp holdout")

mismatches = 0
for item in resolved:
    with open(item['path'], 'rb') as f:
        h_orig = hashlib.sha256(f.read()).hexdigest()
    if h_orig != item['sha256']:
        mismatches += 1
print("resolved-list self-consistency mismatches:", mismatches)
