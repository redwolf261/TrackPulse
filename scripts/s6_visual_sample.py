import json, random
from PIL import Image

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/filtered_records.json') as f:
    records = json.load(f)

random.seed(42)
by_label = {'DRY': [], 'DAMP': [], 'WET': []}
for r in records:
    by_label[r['label_3class']].append(r)

samples = {}
for label, recs in by_label.items():
    n = min(25, len(recs))
    samples[label] = random.sample(recs, n)
    print(f"{label}: sampled {n} of {len(recs)}")

# build contact sheets: 5x5 grid per label
THUMB = 160
COLS = 5
for label, recs in samples.items():
    rows = (len(recs) + COLS - 1) // COLS
    sheet = Image.new('RGB', (COLS*THUMB, rows*THUMB), (30,30,30))
    for i, r in enumerate(recs):
        img = Image.open(r['filepath']).convert('RGB')
        img.thumbnail((THUMB-4, THUMB-4))
        x = (i % COLS) * THUMB
        y = (i // COLS) * THUMB
        sheet.paste(img, (x+2, y+2))
    out_path = f'c:/Users/Rivan/Projects/AI_Grand_Prix/reports/contact_sheet_{label}.png'
    sheet.save(out_path)
    print("saved", out_path)

with open('c:/Users/Rivan/Projects/AI_Grand_Prix/data/visual_samples.json', 'w') as f:
    json.dump({k: [r['filename'] for r in v] for k, v in samples.items()}, f, indent=2)
