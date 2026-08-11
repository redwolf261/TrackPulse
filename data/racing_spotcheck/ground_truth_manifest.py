"""
Ground-truth labels assigned via visual inspection (label-first, no model peeking).
Combines original 13 wet images (images/) + 36 new contrast images (images2/).
"""
import json, os

DATA_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/data/racing_spotcheck'

# ---- original 13 wet images ----
with open(os.path.join(DATA_DIR, 'source_manifest.json'), encoding='utf-8') as f:
    orig = json.load(f)

orig_meta = {
    'racing_00.jpg': {'event': '16 Maldonado Wet (2013 Malaysia)', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_01.jpg': {'event': '2011 Canadian GP - Force India', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_02.jpg': {'event': '2011 Canadian GP - Race stop', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_03.jpg': {'event': '2011 Canadian GP - Water', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_04.jpg': {'event': '2012 British GP - Kovalainen', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_05.jpg': {'event': '2012 British GP - Lotus', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_06.jpg': {'event': '2012 British GP - Massa', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_07.jpg': {'event': 'Adrian Sutil 2010 Malaysia 3rd Qualify', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_08.jpg': {'event': 'Alguersuari Malaysian Qualy 2010', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_09.jpg': {'event': 'Safety Car in Heavy Rain (Mercedes AMG GT S)', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_10.jpg': {'event': 'Eau Rouge McLaren (2011 Belgium, rain/fog)', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_11.jpg': {'event': 'Felipe Massa (wet spray)', 'camera_type': 'trackside', 'gt': 'WET'},
    'racing_12.jpg': {'event': 'Fernando Alonso Belgium GP 2010', 'camera_type': 'trackside', 'gt': 'WET'},
}

# ---- new 36 images: ground truth from visual inspection above ----
new_labels = {
    'contrast_00.jpg': {'event': '2015 Singapore GP pit lane (Ferrari)', 'camera_type': 'trackside', 'gt': 'AMBIGUOUS', 'note': 'track surface obscured by fence/crew'},
    'contrast_01.jpg': {'event': '2019 Turkey/China pit lane (Massa/Ferrari/Hamilton boxes)', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_02.jpg': {'event': 'Catalunya test 2011 - Lotus pitstop (overhead)', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_03.jpg': {'event': '2012 Italian GP Monza pit lane', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_04.jpg': {'event': '2012 Italian GP Ferrari pit', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_05.jpg': {'event': '2012 Italian GP Massa pit', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_06.jpg': {'event': 'Alonso Renault Pitstop Chinese GP 2008', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_07.jpg': {'event': 'Aston Martin pit USGP 2021', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_08.jpg': {'event': 'F1 2011 AUS pit (Red Bull/Toro Rosso)', 'camera_type': 'trackside', 'gt': 'AMBIGUOUS', 'note': 'minimal surface visible'},
    'contrast_09.JPG': {'event': 'F1 Turkey GP 2021 garage (Red Bull livery reveal)', 'camera_type': 'trackside', 'gt': 'AMBIGUOUS', 'note': 'overcast garage, minimal surface visible'},
    'contrast_10.jpg': {'event': 'Fale F1 Monza 2004 Sauber pit', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_11.jpg': {'event': 'F1 fan pitstop challenge activation (not a real track)', 'camera_type': 'UNKNOWN', 'gt': 'AMBIGUOUS', 'note': 'not genuine track surface, indoor rubber mat flooring with visible damp footprints'},
    'contrast_12.jpg': {'event': '2008 British GP Silverstone pit lane wide', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_13.jpg': {'event': '2008 British GP Renault pitstop', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_14.jpg': {'event': '2008 British GP Williams pitstop', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_15.jpg': {'event': '2008 British GP Force India on track', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_16.jpg': {'event': '2008 British GP BMW/Renault side by side', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_17.jpg': {'event': '2008 British GP through-fence view', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_18.jpg': {'event': '2008 British GP crashed Ferrari on flatbed', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_19.jpg': {'event': '2008 British GP McLaren wet running', 'camera_type': 'trackside', 'gt': 'WET'},
    'contrast_20.jpg': {'event': '2008 British GP McLaren dry running', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_21.jpg': {'event': '2008 British GP Toyota on track', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_22.jpg': {'event': '2016 Bahrain GP start (night)', 'camera_type': 'trackside', 'gt': 'DRY', 'note': 'floodlit night race, potential brightness/reflection confound'},
    'contrast_23.jpg': {'event': '2016 Bahrain GP Williams testing (day)', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_24.jpg': {'event': '2016 Bahrain GP McLaren-Honda (night)', 'camera_type': 'trackside', 'gt': 'DRY', 'note': 'floodlit night race'},
    'contrast_25.jpg': {'event': '2016 Bahrain GP Haas (day)', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_26.jpg': {'event': '2016 Bahrain GP Haas lockup (night, tire smoke)', 'camera_type': 'trackside', 'gt': 'DRY', 'note': 'white tire smoke could be confused with spray'},
    'contrast_27.jpg': {'event': '2016 Bahrain GP Mercedes lockup (night, tire smoke)', 'camera_type': 'trackside', 'gt': 'DRY', 'note': 'white tire smoke could be confused with spray'},
    'contrast_28.jpg': {'event': '2015 Singapore GP Mercedes (night)', 'camera_type': 'trackside', 'gt': 'DRY', 'note': 'floodlit night race, motion blur'},
    'contrast_29.jpg': {'event': '2016 Bahrain GP Red Bull (day)', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_30.jpg': {'event': '2015 Singapore GP Lotus lockup w/ sparks (night)', 'camera_type': 'trackside', 'gt': 'DRY', 'note': 'sparks + smoke, night'},
    'contrast_31.jpg': {'event': '2016 Bahrain GP Williams (day)', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_32.jpg': {'event': '2016 Bahrain GP Renault (day)', 'camera_type': 'trackside', 'gt': 'DRY'},
    'contrast_33.jpg': {'event': '2015 Singapore GP Red Bull (night)', 'camera_type': 'trackside', 'gt': 'DRY', 'note': 'floodlit night race'},
    'contrast_34.jpg': {'event': '2015 Singapore GP Rossi crash recovery (night)', 'camera_type': 'trackside', 'gt': 'AMBIGUOUS', 'note': 'no track surface visible, crash/crane scene'},
    'contrast_35.jpg': {'event': '2015 Singapore GP Ferrari wide overhead (night)', 'camera_type': 'trackside', 'gt': 'DRY'},
}

with open(os.path.join(DATA_DIR, 'source_manifest2.json'), encoding='utf-8') as f:
    new_manifest = json.load(f)

full_records = []
for r in orig:
    meta = orig_meta[r['filename']]
    full_records.append({
        'image_id': r['filename'],
        'filepath': os.path.join(DATA_DIR, 'images', r['filename']).replace('\\', '/'),
        'source_event': meta['event'],
        'clip_id': 'UNKNOWN',
        'camera_type': meta['camera_type'],
        'ground_truth': meta['gt'],
        'license': r['license'],
        'source_title': r['source_title'],
        'note': meta.get('note', ''),
    })

for r in new_manifest:
    meta = new_labels[r['filename']]
    full_records.append({
        'image_id': r['filename'],
        'filepath': os.path.join(DATA_DIR, 'images2', r['filename']).replace('\\', '/'),
        'source_event': meta['event'],
        'clip_id': 'UNKNOWN',
        'camera_type': meta['camera_type'],
        'ground_truth': meta['gt'],
        'license': r['license'],
        'source_title': r['source_title'],
        'note': meta.get('note', ''),
    })

print(f"total records: {len(full_records)}")
import collections
gt_counts = collections.Counter(r['ground_truth'] for r in full_records)
print("ground truth distribution:", dict(gt_counts))

with open(os.path.join(DATA_DIR, 'ground_truth_manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(full_records, f, indent=2, ensure_ascii=False)
print("saved ground_truth_manifest.json")
