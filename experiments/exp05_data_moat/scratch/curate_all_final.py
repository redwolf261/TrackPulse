"""
Final full curation across the merged 1253-image staged set
(staged_index4.json = staged_accepted4.json sorted by filename, which is a
superset of round2's 450 + round3's 803 new images). Labels are based on
individual visual verification via contact sheets (sheets3/, sheets4/,
sheets_rain/) plus targeted full-resolution spot-checks for DAMP candidates.
"""
import json

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"

with open(f"{EXP}/scratch/staged_index4.json", encoding="utf-8") as f:
    items = json.load(f)

def rng(lo, hi):
    return list(range(lo, hi + 1))

# ============ ROUND 2 (idx 0-449), from curate_round3_final.py ============
DRY_R2 = (
    rng(0, 4) + rng(5, 6) + [9, 10] +
    rng(20, 22) + [24] +
    rng(40, 47) + [49] +
    rng(243, 249) + rng(250, 254) + rng(260, 264) +
    rng(266, 269) + rng(270, 272) +
    rng(275, 280) + [283] +
    rng(285, 286) + [288, 289] + [293, 294, 295] + [297, 298, 299] +
    [300, 302] +
    rng(325, 327) + [332, 333] + rng(337, 338) + rng(342, 353) +
    rng(364, 372) +
    rng(375, 383) + [385, 389, 390, 392, 393, 394] +
    [396, 402, 404, 405, 407, 408, 411, 413, 414, 416] +
    [420, 422, 423, 424, 425, 426, 428, 429, 430, 435, 437, 444, 445, 449]
)
WET_R2 = (
    rng(60, 79) + rng(90, 102) + [104, 106] + rng(108, 121) +
    rng(124, 128) + [131, 134, 135, 136, 138, 139] + [140, 141, 143]
)
DAMP_R2 = [80, 81, 82, 87, 88, 89]

# ============ ROUND 3 (idx 450-1252), 2016 Monaco GP / 2011-12 Canadian GP /
# 2007-2008/2014/2019 Japanese GP / historic Ferrari 1980s test-track ============

# 2016 Monaco GP: wet start (450-471ish), drying/dry from ~488 onward.
WET_R3_monaco = rng(450, 456) + [458, 462, 463, 464, 465, 467, 469, 471, 472] + \
    [645, 646, 647, 650, 651, 652, 653, 654, 655, 656]
DRY_R3_monaco = (
    rng(488, 499) + rng(500, 549) + rng(550, 599) + rng(658, 664) +
    rng(666, 674)
)

# 2011-2012 Canadian GP (idx ~900-950): wet-to-dry transition, best DAMP finds.
WET_R3_canadian = rng(902, 909) + [1012]
DAMP_R3_canadian = [911, 912, 913, 916]  # verified individually at full-res:
# matte-grey damp track, no spray, drying-phase lighting
DRY_R3_canadian = (
    rng(917, 949) + rng(953, 974) + rng(1025, 1074) + rng(1076, 1099)
)

# 2007/2008/2014/2019 Japanese GP (Suzuka/Fuji, idx ~1125-1249): mixed wet/dry.
WET_R3_japan = [1127, 1136, 1137, 1141, 1148, 1150, 1152, 1160]
DRY_R3_japan = (
    rng(1128, 1135) + [1138, 1139] + rng(1142, 1147) + [1149] +
    [1151, 1153, 1155, 1157, 1159] + rng(1161, 1249)  # broad dry sweep, sunny Fuji/Suzuka
)

# Historic Ferrari 1980s test track (1251, 1252): dry.
DRY_R3_historic = [1251, 1252]

def dedupe(lst):
    seen = set(); out = []
    for x in lst:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

ALL_DRY = dedupe(DRY_R2 + DRY_R3_monaco + DRY_R3_canadian + DRY_R3_japan + DRY_R3_historic)
ALL_WET = dedupe(WET_R2 + WET_R3_monaco + WET_R3_canadian + WET_R3_japan)
ALL_DAMP = dedupe(DAMP_R2 + DAMP_R3_canadian)

# resolve overlaps: DAMP > WET > DRY priority (DAMP most carefully verified)
damp_set = set(ALL_DAMP)
wet_set = set(ALL_WET) - damp_set
dry_set = set(ALL_DRY) - damp_set - wet_set
# drop any index out of range (safety)
n = len(items)
damp_set = {i for i in damp_set if 0 <= i < n}
wet_set = {i for i in wet_set if 0 <= i < n}
dry_set = {i for i in dry_set if 0 <= i < n}

print(f"DAMP: {len(damp_set)}  WET: {len(wet_set)}  DRY: {len(dry_set)}")
print("total curated:", len(damp_set) + len(wet_set) + len(dry_set), "/", n, "staged")

curated = []
for idx in sorted(damp_set):
    curated.append({"idx": idx, "label": "DAMP", **items[idx]})
for idx in sorted(wet_set):
    curated.append({"idx": idx, "label": "WET", **items[idx]})
for idx in sorted(dry_set):
    curated.append({"idx": idx, "label": "DRY", **items[idx]})

with open(f"{EXP}/scratch/curated_all_final.json", "w", encoding="utf-8") as f:
    json.dump(curated, f, indent=1, ensure_ascii=False)
print("saved curated_all_final.json with", len(curated), "labeled images")
