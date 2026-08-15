"""
Final curation based on individual visual verification of contact sheets
(sheets3/ for the full 450-image set, sheets_rain/ for a closer look at the
2008 British GP rain cluster idx 60-149). Labels below reflect what was
ACTUALLY seen in each image, not title-based assumptions.
"""
import json

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"

with open(f"{EXP}/scratch/staged_index3.json", encoding="utf-8") as f:
    items = json.load(f)

def rng(lo, hi):
    return list(range(lo, hi+1))

# ---- DRY: clear/dry tarmac, real photo, usable (verified across sheets 0-17) ----
DRY_IDX = (
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

# ---- WET: 2008 British GP, verified visibly dark/glossy/saturated track with
# spray or clear standing-water sheen. Excludes non-track shots (pit-lane,
# grandstand-only, podium, airshow) even if in the numeric range. ----
WET_IDX = (
    rng(60, 79) +           # heavy spray, dark saturated track (verified)
    rng(90, 102) +          # dark wet track, visible reflections (verified)
    [104, 106] +
    rng(108, 121) +         # dark wet track continuing (verified)
    rng(124, 128) + [131, 134, 135, 136, 138, 139] +
    [140, 141, 143]
)

# ---- DAMP: verified matte-grey track, NO heavy spray/standing water, but
# visibly moist (not the bright dry-sun look of idx 0-49/243+ range).
# Conservative: only indices individually confirmed at full resolution. ----
DAMP_IDX = [80, 81, 82, 87, 88, 89]  # excludes 83/84 (bright/dry-looking on
# closer inspection), excludes 85/86 (off-track grass, not track surface)

def dedupe(lst):
    seen = set(); out = []
    for x in lst:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

DRY_IDX = dedupe(DRY_IDX)
WET_IDX = dedupe(WET_IDX)
DAMP_IDX = dedupe(DAMP_IDX)

damp_set = set(DAMP_IDX)
wet_set = set(WET_IDX) - damp_set
dry_set = set(DRY_IDX) - damp_set - wet_set

print(f"DAMP: {len(damp_set)}  WET: {len(wet_set)}  DRY: {len(dry_set)}")
print("total curated:", len(damp_set) + len(wet_set) + len(dry_set), "/ 450 staged")

curated = []
for idx in sorted(damp_set):
    curated.append({"idx": idx, "label": "DAMP", **items[idx]})
for idx in sorted(wet_set):
    curated.append({"idx": idx, "label": "WET", **items[idx]})
for idx in sorted(dry_set):
    curated.append({"idx": idx, "label": "DRY", **items[idx]})

with open(f"{EXP}/scratch/curated_round3_final.json", "w", encoding="utf-8") as f:
    json.dump(curated, f, indent=1, ensure_ascii=False)
print("saved curated_round3_final.json with", len(curated), "labeled images")
