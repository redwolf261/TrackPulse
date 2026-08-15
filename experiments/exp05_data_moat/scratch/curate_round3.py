"""
Manual visual-triage curation of staged_index3.json (450 candidates from round
2+3 sourcing), based on contact-sheet review. Produces a curated list of
(index, label, note) for images judged genuinely usable for DRY/DAMP/WET
racing-domain training/eval, dropping museum/pit-lane/non-track/document noise.

Labeling notes from visual review:
- DRY: clear dry tarmac, no visible moisture/sheen, any era/series - Spa 2019
  event photos (325-374 range minus non-track), Japan/Suzuka museum-adjacent
  on-track shots, historic F1/vintage racing on visibly dry track, DTM/other
  touring car dry track action.
- WET: 2008 British GP (heavy rain race) - visible spray, saturated dark
  glossy track, standing water - indices in the 50-149 cluster showing spray/
  heavy wet.
- DAMP: within the same 2008 British GP set, several frames show a track
  surface that is visibly moist/grey/matte but WITHOUT heavy spray or
  standing-water glare - later-race drying phase. These are the highest-value
  finds. Also a few "drying track" titled images from round-2 search.
"""
import json, os

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"

with open(f"{EXP}/scratch/staged_index3.json", encoding="utf-8") as f:
    items = json.load(f)  # index i corresponds to items[i], sorted by filename

def by_range(lo, hi):
    return list(range(lo, hi+1))

# ---- DRY selections (clear dry track visible, real photo, not museum-only) ----
DRY_IDX = (
    by_range(0, 4) + by_range(5, 6) + [9, 10] +
    by_range(20, 22) + [24] +
    by_range(40, 47) + [49] +
    by_range(243, 249) + by_range(250, 254) + by_range(260, 264) +
    by_range(266, 269) + by_range(270, 272) +
    by_range(275, 280) + [283] +
    by_range(285, 286) + [288, 289] + [293, 294] + [295] + [297, 298, 299] +
    [300, 302] +
    by_range(325, 327) + [332, 333] + by_range(337, 338) + by_range(342, 353) +
    by_range(364, 372) +
    by_range(375, 383) + [385, 389, 390, 392, 393, 394] +
    [396, 402, 404, 405, 407, 408, 411, 413, 414, 416] +
    [420, 422, 423, 424, 425, 426, 428, 429, 430, 435, 437, 444, 445, 449]
)

# ---- WET selections (2008 British GP heavy rain, visible spray/saturation) ----
WET_IDX = (
    [79] +
    by_range(90, 99) +
    by_range(105, 121) +
    by_range(125, 149)
)

# ---- DAMP selections: within the wet-race cluster, frames with visibly moist
# but not heavily-sprayed/saturated track (drying-phase). Conservative subset -
# reviewed for grey/matte (not glossy-dark) tarmac, no heavy spray trail. ----
DAMP_IDX = (
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89] +
    [100, 101, 102, 103, 104]
)

def dedupe_keep_order(lst):
    seen = set(); out = []
    for x in lst:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

DRY_IDX = dedupe_keep_order(DRY_IDX)
WET_IDX = dedupe_keep_order(WET_IDX)
DAMP_IDX = dedupe_keep_order(DAMP_IDX)

# resolve conflicts: an index can only be in ONE class; priority DAMP > WET > DRY
# (DAMP is highest value / most carefully selected)
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

with open(f"{EXP}/scratch/curated_round3.json", "w", encoding="utf-8") as f:
    json.dump(curated, f, indent=1, ensure_ascii=False)
print("saved curated_round3.json with", len(curated), "labeled images")
