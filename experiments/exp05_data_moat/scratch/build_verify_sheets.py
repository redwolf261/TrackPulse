import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"
SHEETS = f"{EXP}/scratch/sheets_verify"

with open(f"{EXP}/scratch/staged_index4.json", encoding="utf-8") as f:
    items = json.load(f)

RANGES = [(1161, 1249)]
subset_idx = []
for lo, hi in RANGES:
    subset_idx.extend(range(lo, hi+1))

COLS, ROWS = 5, 5
PER_SHEET = COLS * ROWS
THUMB = 200
LABEL_H = 20
try:
    font = ImageFont.truetype("arial.ttf", 13)
except Exception:
    font = ImageFont.load_default()

n_sheets = (len(subset_idx) + PER_SHEET - 1) // PER_SHEET
for s in range(n_sheets):
    batch = subset_idx[s*PER_SHEET:(s+1)*PER_SHEET]
    sheet = Image.new("RGB", (COLS*THUMB, ROWS*(THUMB+LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, idx in enumerate(batch):
        r, c = divmod(i, COLS)
        it = items[idx]
        path = os.path.join(STAGING, it["staged_filename"])
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((THUMB, THUMB))
        except Exception:
            continue
        x0 = c*THUMB + (THUMB - img.width)//2
        y0 = r*(THUMB+LABEL_H) + (THUMB - img.height)//2
        sheet.paste(img, (x0, y0))
        draw.rectangle([c*THUMB, r*(THUMB+LABEL_H)+THUMB, (c+1)*THUMB, (r+1)*(THUMB+LABEL_H)], fill="black")
        draw.text((c*THUMB+4, r*(THUMB+LABEL_H)+THUMB+2), str(idx), fill="white", font=font)
    sheet.save(f"{SHEETS}/verify_{s:03d}.jpg", "JPEG", quality=80)
print(f"saved {n_sheets} verify sheets for idx range(s) {RANGES}")
