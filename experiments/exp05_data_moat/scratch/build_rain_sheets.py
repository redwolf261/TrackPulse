import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"
SHEETS = f"{EXP}/scratch/sheets_rain"
os.makedirs(SHEETS, exist_ok=True)

with open(f"{EXP}/scratch/staged_index3.json", encoding="utf-8") as f:
    items = json.load(f)

LO, HI = 60, 149  # the full 2008 British GP rain-race cluster
subset = list(range(LO, HI+1))

COLS, ROWS = 4, 4
PER_SHEET = COLS * ROWS
THUMB = 260
LABEL_H = 22
try:
    font = ImageFont.truetype("arial.ttf", 15)
except Exception:
    font = ImageFont.load_default()

n_sheets = (len(subset) + PER_SHEET - 1) // PER_SHEET
for s in range(n_sheets):
    batch = subset[s*PER_SHEET:(s+1)*PER_SHEET]
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
    sheet.save(f"{SHEETS}/rainsheet_{s:03d}.jpg", "JPEG", quality=82)

print(f"saved {n_sheets} sheets covering idx {LO}-{HI}")
