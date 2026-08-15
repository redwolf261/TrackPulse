import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"
SHEETS = f"{EXP}/scratch/sheets4"
os.makedirs(SHEETS, exist_ok=True)

with open(f"{EXP}/scratch/staged_accepted4.json", encoding="utf-8") as f:
    items = json.load(f)
items.sort(key=lambda x: x["staged_filename"])
with open(f"{EXP}/scratch/staged_index4.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=1, ensure_ascii=False)

# only build sheets for the NEW round (idx 450 onward, since 0-449 already done)
START = 450
subset = items[START:]

COLS, ROWS = 5, 5
PER_SHEET = COLS * ROWS
THUMB = 200
LABEL_H = 20
try:
    font = ImageFont.truetype("arial.ttf", 13)
except Exception:
    font = ImageFont.load_default()

n_sheets = (len(subset) + PER_SHEET - 1) // PER_SHEET
for s in range(n_sheets):
    batch = subset[s*PER_SHEET:(s+1)*PER_SHEET]
    sheet = Image.new("RGB", (COLS*THUMB, ROWS*(THUMB+LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, it in enumerate(batch):
        r, c = divmod(i, COLS)
        path = os.path.join(STAGING, it["staged_filename"])
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((THUMB, THUMB))
        except Exception:
            continue
        x0 = c*THUMB + (THUMB - img.width)//2
        y0 = r*(THUMB+LABEL_H) + (THUMB - img.height)//2
        sheet.paste(img, (x0, y0))
        idx_global = START + s*PER_SHEET + i
        draw.rectangle([c*THUMB, r*(THUMB+LABEL_H)+THUMB, (c+1)*THUMB, (r+1)*(THUMB+LABEL_H)], fill="black")
        draw.text((c*THUMB+4, r*(THUMB+LABEL_H)+THUMB+2), str(idx_global), fill="white", font=font)
    sheet.save(f"{SHEETS}/sheet_{s:03d}.jpg", "JPEG", quality=78)

print(f"saved {n_sheets} sheets ({len(subset)} images, idx {START}-{START+len(subset)-1}) to {SHEETS}")
print("total staged overall:", len(items))
