"""
Filter staged_accepted.json to plausible racing/motorsport photos (drop PDFs,
maps, documents, unrelated scans by title keyword heuristics + file ext check),
then build contact sheets (grids of thumbnails with index labels) for visual
triage in batches of 24.
"""
import json, os, re
from PIL import Image, ImageDraw, ImageFont

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"
SHEETS = f"{EXP}/scratch/sheets"
os.makedirs(SHEETS, exist_ok=True)

with open(f"{EXP}/scratch/staged_accepted.json", encoding="utf-8") as f:
    items = json.load(f)

BAD_TITLE_PATTERNS = [
    r"\.pdf", r"\.djvu", r"\bmap\b", r"\bdocument\b", r"\bletter\b",
    r"\bconsultation\b", r"\breport\b", r"appendix", r"\bplan\b",
    r"newspaper", r"\bvol\.", r"gazette", r"census", r"survey",
    r"flood plain", r"minutes of", r"\bbook\b", r"catalogue",
    r"schedule", r"handbook", r"manual\b", r"bulletin",
]
BAD_RE = re.compile("|".join(BAD_TITLE_PATTERNS), re.IGNORECASE)

filtered = []
dropped = 0
for it in items:
    title = it["title"]
    fn = it["staged_filename"]
    path = os.path.join(STAGING, fn)
    if not os.path.exists(path):
        continue
    if BAD_RE.search(title):
        dropped += 1
        continue
    # verify it's actually a real photo-like image (not tiny/weird aspect after resize)
    try:
        img = Image.open(path)
        w, h = img.size
        if w < 300 or h < 300:
            dropped += 1
            continue
    except Exception:
        dropped += 1
        continue
    filtered.append(it)

print(f"kept {len(filtered)} / {len(items)} (dropped {dropped} by title/size heuristics)")

with open(f"{EXP}/scratch/staged_filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=1, ensure_ascii=False)

# ---- build contact sheets, 24 per sheet, 6x4 grid ----
COLS, ROWS = 6, 4
PER_SHEET = COLS * ROWS
THUMB = 220
LABEL_H = 22

try:
    font = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()

n_sheets = (len(filtered) + PER_SHEET - 1) // PER_SHEET
for s in range(n_sheets):
    batch = filtered[s*PER_SHEET:(s+1)*PER_SHEET]
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
        idx_global = s*PER_SHEET + i
        label = f"{idx_global}"
        draw.rectangle([c*THUMB, r*(THUMB+LABEL_H)+THUMB, (c+1)*THUMB, (r+1)*(THUMB+LABEL_H)], fill="black")
        draw.text((c*THUMB+4, r*(THUMB+LABEL_H)+THUMB+2), label, fill="white", font=font)
    out_path = f"{SHEETS}/sheet_{s:03d}.jpg"
    sheet.save(out_path, "JPEG", quality=80)

print(f"saved {n_sheets} contact sheets to {SHEETS}")
