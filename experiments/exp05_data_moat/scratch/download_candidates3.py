"""
v3: same as v2 but with retry+backoff on download failures (likely rate-limit
related), and continues indexing after the existing staged files (doesn't wipe).
"""
import json, os, hashlib, urllib.request, time, re, glob
from io import BytesIO
from PIL import Image

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"
os.makedirs(STAGING, exist_ok=True)
UA = "TrackPulseHackathonBot/1.0 (research use; contact: researchx03@gmail.com)"

with open(f"{EXP}/scratch/candidates_raw2.json", encoding="utf-8") as f:
    candidates = json.load(f)
with open(f"{EXP}/scratch/existing_hash_pool.json", encoding="utf-8") as f:
    existing_hashes = set(json.load(f))
with open(f"{EXP}/scratch/staged_accepted2.json", encoding="utf-8") as f:
    already = json.load(f)

already_titles = {a["title"] for a in already}
already_hashes = {a["orig_sha256"] for a in already}
print(f"already have {len(already)} accepted from prior run; skipping their titles")

BAD_TITLE_PATTERNS = [
    r"\bmicroform\b", r"\breview\b(?!.*race)", r"\bstudy\b", r"commonwealth",
    r"\bmanual\b", r"hurricane", r"saturn", r"reservoir", r"\bmuseum\b(?!.*track)",
    r"citroen 7 cv", r"lotus europa\b(?! .*track)", r"\bbook\b", r"catalogue",
    r"newspaper", r"gazette", r"\bmap\b", r"floor ?plan", r"blueprint",
]
BAD_RE = re.compile("|".join(BAD_TITLE_PATTERNS), re.IGNORECASE)
ACCEPTABLE_LICENSE_SUBSTR = ["cc-by", "cc by", "cc0", "cc 0", "public domain", "pd-", "cc-zero", "attribution", "no restrictions"]
REJECT_LICENSE_SUBSTR = ["non-commercial", "nc-", "nd-", "copyrighted", "fair use", "all rights reserved"]

def license_ok(extmeta):
    lic = (extmeta.get("LicenseShortName", {}).get("value") or "").lower()
    licurl = (extmeta.get("LicenseUrl", {}).get("value") or "").lower()
    combined = lic + " " + licurl
    if any(b in combined for b in REJECT_LICENSE_SUBSTR):
        return False
    if any(a in combined for a in ACCEPTABLE_LICENSE_SUBSTR):
        return True
    return False

def download(url, retries=5):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            wait = 1.5 * (attempt + 1)
            time.sleep(wait)
    return None

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

MAX_DIM = 1024
QUALITY = 85
seen_this_round = set(already_hashes)
accepted = list(already)
rejected = {"title": 0, "license": 0, "dup": 0, "download_fail": 0, "too_small": 0, "already": 0}

existing_files = glob.glob(os.path.join(STAGING, "cand_*.jpg"))
idx = len(existing_files)  # continue numbering

for i, c in enumerate(candidates):
    title = c["title"]
    if title in already_titles:
        rejected["already"] += 1
        continue
    if BAD_RE.search(title):
        rejected["title"] += 1
        continue
    extmeta = c.get("extmetadata", {})
    if not license_ok(extmeta):
        rejected["license"] += 1
        continue
    url = c.get("thumburl") or c.get("url")
    if not url:
        continue
    raw = download(url)
    if raw is None:
        rejected["download_fail"] += 1
        continue
    h = sha256_bytes(raw)
    if h in existing_hashes or h in seen_this_round:
        rejected["dup"] += 1
        continue
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        rejected["download_fail"] += 1
        continue
    w, hh = img.size
    if min(w, hh) < 250:
        rejected["too_small"] += 1
        continue
    scale = MAX_DIM / max(w, hh)
    if scale < 1.0:
        img = img.resize((max(1,int(w*scale)), max(1,int(hh*scale))), Image.LANCZOS)
    out_name = f"cand_{idx:04d}.jpg"
    out_path = os.path.join(STAGING, out_name)
    img.save(out_path, "JPEG", quality=QUALITY)
    with open(out_path, "rb") as f:
        saved_hash = sha256_bytes(f.read())
    seen_this_round.add(h)
    accepted.append({
        "staged_filename": out_name, "title": title, "queries": c.get("queries", []),
        "source_url": c.get("url"), "license_short": extmeta.get("LicenseShortName", {}).get("value"),
        "artist": extmeta.get("Artist", {}).get("value"),
        "description": (extmeta.get("ImageDescription", {}).get("value") or "")[:300],
        "orig_sha256": h, "saved_sha256": saved_hash, "orig_w": w, "orig_h": hh,
    })
    idx += 1
    if idx % 20 == 0:
        print(f"accepted total {len(accepted)} (idx={idx}) / scanned {i+1}/{len(candidates)}", flush=True)
    time.sleep(0.3)

print(f"\nDONE. scanned={len(candidates)} total_accepted={len(accepted)} rejected={rejected}", flush=True)
with open(f"{EXP}/scratch/staged_accepted3.json", "w", encoding="utf-8") as f:
    json.dump(accepted, f, indent=1, ensure_ascii=False)
print("saved staged_accepted3.json", flush=True)
