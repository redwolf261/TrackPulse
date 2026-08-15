"""
Download candidates (thumburl, 1024px) from candidates_raw.json, resize/reencode
to JPEG q85 max-dim 1024, hash, skip anything colliding with existing_hash_pool.json
or already-downloaded-this-round hashes. Filters for CC-style licenses only.
Saves to staging/ with a metadata sidecar per accepted image.
"""
import json, os, hashlib, urllib.request, time, sys
from io import BytesIO
from PIL import Image

ROOT = "C:/Users/Rivan/Projects/AI_Grand_Prix"
EXP = f"{ROOT}/experiments/exp05_data_moat"
STAGING = f"{EXP}/staging"
os.makedirs(STAGING, exist_ok=True)
UA = "TrackPulseHackathonBot/1.0 (research use; contact: researchx03@gmail.com)"

with open(f"{EXP}/scratch/candidates_raw.json", encoding="utf-8") as f:
    candidates = json.load(f)
with open(f"{EXP}/scratch/existing_hash_pool.json", encoding="utf-8") as f:
    existing_hashes = set(json.load(f))

ACCEPTABLE_LICENSE_SUBSTR = [
    "cc-by", "cc0", "public domain", "pd-", "cc-zero", "attribution"
]
REJECT_LICENSE_SUBSTR = ["non-commercial", "nc-", "nd-", "copyrighted", "fair use", "all rights reserved"]

def license_ok(extmeta):
    lic = (extmeta.get("LicenseShortName", {}).get("value") or "").lower()
    licurl = (extmeta.get("LicenseUrl", {}).get("value") or "").lower()
    combined = lic + " " + licurl
    if any(b in combined for b in REJECT_LICENSE_SUBSTR):
        return False, lic
    if any(a in combined for a in ACCEPTABLE_LICENSE_SUBSTR):
        return True, lic
    # unknown/missing license metadata -> reject to be safe
    return False, lic

seen_this_round = set()
accepted = []
rejected_license = 0
rejected_dup = 0
rejected_download_fail = 0
rejected_too_small = 0

def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

MAX_DIM = 1024
QUALITY = 85

idx = 0
for i, c in enumerate(candidates):
    extmeta = c.get("extmetadata", {})
    ok, lic = license_ok(extmeta)
    if not ok:
        rejected_license += 1
        continue
    url = c.get("thumburl") or c.get("url")
    if not url:
        continue
    try:
        raw = download(url)
    except Exception as e:
        rejected_download_fail += 1
        continue
    h = sha256_bytes(raw)
    if h in existing_hashes or h in seen_this_round:
        rejected_dup += 1
        continue
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        rejected_download_fail += 1
        continue
    w, hh = img.size
    if min(w, hh) < 200:
        rejected_too_small += 1
        continue
    # resize
    scale = MAX_DIM / max(w, hh)
    if scale < 1.0:
        img = img.resize((max(1,int(w*scale)), max(1,int(hh*scale))), Image.LANCZOS)
    out_name = f"cand_{idx:04d}.jpg"
    out_path = os.path.join(STAGING, out_name)
    img.save(out_path, "JPEG", quality=QUALITY)
    # re-hash the SAVED (resized) file too, for the training-pool-stage hash check
    with open(out_path, "rb") as f:
        saved_hash = sha256_bytes(f.read())
    seen_this_round.add(h)
    accepted.append({
        "staged_filename": out_name,
        "title": c["title"],
        "queries": c.get("queries", []),
        "source_url": c.get("url"),
        "license_short": extmeta.get("LicenseShortName", {}).get("value"),
        "artist": extmeta.get("Artist", {}).get("value"),
        "description": (extmeta.get("ImageDescription", {}).get("value") or "")[:300],
        "orig_sha256": h,
        "saved_sha256": saved_hash,
        "orig_w": w, "orig_h": hh,
    })
    idx += 1
    if idx % 25 == 0:
        print(f"accepted {idx} / scanned {i+1}/{len(candidates)}")
    time.sleep(0.05)

print(f"\nDONE. scanned={len(candidates)} accepted={len(accepted)} "
      f"rejected_license={rejected_license} rejected_dup={rejected_dup} "
      f"rejected_download_fail={rejected_download_fail} rejected_too_small={rejected_too_small}")

with open(f"{EXP}/scratch/staged_accepted.json", "w", encoding="utf-8") as f:
    json.dump(accepted, f, indent=1, ensure_ascii=False)
print("saved staged_accepted.json")
