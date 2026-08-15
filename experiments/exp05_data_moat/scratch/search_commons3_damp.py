"""
exp05 sourcing round 3: DAMP-focused pass. Rain-affected race categories tend to
contain a mix of WET (heavy spray) and DAMP (drying/transitional) shots within
the same category/event, since races often start wet and dry out. Pull broad
category listings from known rain-affected races across eras/series, plus
targeted queries, for later visual triage into DAMP vs WET vs DRY.
"""
import json, time, urllib.parse, urllib.request, sys

API = "https://commons.wikimedia.org/w/api.php"
UA = "TrackPulseHackathonBot/1.0 (research use; contact: researchx03@gmail.com)"

def api_get(params):
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"  retry {attempt}: {e}", file=sys.stderr)
            time.sleep(2)
    return {}

def search_images(query, limit=100):
    results = []
    sroffset = 0
    q = f"{query} filetype:bitmap"
    while len(results) < limit:
        data = api_get({
            "action": "query", "list": "search", "srnamespace": 6,
            "srsearch": q, "srlimit": min(50, limit - len(results)),
            "sroffset": sroffset,
        })
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        results.extend(hits)
        cont = data.get("continue", {}).get("sroffset")
        if cont is None:
            break
        sroffset = cont
        time.sleep(0.2)
    return results

def category_members(category, limit=300):
    results = []
    cmcontinue = None
    while len(results) < limit:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": "file",
            "cmlimit": min(500, limit - len(results)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        hits = data.get("query", {}).get("categorymembers", [])
        if not hits:
            break
        results.extend(hits)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.2)
    return results

def get_imageinfo(titles):
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i+50]
        data = api_get({
            "action": "query", "titles": "|".join(batch), "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime", "iiurlwidth": 1024,
        })
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            title = page.get("title")
            infos = page.get("imageinfo")
            if not infos:
                continue
            out[title] = infos[0]
        time.sleep(0.3)
    return out

# rain-affected races across many eras/series -- likely to contain both WET
# (start/peak rain) and DAMP (drying-out) frames within the same event category
RAIN_EVENT_CATEGORIES = [
    "2021 Belgian Grand Prix",
    "2008 British Grand Prix",
    "2007 European Grand Prix",
    "2000 European Grand Prix",
    "2021 Turkish Grand Prix",
    "2019 German Grand Prix",
    "2016 Monaco Grand Prix",
    "2012 Malaysian Grand Prix",
    "2010 Malaysian Grand Prix",
    "2011 Canadian Grand Prix",
    "2014 Japanese Grand Prix",
    "1998 Belgian Grand Prix",
    "1999 European Grand Prix",
    "2019 Japanese Grand Prix",
    "2021 Russian Grand Prix",
    "2023 Dutch Grand Prix",
    "2008 Japanese Grand Prix",
]
DAMP_QUERIES2 = [
    "track drying out rain race", "damp tarmac racing car",
    "drying racing line wet track", "rain stopped track still wet racing",
    "light drizzle race track car", "misty wet track racing car",
    "wet weather tyres drying track", "safety car rain drying track",
]

def run():
    all_candidates = {}
    def add(title, src):
        if title not in all_candidates:
            all_candidates[title] = {"queries": []}
        all_candidates[title]["queries"].append(src)

    for c in RAIN_EVENT_CATEGORIES:
        print("category:", c)
        for h in category_members(c, limit=250):
            add(h["title"], f"cat:{c}")
    for q in DAMP_QUERIES2:
        print("search:", q)
        for h in search_images(q, limit=60):
            add(h["title"], f"search:{q}")

    print(f"total unique candidate titles: {len(all_candidates)}")
    titles = list(all_candidates.keys())
    imageinfo = get_imageinfo(titles)
    print(f"got imageinfo for {len(imageinfo)} titles")

    merged = []
    for title, meta in all_candidates.items():
        info = imageinfo.get(title)
        if not info:
            continue
        mime = info.get("mime", "")
        if not mime.startswith("image/") or mime in ("image/vnd.djvu",):
            continue
        merged.append({
            "title": title, "queries": meta["queries"], "url": info.get("url"),
            "thumburl": info.get("thumburl"), "width": info.get("width"),
            "height": info.get("height"), "mime": mime,
            "extmetadata": info.get("extmetadata", {}),
        })

    with open("C:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp05_data_moat/scratch/candidates_raw3_damp.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=1, ensure_ascii=False)
    print(f"saved {len(merged)} candidates with imageinfo (mime-filtered)")

if __name__ == "__main__":
    run()
