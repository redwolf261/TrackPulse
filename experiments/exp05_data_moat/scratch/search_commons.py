"""
exp05 sourcing step 1: search Wikimedia Commons for candidate images across many
queries/categories, dedupe by pageid, fetch imageinfo (url, license, artist,
extmetadata), and save a big candidate list to disk for later triage/download.
No downloading here -- just metadata harvesting, to be polite with API usage.
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
            print(f"  retry {attempt} for {params.get('srsearch') or params.get('titles')}: {e}", file=sys.stderr)
            time.sleep(2)
    return {}

def search_images(query, limit=100):
    """Full-text search in File namespace, paginated."""
    results = []
    sroffset = 0
    while len(results) < limit:
        data = api_get({
            "action": "query", "list": "search", "srnamespace": 6,
            "srsearch": query, "srlimit": min(50, limit - len(results)),
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

def category_members(category, limit=200):
    """List files in a category."""
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
    """Batch fetch imageinfo (url + extmetadata) for a list of titles, 50 at a time."""
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i+50]
        data = api_get({
            "action": "query", "titles": "|".join(batch), "prop": "imageinfo",
            "iiprop": "url|size|extmetadata", "iiurlwidth": 1024,
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

# ---- query list ----
DAMP_QUERIES = [
    "damp track motorsport", "drying track F1", "track evolution rain",
    "light rain Formula 1 track", "wet drying circuit", "intermediate tyres track",
    "track drying after rain racing", "post rain circuit racing",
    "damp asphalt circuit", "greasy track motorsport", "moist track racing",
    "drying line racing circuit", "rain affected track drying",
]
DAMP_CATEGORIES = [
    "Formula One cars in the rain",
    "2021 Belgian Grand Prix",
    "2008 British Grand Prix",
    "2007 European Grand Prix",
    "2000 European Grand Prix",
    "Rain in motorsport",
]
DRY_QUERIES = [
    "Formula 1 pit lane dry", "onboard camera Formula 1 track",
    "touring car racing dry track", "endurance racing dry circuit",
    "Le Mans dry track", "DTM dry track racing", "IndyCar dry track",
    "Formula E dry track", "historic Grand Prix dry track 1990s",
    "Formula 3 dry track racing", "GT racing dry circuit",
]
DRY_CATEGORIES = [
    "24 Hours of Le Mans",
    "DTM (touring car racing)",
    "IndyCar Series",
    "Formula E",
    "Nurburgring",
    "Circuit de Spa-Francorchamps",
]
WET_QUERIES = [
    "wet Formula 1 track racing", "heavy rain motorsport race",
    "spray wet track racing car", "rain race track flooded",
    "wet touring car race", "monsoon race track",
]

def run():
    all_candidates = {}  # title -> {source, queries}
    def add(title, src):
        if title not in all_candidates:
            all_candidates[title] = {"queries": []}
        all_candidates[title]["queries"].append(src)

    for q in DAMP_QUERIES:
        print("search:", q)
        for h in search_images(q, limit=60):
            add(h["title"], f"search:{q}")
    for c in DAMP_CATEGORIES:
        print("category:", c)
        for h in category_members(c, limit=150):
            add(h["title"], f"cat:{c}")
    for q in DRY_QUERIES:
        print("search:", q)
        for h in search_images(q, limit=40):
            add(h["title"], f"search:{q}")
    for c in DRY_CATEGORIES:
        print("category:", c)
        for h in category_members(c, limit=100):
            add(h["title"], f"cat:{c}")
    for q in WET_QUERIES:
        print("search:", q)
        for h in search_images(q, limit=40):
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
        merged.append({
            "title": title,
            "queries": meta["queries"],
            "url": info.get("url"),
            "thumburl": info.get("thumburl"),
            "width": info.get("width"), "height": info.get("height"),
            "extmetadata": info.get("extmetadata", {}),
        })

    with open("C:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp05_data_moat/scratch/candidates_raw.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=1, ensure_ascii=False)
    print(f"saved {len(merged)} candidates with imageinfo")

if __name__ == "__main__":
    run()
