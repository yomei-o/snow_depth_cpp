# Wikimedia Commons のカテゴリから画像を集める（検証用の小さなセット）。
# 使い方: python fetch_commons.py OUTDIR label=Category:Name [label=Category:Name ...]
import sys, os, json, time, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA = {"User-Agent": "snow-sanity-fetch/0.1 (research script)"}
API = "https://commons.wikimedia.org/w/api.php"
def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for wait in (0, 30, 60, 120, 240):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception as e:
            err = e
    raise err
out = sys.argv[1]; per = 40
meta = []
for spec in sys.argv[2:]:
    label, cat = spec.split("=", 1)
    os.makedirs(os.path.join(out, label), exist_ok=True)
    d = get({"action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": cat,
             "gcmtype": "file", "gcmlimit": 200, "prop": "imageinfo", "iiprop": "url|extmetadata|mime",
             "iiurlwidth": 500})
    pages = list(d.get("query", {}).get("pages", {}).values())
    n = 0
    for p in pages:
        ii = p.get("imageinfo", [{}])[0]
        if ii.get("mime") not in ("image/jpeg", "image/png"): continue
        lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
        fn = os.path.join(out, label, "%03d.jpg" % n)
        ok = os.path.exists(fn) and os.path.getsize(fn) > 0
        for wait in (() if ok else (0, 10, 30, 60)):
            time.sleep(wait)
            try:
                with urllib.request.urlopen(urllib.request.Request(ii["thumburl"], headers=UA), timeout=30) as r:
                    open(fn, "wb").write(r.read())
                ok = True; break
            except Exception as e:
                err = e
        if not ok:
            print("skip", p["title"], err); continue
        meta.append({"file": fn, "label": label, "title": p["title"], "page": ii.get("descriptionurl"), "license": lic})
        n += 1; time.sleep(1.5)
        if n >= per: break
    print(label, cat, len(pages), "->", n, flush=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
