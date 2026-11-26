# 確認用の一覧画像を作る: python sheet.py meta.json... outprefix
import sys, json, os
from PIL import Image, ImageDraw
metas = sys.argv[1:-1]; pre = sys.argv[-1]
items = []
for m in metas:
    base = os.path.dirname(os.path.dirname(m))   # data/
    for e in json.load(open(m, encoding="utf-8")):
        rel = e["file"].replace("\\", "/").split("/", 1)[1]
        items.append(os.path.join(base, rel))
W, R, th = 6, 4, 210
for s in range(0, len(items), W * R):
    chunk = items[s:s + W * R]
    sheet = Image.new("RGB", (W * th, R * (th + 16)), "white"); d = ImageDraw.Draw(sheet)
    for i, f in enumerate(chunk):
        im = Image.open(f).convert("RGB"); im.thumbnail((th - 4, th - 4))
        x = (i % W) * th; y = (i // W) * (th + 16)
        sheet.paste(im, (x, y + 16)); d.text((x + 2, y + 2), "%d %s" % (s + i, "/".join(f.replace("\\", "/").split("/")[-2:])), fill="black")
    sheet.save("%s_%02d.jpg" % (pre, s // (W * R)), quality=80)
json.dump(items, open(pre + "_index.json", "w"), indent=0)
print(len(items))
