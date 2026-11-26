# C++ と Python の突き合わせ。
#  1) 同じ特徴ファイルから学習した層の重み（head_cpp.json と head_py.json）
#  2) 同じ画素（PIL で復号して PNG にしたもの）からの特徴と判定
# usage: python parity.py snow_cli.exe weights.bin head_cpp.json head_py.json image...
import sys, os, json, subprocess, tempfile, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import clipvit, snow
cli, wpath, hc, hp = sys.argv[1:5]; imgs = sys.argv[5:]
mc, mp = snow.load_model(hc), snow.load_model(hp)
dw = max(np.abs(mc[k] - mp[k]).max() for k in ("road", "around"))
scale = max(np.abs(mc[k]).max() for k in ("road", "around"))
print("head weights: max|C++ - Py| = %.3e  (max|W| = %.3f)" % (dw, scale))
w = clipvit.load_weights(wpath)
tmp = tempfile.mkdtemp(); pngs = []
for i, fn in enumerate(imgs):
    p = os.path.join(tmp, "%03d.png" % i); Image.open(fn).convert("RGB").save(p); pngs.append(p)
out = subprocess.run([cli, "predict", wpath, hc, *pngs, "--feat"], capture_output=True, text=True, check=True).stdout.split("\n")
worst_cos, worst_p = 1.0, 0.0
for p, line in zip(pngs, out):
    fc = np.array([float(v) for v in line.split()])
    fp = clipvit.encode_image(w, clipvit.preprocess(np.asarray(Image.open(p).convert("RGB")), mode="pad"))
    c = float(fc @ fp / np.linalg.norm(fc) / np.linalg.norm(fp))
    rc, rp = snow.predict(mc, fc), snow.predict(mp, fp)
    dp = max(np.abs(np.array(rc[k]["p"]) - np.array(rp[k]["p"])).max() for k in ("road", "around"))
    worst_cos = min(worst_cos, c); worst_p = max(worst_p, dp)
    print("%s cos=%.8f  max|dp|=%.2e  road=%d/%d" % (os.path.basename(p), c, dp, rc["road"]["level"], rp["road"]["level"]))
ok = dw < 1e-8 * max(1, scale) and worst_cos > 0.99999 and worst_p < 1e-3
print("WORST head=%.2e cos=%.8f dp=%.2e -> %s" % (dw, worst_cos, worst_p, "PASS" if ok else "FAIL"))
