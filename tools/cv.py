# 5 分割の交差検証で精度を測る（python/snow.py の学習をそのまま使う）。
# usage: python cv.py in.feats
import sys, os, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import snow
F, road, around = snow.load_feats(sys.argv[1]); Xn = np.array([snow.normalize(f) for f in F]); N = len(F)
idx = np.random.RandomState(0).permutation(N)
P = {"road": np.zeros((N, 6)), "around": np.zeros((N, 6))}
for k in range(5):
    te = idx[k::5]; tr = np.setdiff1d(idx, te)
    mean, Wr, Wa = snow.fit(Xn[tr], road[tr], around[tr])
    X = (Xn[te] - mean) * snow.XS
    P["road"][te] = snow.probs(Wr, X); P["around"][te] = snow.probs(Wa, X)
for key, y in (("road", road), ("around", around)):
    r = P[key].argmax(1); e = np.isin(y, [0, 1, 2])
    print("%-6s exact=%.2f  within1=%.2f  0-2の一致=%.2f" % (key, (r == y).mean(), (abs(r - y) <= 1).mean(), (r[e] == y[e]).mean()))
    cm = np.zeros((6, 6), int)
    for a, b in zip(y, r): cm[a, b] += 1
    print("  行=目視ラベル 列=判定\n" + "\n".join("  " + " ".join("%3d" % v for v in row) for row in cm))
