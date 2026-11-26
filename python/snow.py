"""snow: CLIP の画像特徴から「路面の雪」と「まわりの積雪」の段階を出す（C++ 版 cpp/snow.hpp と同じ計算）。
  python snow.py feats   weights.bin labels.json datadir out.feats
  python snow.py train   in.feats head.json [lam epochs lr]
  python snow.py predict weights.bin head.json image... [--feat]
"""
import sys, os, json, struct, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clipvit

C = 6; XS = 10.0
# 学習の既定値。特徴から平均を引くので安定限界は lr≈1.3（C++ の snow.hpp と同じ説明）
LAM, EPOCHS, LR = 3e-4, 3000, 0.4
LEVELS = [("雪なし", 0), ("降り始め・うっすら", 0.5), ("〜5cm", 3), ("〜15cm", 10), ("〜30cm", 22), ("30cm超", 45)]

def normalize(f):
    f = np.asarray(f, np.float64); return f / np.sqrt((f * f).sum())

def probs(W, X):
    """W: (D+1, C)、X: (N, D) 平均を引いて XS 倍したもの"""
    z = X @ W[:-1] + W[-1]
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)

def train(X, y, lam=LAM, epochs=EPOCHS, lr=LR):
    """X は中心化・XS 倍済み。W=0 から全データの勾配で epochs 回。バイアスには正則化をかけない。"""
    N, D = X.shape; W = np.zeros((D + 1, C)); Y = np.eye(C)[y]; Xs = X
    for _ in range(epochs):
        G = probs(W, X) - Y
        g = np.vstack([Xs.T @ G, G.sum(0, keepdims=True)]) / N
        g[:-1] += lam * W[:-1]
        W -= lr * g
    return W

def load_feats(path):
    b = open(path, "rb").read(); assert b[:4] == b"SNWF"
    N, D = struct.unpack_from("<II", b, 4); o = 12
    F = np.frombuffer(b, np.float32, N * D, o).reshape(N, D); o += 4 * N * D
    road = np.frombuffer(b, np.int32, N, o); o += 4 * N
    around = np.frombuffer(b, np.int32, N, o)
    return F, road.astype(int), around.astype(int)

def save_feats(path, F, road, around):
    F = np.asarray(F, np.float32)
    with open(path, "wb") as f:
        f.write(b"SNWF" + struct.pack("<II", *F.shape)); f.write(F.tobytes())
        f.write(np.asarray(road, np.int32).tobytes()); f.write(np.asarray(around, np.int32).tobytes())

def load_model(path):
    j = json.load(open(path, encoding="utf-8")); D = j["dim"]
    return {"levels": [(l["label"], l["cm"]) for l in j["levels"]], "mean": np.array(j["mean"]),
            "road": np.array(j["road"]).reshape(D + 1, C), "around": np.array(j["around"]).reshape(D + 1, C)}

def fit(Xn, road, around, lam=LAM, epochs=EPOCHS, lr=LR):
    """Xn: 正規化済み特徴 (N, D)。平均を引いて路面とまわりの 2 つを学習する"""
    mean = Xn.mean(0); X = (Xn - mean) * XS
    return mean, train(X, road, lam, epochs, lr), train(X, around, lam, epochs, lr)

def save_model(path, mean, Wr, Wa, note):
    j = {"note": note, "dim": Wr.shape[0] - 1, "levels": [{"label": l, "cm": c} for l, c in LEVELS],
         "mean": [float(v) for v in mean],
         "road": [float(v) for v in Wr.ravel()], "around": [float(v) for v in Wa.ravel()]}
    json.dump(j, open(path, "w", encoding="utf-8"), ensure_ascii=False)

def predict(model, f):
    x = ((normalize(f) - model["mean"]) * XS)[None]; out = {}
    for k in ("road", "around"):
        p = probs(model[k], x)[0]
        out[k] = {"p": p.tolist(), "level": int(p.argmax()), "cm": float(p @ np.array([c for _, c in model["levels"]]))}
    return out

def _img(fn):
    from PIL import Image
    return np.asarray(Image.open(fn).convert("RGB"))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if a and a[0] == "feats" and len(a) == 5:
        w = clipvit.load_weights(a[1]); lab = json.load(open(a[2]))
        F = [clipvit.encode_image(w, clipvit.preprocess(_img(os.path.join(a[3], e["file"])), mode="pad")) for e in lab]
        save_feats(a[4], F, [e["road"] for e in lab], [e["around"] for e in lab]); print(len(F), "images ->", a[4])
    elif a and a[0] == "train" and len(a) in (3, 6):
        lam, ep, lr = (float(a[3]), int(a[4]), float(a[5])) if len(a) == 6 else (LAM, EPOCHS, LR)
        F, road, around = load_feats(a[1]); X = np.array([normalize(f) for f in F])
        note = "trained by Python on %d images, lam=%g epochs=%d lr=%g" % (len(F), lam, ep, lr)
        save_model(a[2], *fit(X, road, around, lam, ep, lr), note); print(note, "->", a[2])
    elif a and a[0] == "predict" and len(a) >= 4:
        w = clipvit.load_weights(a[1]); m = load_model(a[2])
        for fn in [x for x in a[3:] if not x.startswith("--")]:
            f = clipvit.encode_image(w, clipvit.preprocess(_img(fn), mode="pad"))
            if "--feat" in a: print(" ".join("%.6f" % v for v in f)); continue
            r = predict(m, f)
            print("%s\troad=%d %.1fcm (%s)\taround=%d %.1fcm" % (fn, r["road"]["level"], r["road"]["cm"],
                  " ".join("%.3f" % v for v in r["road"]["p"]), r["around"]["level"], r["around"]["cm"]))
    else:
        print(__doc__); sys.exit(1)
