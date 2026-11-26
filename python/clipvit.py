"""clipvit: CLIP ViT-B/32 の画像側を numpy だけで動かす参照実装（C++ 版 cpp/clipvit.hpp と同じ計算）。
oshare_cpp の oshare.py から取り出した共通部分。"""
import struct, json, numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

def load_weights(path):
    b = open(path, "rb").read()
    assert b[:4] == b"OSHW"
    _, n = struct.unpack_from("<II", b, 4); o = 12; w = {}
    for _ in range(n):
        (L,) = struct.unpack_from("<H", b, o); o += 2
        k = b[o:o + L].decode(); o += L
        t, nd = struct.unpack_from("<BB", b, o); o += 2
        shp = struct.unpack_from("<%dI" % nd, b, o); o += 4 * nd
        cnt = int(np.prod(shp))
        if t == 0:
            a = np.frombuffer(b, np.float32, cnt, o).reshape(shp); o += 4 * cnt
        else:
            s = np.frombuffer(b, np.float32, shp[0], o); o += 4 * shp[0]
            q = np.frombuffer(b, np.int8, cnt, o).reshape(shp); o += cnt
            a = (q.astype(np.float32) * s[:, None]).astype(np.float32)
        w[k] = a
    return w

# ---- PIL 互換の bicubic 縮小（Image.resize(BICUBIC) と同じ係数・同じ丸め） ----
def _cubic(x, a=-0.5):
    x = abs(x)
    if x < 1: return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2: return (((x - 5) * x + 8) * x - 4) * a
    return 0.0

def _coeffs(insize, outsize):
    scale = insize / outsize
    fs = max(scale, 1.0); support = 2.0 * fs
    res = []
    for i in range(outsize):
        center = (i + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), insize)
        ws = [_cubic((x - center + 0.5) / fs) for x in range(xmin, xmax)]
        tot = sum(ws)
        res.append((xmin, [w / tot for w in ws] if tot != 0 else ws))
    return res

def _resample_axis(img, outsize, axis):
    # img: uint8 (H,W,C)。PIL は 8bit で各パスごとに丸め・クリップする（係数は 22bit 固定小数）
    insize = img.shape[axis]
    PREC = 22
    out_shape = list(img.shape); out_shape[axis] = outsize
    out = np.empty(out_shape, np.uint8)
    src = img.astype(np.int64)
    for i, (xmin, ws) in enumerate(_coeffs(insize, outsize)):
        k = np.array([int(w * (1 << PREC) + (0.5 if w >= 0 else -0.5)) for w in ws], np.int64)
        seg = np.take(src, range(xmin, xmin + len(ws)), axis=axis)
        sh = [1] * img.ndim; sh[axis] = len(ws)
        acc = (seg * k.reshape(sh)).sum(axis=axis) + (1 << (PREC - 1))
        v = np.clip(acc >> PREC, 0, 255).astype(np.uint8)
        idx = [slice(None)] * img.ndim; idx[axis] = i
        out[tuple(idx)] = v
    return out

def pil_bicubic(img, w, h):
    # PIL は横→縦の順（横が変わらない場合は省略）
    if img.shape[1] != w: img = _resample_axis(img, w, 1)
    if img.shape[0] != h: img = _resample_axis(img, h, 0)
    return img

def pad_square(rgb):
    """長辺に合わせて正方形にする。余白は CLIP の平均色（uint8 に丸めた値）"""
    H, W = rgb.shape[:2]; n = max(H, W)
    out = np.empty((n, n, 3), np.uint8); out[:] = np.round(MEAN * 255).astype(np.uint8)
    y = (n - H) // 2; x = (n - W) // 2
    out[y:y + H, x:x + W] = rgb
    return out

def preprocess(rgb, size=224, mode="crop"):
    """rgb: uint8 (H,W,3)。mode=pad なら先に正方形へ余白を足す。
    短辺を size に bicubic、中央切り抜き、正規化 → (3,size,size) float32"""
    if mode == "pad": rgb = pad_square(rgb)
    H, W = rgb.shape[:2]
    if W <= H: nw, nh = size, int(size * H / W)
    else: nh, nw = size, int(size * W / H)
    r = pil_bicubic(rgb, nw, nh)
    top = int(round((nh - size) / 2.0)); left = int(round((nw - size) / 2.0))
    r = r[top:top + size, left:left + size]
    x = (r.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1).copy()

def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

def encode_image(w, x):
    """x: (3,224,224) → 正規化前の 512 次元特徴"""
    P = 32; G = 224 // P; D = 768; Hh = 12
    cw = w["conv1.weight"].reshape(D, -1)                      # (768, 3*32*32)
    patches = x.reshape(3, G, P, G, P).transpose(1, 3, 0, 2, 4).reshape(G * G, -1)
    t = patches @ cw.T
    t = np.concatenate([w["class_embedding"][None], t], 0) + w["positional_embedding"]
    t = _ln(t, w["ln_pre.weight"], w["ln_pre.bias"])
    N = t.shape[0]; hd = D // Hh
    for i in range(12):
        p = "transformer.resblocks.%d." % i
        h = _ln(t, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = h @ w[p + "attn.in_proj_weight"].T + w[p + "attn.in_proj_bias"]
        q, k, v = (qkv[:, j * D:(j + 1) * D].reshape(N, Hh, hd).transpose(1, 0, 2) for j in range(3))
        a = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(N, D)
        t = t + o @ w[p + "attn.out_proj.weight"].T + w[p + "attn.out_proj.bias"]
        h = _ln(t, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        h = h @ w[p + "mlp.c_fc.weight"].T + w[p + "mlp.c_fc.bias"]
        h = h * (1.0 / (1.0 + np.exp(-1.702 * h)))             # QuickGELU（OpenAI 版）
        t = t + h @ w[p + "mlp.c_proj.weight"].T + w[p + "mlp.c_proj.bias"]
    c = _ln(t[0], w["ln_post.weight"], w["ln_post.bias"])
    return (c @ w["proj"]).astype(np.float32)

