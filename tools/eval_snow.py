# ゼロショットと線形プローブ（5 分割の交差検証）で段階の当たり方を測る。
# usage: python eval_snow.py data_dir labels.json prompts_src.json
import sys, json, os, numpy as np, torch, open_clip
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
data, lab, src = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3], encoding="utf-8"))
models = [("ViT-B-32-quickgelu", "openai"), ("ViT-B-16", "laion2b_s34b_b88k"), ("ViT-L-14-quickgelu", "openai")]
def pad(im):
    w, h = im.size; n = max(w, h); out = Image.new("RGB", (n, n), (123, 117, 104)); out.paste(im, ((n - w) // 2, (n - h) // 2)); return out
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
def report(name, pred_lv, y):
    r = np.clip(np.round(pred_lv), 0, 5)
    edge = np.isin(y, [0, 1, 2])
    print("  %-14s spearman=%.3f  exact=%.2f  within1=%.2f  MAE=%.2f | 0-2だけ exact=%.2f" % (
        name, spearman(pred_lv, y), (r == y).mean(), (abs(r - y) <= 1).mean(), abs(pred_lv - y).mean(), (r[edge] == y[edge]).mean()))
for mn, pre in models:
    m, _, prep = open_clip.create_model_and_transforms(mn, pretrained=pre); m = m.cuda().eval(); tok = open_clip.get_tokenizer(mn)
    with torch.no_grad():
        F = []
        for e in lab:
            im = Image.open(os.path.join(data, e["file"])).convert("RGB")
            f = m.encode_image(prep(pad(im)).unsqueeze(0).cuda())[0]; F.append((f / f.norm()).cpu().numpy())
        F = np.array(F)
        print(mn, pre)
        for key in ("road", "around"):
            y = np.array([e[key] for e in lab])
            T = []
            for lv in src[key]:
                t = m.encode_text(tok(lv["prompts"]).cuda()); t = t / t.norm(dim=-1, keepdim=True); t = t.mean(0); T.append((t / t.norm()).cpu().numpy())
            T = np.array(T); lg = 100 * F @ T.T; p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            report(key + " zero", p @ np.arange(6), y)
            # 線形プローブ: リッジ回帰で段階を直接回帰、5 分割 CV
            rng = np.random.RandomState(0); idx = rng.permutation(len(y)); pred = np.zeros(len(y))
            for k in range(5):
                te = idx[k::5]; tr = np.setdiff1d(idx, te)
                X = np.c_[F[tr], np.ones(len(tr))]; lam = 1.0
                wgt = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y[tr])
                pred[te] = np.c_[F[te], np.ones(len(te))] @ wgt
            report(key + " probe", pred, y)
