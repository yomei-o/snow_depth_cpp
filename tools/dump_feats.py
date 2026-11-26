# ViT-B/32 の画像特徴（正規化済み）を npz に保存。python/oshare 系の numpy 実装ではなく open_clip で速く。
import sys, json, os, numpy as np, torch, open_clip
from PIL import Image
data, lab, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
m, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai"); m = m.cuda().eval()
def pad(im):
    w, h = im.size; n = max(w, h); o = Image.new("RGB", (n, n), (123, 117, 104)); o.paste(im, ((n - w) // 2, (n - h) // 2)); return o
F = []
with torch.no_grad():
    for e in lab:
        f = m.encode_image(prep(pad(Image.open(os.path.join(data, e["file"])).convert("RGB"))).unsqueeze(0).cuda())[0]
        F.append((f / f.norm()).cpu().numpy())
np.savez(out, F=np.array(F), road=np.array([e["road"] for e in lab]), around=np.array([e["around"] for e in lab]))
