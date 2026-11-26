# snow_depth_cpp — 道路の積雪チェッカー（C++ / WebAssembly / Python）

道路の写真から、**路面の雪**と**まわりの積雪**をそれぞれ 6 段階で出す。
段階は「雪なし／降り始め・うっすら／〜5cm／〜15cm／〜30cm／30cm超」。
カメラで撮った写真でも、読み込んだ写真でもよい。推論はブラウザの中だけで行い、写真は外に送らない。

デモ: https://yomei-o.github.io/snow_depth_cpp/

## しくみ

- 画像の特徴は **OpenAI CLIP ViT-B/32**（MIT）。エンジン `cpp/clipvit.hpp` / `python/clipvit.py` は
  [oshare_cpp](https://github.com/yomei-o/oshare_cpp) のものを取り出した共通部分（torch と画素単位・特徴 cos 1.0000000 で一致を確認済み）。
- その上に **線形 softmax の層**（512→6）を 2 つ（路面用・まわり用）載せる。`cpp/snow.hpp` / `python/snow.py`。
  - 特徴は L2 正規化 → 学習データの平均を引く → 10 倍。全データ一括の勾配降下（double、lam=3e-4、3000 回、lr=0.4）。
  - 学習も推論も C++ と Python の両方にある。同じ特徴ファイルから学習した重みの差は **3.9e-15**。
- cm の数字は、各段階の代表値（0, 0.5, 3, 10, 22, 45 cm）を確からしさで重みづけした目安。
- 路面とまわりを分けたのは、雪国では除雪された路面は乾いていても、路肩に 1m を超える雪の壁があることが多いから。

## 学習データ（目視ラベル）

Wikimedia Commons の道路の写真 282 枚を集めた（`tools/fetch_commons.py`、出典とライセンスは `model/sources/*.json`）。
雪国の雪道 90 枚と、きわどい写真 192 枚（東京の降り始め・シャーベット・霜・ブラックアイス・雨の日の路面・埋まった車）。
**Claude が一覧画像を目で見て段階をつけた**（`model/labels_src.txt`）。道路が写っていない 86 枚を外した 196 枚を使っている。
測った積雪ではなく、ラベルをつけた人も 1 人なので、ラベル自体にぶれがある。

| 段階 | 路面 | まわり |
|---|---|---|
| 雪なし | 52 | 14 |
| 降り始め・うっすら | 40 | 23 |
| 〜5cm | 60 | 65 |
| 〜15cm | 24 | 49 |
| 〜30cm | 15 | 24 |
| 30cm超 | 5 | 21 |

## 数字（5 分割の交差検証、自前エンジンの int8 特徴、`tools/cv.py`）

| | 段階が一致 | 1 段階以内 | 雪なし〜5cm の写真での一致 |
|---|---|---|---|
| 路面 | 59% | 92% | 64% |
| まわり | 55% | 89% | 53% |
| （参考）路面をゼロショット（言葉と比べるだけ） | 24% | 69% | 18% |

- いちばん多い間違いは、路面の「雪なし」と「降り始め・うっすら」の取り違え（52 枚中 11 枚、40 枚中 13 枚）。
- 大きいモデル（ViT-L/14）でも 1 段階以内が 95% と少ししか変わらなかったので、B/32 のままにした（`tools/eval_snow.py`）。
- デモのサンプル 13 枚は学習に使った写真なので当たりやすい。

## 使い方

```sh
# C++（Windows は VS2022）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build/Release/snow_cli predict models/vitb32_int8.bin model/head.json road.jpg
build/Release/snow_cli feats   models/vitb32_int8.bin model/labels.json data out.feats   # 写真から特徴を作る
build/Release/snow_cli train   model/train.feats head.json                                # 層を学習

# Python（numpy + Pillow だけ。サブコマンドは C++ と同じ）
python python/snow.py predict models/vitb32_int8.bin model/head.json road.jpg
python python/snow.py train model/train.feats head.json

# 突き合わせ
python tools/parity.py build/Release/snow_cli models/vitb32_int8.bin head_cpp.json head_py.json image...

# WASM（docs/snow.js を作る）
./build_wasm.sh
```

重み `models/vitb32_int8.bin` は `docs/model/w0.bin` と `w1.bin` を連結したもの。
`model/train.feats` は学習に使った 196 枚の特徴（C++ エンジンで作成）なので、写真を集め直さなくても学習を再現できる。

## ライセンス

- コード: MIT
- CLIP の重み: OpenAI CLIP（MIT）
- サンプル写真: Wikimedia Commons（CC BY / CC BY-SA / CC0 / パブリックドメイン。ページ下部と `docs/samples/credits.json` にクレジット）
