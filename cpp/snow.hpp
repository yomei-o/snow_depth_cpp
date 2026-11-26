// snow.hpp — CLIP の画像特徴から「路面の雪」と「まわりの積雪」の段階を出す線形 softmax 層。
// 学習（全データ一括の勾配降下、double）も推論もここ。python/snow.py と同じ計算をする。
#pragma once
#include "clipvit.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

namespace snow {

constexpr int C = 6;          // 段階の数
constexpr double XS = 10.0;   // 特徴の倍率（正規化済み特徴は値が小さいので学習率を揃えるため）
// 学習の既定値。特徴から平均を引くので、勾配降下の安定限界は lr≈1.3（引かないと 0.06 で、
// その外で回すと端で振動して C++ と Python の丸めの差が 1e-5 まで増幅された）。
constexpr double LAM = 3e-4; constexpr int EPOCHS = 3000; constexpr double LR = 0.4;

struct Level { std::string label; double cm; };

// 段階の名前と代表値（python/snow.py の LEVELS と同じ）
inline std::vector<Level> default_levels() {
    return {{"雪なし", 0}, {"降り始め・うっすら", 0.5}, {"〜5cm", 3}, {"〜15cm", 10}, {"〜30cm", 22}, {"30cm超", 45}};
}

struct HeadW {                 // W[(D+1)*C]、最後の行がバイアス。z = ((x - mean) * XS)・W + b
    std::vector<double> W;
};

struct Model {
    std::vector<Level> levels;
    HeadW road, around;
    std::vector<double> mean;   // 学習データの特徴（正規化済み）の平均
    int D = 512;
};

// 特徴は L2 正規化してから使う
inline std::vector<double> normalize(const std::vector<float>& f) {
    double n = 0; for (float v : f) n += (double)v * v; n = std::sqrt(n);
    std::vector<double> o(f.size()); for (size_t i = 0; i < f.size(); i++) o[i] = f[i] / n;
    return o;
}

// x は平均を引いて XS 倍したもの
inline void probs(const HeadW& h, const double* x, int D, double* p) {
    double z[C], mx = -1e300;
    for (int c = 0; c < C; c++) {
        double s = h.W[(size_t)D * C + c];
        for (int i = 0; i < D; i++) s += x[i] * h.W[(size_t)i * C + c];
        z[c] = s; if (s > mx) mx = s;
    }
    double sum = 0; for (int c = 0; c < C; c++) { z[c] = std::exp(z[c] - mx); sum += z[c]; }
    for (int c = 0; c < C; c++) p[c] = z[c] / sum;
}

// X: N×D（平均を引いて XS 倍したもの）、y: 0..C-1。W=0 から epochs 回、全データの勾配で更新。
// バイアスには正則化をかけない。
inline HeadW train(const std::vector<double>& X, const std::vector<int>& y, int D, double lam, int epochs, double lr) {
    const int N = (int)y.size();
    HeadW h; h.W.assign((size_t)(D + 1) * C, 0.0);
    std::vector<double> G((size_t)(D + 1) * C), P((size_t)N * C);
    for (int ep = 0; ep < epochs; ep++) {
        for (int n = 0; n < N; n++) probs(h, &X[(size_t)n * D], D, &P[(size_t)n * C]);
        std::fill(G.begin(), G.end(), 0.0);
        for (int n = 0; n < N; n++) {
            const double* x = &X[(size_t)n * D];
            for (int c = 0; c < C; c++) {
                double g = P[(size_t)n * C + c] - (y[n] == c ? 1.0 : 0.0);
                for (int i = 0; i < D; i++) G[(size_t)i * C + c] += x[i] * g;
                G[(size_t)D * C + c] += g;
            }
        }
        for (size_t k = 0; k < G.size(); k++) {
            double g = G[k] / N;
            if (k < (size_t)D * C) g += lam * h.W[k];
            h.W[k] -= lr * g;
        }
    }
    return h;
}

struct Pred {
    double p[C];
    double cm;       // 段階の代表値の期待値
    int level;       // 確率が一番高い段階
};

// 正規化済みの特徴から平均を引いて XS 倍する
inline std::vector<double> center(const Model& m, const std::vector<double>& x) {
    std::vector<double> o(x.size()); for (size_t i = 0; i < x.size(); i++) o[i] = (x[i] - m.mean[i]) * XS; return o;
}

// x: normalize 済みの特徴
inline Pred predict(const Model& m, const HeadW& h, const std::vector<double>& x) {
    Pred r; std::vector<double> xc = center(m, x); probs(h, xc.data(), m.D, r.p);
    r.cm = 0; r.level = 0;
    for (int c = 0; c < C; c++) { r.cm += r.p[c] * m.levels[c].cm; if (r.p[c] > r.p[r.level]) r.level = c; }
    return r;
}

// ---- head.json の読み書き
inline std::vector<double> vec_(const clipvit::Json& a) { std::vector<double> v; for (auto& x : a.arr) v.push_back(x.num); return v; }

inline Model load_model(const std::string& text) {
    clipvit::Json j = clipvit::Json::parse(text);
    Model m; m.D = (int)j["dim"].num;
    for (auto& l : j["levels"].arr) m.levels.push_back({l["label"].str, l["cm"].num});
    m.road.W = vec_(j["road"]); m.around.W = vec_(j["around"]); m.mean = vec_(j["mean"]);
    if ((int)m.mean.size() != m.D || (int)m.levels.size() != C || m.road.W.size() != (size_t)(m.D + 1) * C || m.around.W.size() != m.road.W.size())
        throw std::runtime_error("head.json の形が合わない");
    return m;
}

inline std::string json_str_(const std::string& s) {
    std::string o = "\"";
    for (char ch : s) { if (ch == '"' || ch == '\\') o += '\\'; o += ch; }
    return o + "\"";
}

inline std::string save_model(const Model& m, const std::string& note) {
    std::string o = "{\"note\":" + json_str_(note) + ",\"dim\":" + std::to_string(m.D) + ",\"levels\":[";
    char b[64];
    for (size_t i = 0; i < m.levels.size(); i++) {
        std::snprintf(b, sizeof b, "%.17g", m.levels[i].cm);
        o += (i ? "," : "") + std::string("{\"label\":") + json_str_(m.levels[i].label) + ",\"cm\":" + b + "}";
    }
    o += "],\"mean\":[";
    for (int i = 0; i < m.D; i++) { std::snprintf(b, sizeof b, "%.17g", m.mean[i]); o += (i ? "," : "") + std::string(b); }
    o += "]";
    for (int k = 0; k < 2; k++) {
        const HeadW& h = k ? m.around : m.road;
        o += k ? ",\"around\":[" : ",\"road\":[";
        for (size_t i = 0; i < h.W.size(); i++) { std::snprintf(b, sizeof b, "%.17g", h.W[i]); o += (i ? "," : "") + std::string(b); }
        o += "]";
    }
    return o + "}";
}

// Xn: N×D の正規化済み特徴。平均を求めて中心化し、路面とまわりの 2 つを学習する
inline Model fit(const std::vector<double>& Xn, const std::vector<int>& road, const std::vector<int>& around, int D,
                 double lam = LAM, int epochs = EPOCHS, double lr = LR) {
    const int N = (int)road.size();
    Model m; m.D = D; m.levels = default_levels(); m.mean.assign(D, 0.0);
    for (int n = 0; n < N; n++) for (int i = 0; i < D; i++) m.mean[i] += Xn[(size_t)n * D + i];
    for (int i = 0; i < D; i++) m.mean[i] /= N;
    std::vector<double> X((size_t)N * D);
    for (int n = 0; n < N; n++) for (int i = 0; i < D; i++) X[(size_t)n * D + i] = (Xn[(size_t)n * D + i] - m.mean[i]) * XS;
    m.road = train(X, road, D, lam, epochs, lr);
    m.around = train(X, around, D, lam, epochs, lr);
    return m;
}

// ---- 特徴ファイル: "SNWF" u32 N u32 D float[N*D] i32 road[N] i32 around[N]
struct Feats { int N = 0, D = 0; std::vector<float> F; std::vector<int> road, around; };

inline Feats load_feats(const std::string& b) {
    Feats f;
    if (b.size() < 12 || b.compare(0, 4, "SNWF") != 0) throw std::runtime_error("bad feats file");
    uint32_t N, D; std::memcpy(&N, b.data() + 4, 4); std::memcpy(&D, b.data() + 8, 4);
    f.N = (int)N; f.D = (int)D;
    size_t o = 12, nf = (size_t)N * D;
    if (b.size() != o + nf * 4 + (size_t)N * 8) throw std::runtime_error("feats size mismatch");
    f.F.resize(nf); std::memcpy(f.F.data(), b.data() + o, nf * 4); o += nf * 4;
    f.road.resize(N); std::memcpy(f.road.data(), b.data() + o, N * 4); o += N * 4;
    f.around.resize(N); std::memcpy(f.around.data(), b.data() + o, N * 4);
    return f;
}

}  // namespace snow
