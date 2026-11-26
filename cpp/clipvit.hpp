// clipvit.hpp — CLIP ViT-B/32（OpenAI, QuickGELU）の画像側。依存なしのヘッダオンリー。
// oshare_cpp の oshare.hpp から取り出した共通部分。python/clipvit.py と同じ計算をする
// （前処理は PIL の bicubic と整数演算まで同じ）。
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <cctype>

namespace clipvit {

// ------------------------------------------------------------------ 重み
// Linear 重みは int8（行ごと scale）のまま持ち、計算時に float にする。
struct Tensor {
    std::vector<uint32_t> shape;
    std::vector<float> f;        // 型0: そのまま
    std::vector<int8_t> q;       // 型1: 転置して [in][out] で持つ（axpy 形で回すため）
    std::vector<float> scale;    // 型1: out ごと
    bool quant = false;
    size_t numel() const { size_t n = 1; for (auto s : shape) n *= s; return n; }
};

struct Weights {
    std::map<std::string, Tensor> t;
    const Tensor& at(const std::string& k) const {
        auto it = t.find(k);
        if (it == t.end()) throw std::runtime_error("weight not found: " + k);
        return it->second;
    }
    void load(const uint8_t* b, size_t n) {
        size_t o = 0;
        auto need = [&](size_t k) { if (o + k > n) throw std::runtime_error("weights truncated"); };
        need(12);
        if (std::memcmp(b, "OSHW", 4) != 0) throw std::runtime_error("bad weights magic");
        uint32_t cnt; std::memcpy(&cnt, b + 8, 4); o = 12;
        for (uint32_t i = 0; i < cnt; i++) {
            need(2); uint16_t L; std::memcpy(&L, b + o, 2); o += 2;
            need(L + 2); std::string k((const char*)b + o, L); o += L;
            uint8_t ty = b[o], nd = b[o + 1]; o += 2;
            Tensor T; T.shape.resize(nd);
            need(4 * nd); std::memcpy(T.shape.data(), b + o, 4 * nd); o += 4 * nd;
            size_t ne = T.numel();
            if (ty == 0) {
                need(4 * ne); T.f.resize(ne); std::memcpy(T.f.data(), b + o, 4 * ne); o += 4 * ne;
            } else {
                uint32_t R = T.shape[0], C = T.shape[1];
                need(4 * R + ne);
                T.scale.resize(R); std::memcpy(T.scale.data(), b + o, 4 * R); o += 4 * R;
                const int8_t* src = (const int8_t*)(b + o); o += ne;
                T.q.resize(ne);
                for (uint32_t r = 0; r < R; r++)
                    for (uint32_t c = 0; c < C; c++) T.q[(size_t)c * R + r] = src[(size_t)r * C + c];
                T.quant = true;
            }
            t[k] = std::move(T);
        }
    }
};

// ------------------------------------------------------------------ 前処理（PIL 互換）
static const float MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
static const float STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

struct Image { int w = 0, h = 0; std::vector<uint8_t> px; };  // RGB

inline double cubic_(double x) {
    const double a = -0.5; x = std::fabs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

// PIL の precompute_coeffs + normalize_coeffs_8bpc と同じ
inline void coeffs_(int in, int out, std::vector<int>& xmin, std::vector<int>& xlen, std::vector<std::vector<int32_t>>& k) {
    const int PREC = 22;
    double scale = (double)in / out, fs = std::max(scale, 1.0), support = 2.0 * fs;
    xmin.resize(out); xlen.resize(out); k.resize(out);
    for (int i = 0; i < out; i++) {
        double center = (i + 0.5) * scale;
        int lo = std::max((int)(center - support + 0.5), 0);
        int hi = std::min((int)(center + support + 0.5), in);
        std::vector<double> ws;
        double tot = 0;
        for (int x = lo; x < hi; x++) { double w = cubic_((x - center + 0.5) / fs); ws.push_back(w); tot += w; }
        xmin[i] = lo; xlen[i] = hi - lo; k[i].resize(ws.size());
        for (size_t j = 0; j < ws.size(); j++) {
            double w = tot != 0 ? ws[j] / tot : ws[j];
            k[i][j] = (int32_t)(w * (1 << PREC) + (w >= 0 ? 0.5 : -0.5));
        }
    }
}

inline uint8_t clip8_(int64_t acc) {
    int64_t v = acc >> 22;
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

inline Image resize_h_(const Image& s, int nw) {
    std::vector<int> xm, xl; std::vector<std::vector<int32_t>> k;
    coeffs_(s.w, nw, xm, xl, k);
    Image d; d.w = nw; d.h = s.h; d.px.resize((size_t)nw * s.h * 3);
    for (int y = 0; y < s.h; y++)
        for (int x = 0; x < nw; x++)
            for (int c = 0; c < 3; c++) {
                int64_t acc = 1 << 21;
                for (int j = 0; j < xl[x]; j++) acc += (int64_t)s.px[((size_t)y * s.w + xm[x] + j) * 3 + c] * k[x][j];
                d.px[((size_t)y * nw + x) * 3 + c] = clip8_(acc);
            }
    return d;
}

inline Image resize_v_(const Image& s, int nh) {
    std::vector<int> ym, yl; std::vector<std::vector<int32_t>> k;
    coeffs_(s.h, nh, ym, yl, k);
    Image d; d.w = s.w; d.h = nh; d.px.resize((size_t)s.w * nh * 3);
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < s.w; x++)
            for (int c = 0; c < 3; c++) {
                int64_t acc = 1 << 21;
                for (int j = 0; j < yl[y]; j++) acc += (int64_t)s.px[((size_t)(ym[y] + j) * s.w + x) * 3 + c] * k[y][j];
                d.px[((size_t)y * s.w + x) * 3 + c] = clip8_(acc);
            }
    return d;
}

inline Image pil_bicubic(const Image& s, int w, int h) {
    Image r = s;
    if (r.w != w) r = resize_h_(r, w);
    if (r.h != h) r = resize_v_(r, h);
    return r;
}

inline Image pad_square(const Image& s) {
    int n = std::max(s.w, s.h);
    Image d; d.w = d.h = n; d.px.resize((size_t)n * n * 3);
    uint8_t fill[3];
    for (int c = 0; c < 3; c++) fill[c] = (uint8_t)std::nearbyint(MEAN[c] * 255.0f);
    for (size_t i = 0; i < (size_t)n * n; i++) for (int c = 0; c < 3; c++) d.px[i * 3 + c] = fill[c];
    int oy = (n - s.h) / 2, ox = (n - s.w) / 2;
    for (int y = 0; y < s.h; y++) std::memcpy(&d.px[((size_t)(y + oy) * n + ox) * 3], &s.px[(size_t)y * s.w * 3], (size_t)s.w * 3);
    return d;
}

// 戻り値: (3,224,224) の float。pad=true なら先に正方形へ余白を足す
inline std::vector<float> preprocess(const Image& src, bool pad = true, int size = 224) {
    Image im = pad ? pad_square(src) : src;
    int nw, nh;
    if (im.w <= im.h) { nw = size; nh = (int)((double)size * im.h / im.w); }
    else { nh = size; nw = (int)((double)size * im.w / im.h); }
    Image r = pil_bicubic(im, nw, nh);
    int top = (int)std::nearbyint((nh - size) / 2.0), left = (int)std::nearbyint((nw - size) / 2.0);
    std::vector<float> x((size_t)3 * size * size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < size; y++)
            for (int xx = 0; xx < size; xx++) {
                float v = r.px[((size_t)(y + top) * nw + xx + left) * 3 + c] / 255.0f;
                x[((size_t)c * size + y) * size + xx] = (v - MEAN[c]) / STD[c];
            }
    return x;
}

// ------------------------------------------------------------------ ViT
// Y[N][out] = X[N][in] · W^T + b。W は [out][in]（f32）か、転置 int8 [in][out]。
inline void linear_(const float* X, int N, int in, const Tensor& W, const Tensor* b, float* Y) {
    const int out = (int)W.shape[0];
    const int OB = 256;
    std::vector<float> wbuf(OB);
    for (int ob = 0; ob < out; ob += OB) {
        int on = std::min(OB, out - ob);
        for (int t = 0; t < N; t++) std::fill(Y + (size_t)t * out + ob, Y + (size_t)t * out + ob + on, 0.0f);
        for (int i = 0; i < in; i++) {
            float* wb = wbuf.data();
            if (W.quant) { const int8_t* q = &W.q[(size_t)i * out + ob]; for (int o = 0; o < on; o++) wb[o] = (float)q[o]; }
            else { for (int o = 0; o < on; o++) wb[o] = W.f[(size_t)(ob + o) * in + i]; }
            for (int t = 0; t < N; t++) {
                float xv = X[(size_t)t * in + i];
                float* y = Y + (size_t)t * out + ob;
                for (int o = 0; o < on; o++) y[o] += xv * wb[o];
            }
        }
        for (int t = 0; t < N; t++) {
            float* y = Y + (size_t)t * out + ob;
            if (W.quant) for (int o = 0; o < on; o++) y[o] *= W.scale[ob + o];
            if (b) for (int o = 0; o < on; o++) y[o] += b->f[ob + o];
        }
    }
}

inline void layernorm_(float* x, int N, int D, const Tensor& g, const Tensor& b, float* y) {
    for (int t = 0; t < N; t++) {
        const float* r = x + (size_t)t * D; float* o = y + (size_t)t * D;
        double m = 0; for (int i = 0; i < D; i++) m += r[i]; m /= D;
        double v = 0; for (int i = 0; i < D; i++) { double d = r[i] - m; v += d * d; } v /= D;
        float inv = (float)(1.0 / std::sqrt(v + 1e-5));
        for (int i = 0; i < D; i++) o[i] = (float)(r[i] - m) * inv * g.f[i] + b.f[i];
    }
}

// x: preprocess の出力 → 正規化前の 512 次元
inline std::vector<float> encode_image(const Weights& w, const std::vector<float>& x) {
    const int P = 32, G = 7, D = 768, H = 12, HD = 64, N = G * G + 1, K = 3 * P * P;
    // パッチ化: (G*G, 3*32*32)、並びは conv1.weight の (c, ky, kx)
    std::vector<float> patches((size_t)G * G * K);
    for (int gy = 0; gy < G; gy++) for (int gx = 0; gx < G; gx++)
        for (int c = 0; c < 3; c++) for (int ky = 0; ky < P; ky++) for (int kx = 0; kx < P; kx++)
            patches[(size_t)(gy * G + gx) * K + (c * P + ky) * P + kx] = x[((size_t)c * 224 + gy * P + ky) * 224 + gx * P + kx];
    Tensor cw = w.at("conv1.weight"); cw.shape = {(uint32_t)D, (uint32_t)K};
    std::vector<float> t((size_t)N * D), h((size_t)N * D);
    linear_(patches.data(), G * G, K, cw, nullptr, &t[D]);
    const auto& ce = w.at("class_embedding").f; const auto& pe = w.at("positional_embedding").f;
    for (int i = 0; i < D; i++) t[i] = ce[i];
    for (size_t i = 0; i < (size_t)N * D; i++) t[i] += pe[i];
    layernorm_(t.data(), N, D, w.at("ln_pre.weight"), w.at("ln_pre.bias"), t.data());
    std::vector<float> qkv((size_t)N * 3 * D), att((size_t)N * D), tmp((size_t)N * D), mlp((size_t)N * 4 * D), sc(N);
    for (int L = 0; L < 12; L++) {
        std::string p = "transformer.resblocks." + std::to_string(L) + ".";
        layernorm_(t.data(), N, D, w.at(p + "ln_1.weight"), w.at(p + "ln_1.bias"), h.data());
        linear_(h.data(), N, D, w.at(p + "attn.in_proj_weight"), &w.at(p + "attn.in_proj_bias"), qkv.data());
        const float s = 1.0f / std::sqrt((float)HD);
        for (int hh = 0; hh < H; hh++)
            for (int i = 0; i < N; i++) {
                const float* q = &qkv[(size_t)i * 3 * D + hh * HD];
                float mx = -1e30f;
                for (int j = 0; j < N; j++) {
                    const float* k = &qkv[(size_t)j * 3 * D + D + hh * HD];
                    float a = 0; for (int d = 0; d < HD; d++) a += q[d] * k[d];
                    sc[j] = a * s; mx = std::max(mx, sc[j]);
                }
                float sum = 0; for (int j = 0; j < N; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                float* o = &att[(size_t)i * D + hh * HD];
                for (int d = 0; d < HD; d++) o[d] = 0;
                for (int j = 0; j < N; j++) {
                    const float* v = &qkv[(size_t)j * 3 * D + 2 * D + hh * HD];
                    float a = sc[j] / sum;
                    for (int d = 0; d < HD; d++) o[d] += a * v[d];
                }
            }
        linear_(att.data(), N, D, w.at(p + "attn.out_proj.weight"), &w.at(p + "attn.out_proj.bias"), tmp.data());
        for (size_t i = 0; i < t.size(); i++) t[i] += tmp[i];
        layernorm_(t.data(), N, D, w.at(p + "ln_2.weight"), w.at(p + "ln_2.bias"), h.data());
        linear_(h.data(), N, D, w.at(p + "mlp.c_fc.weight"), &w.at(p + "mlp.c_fc.bias"), mlp.data());
        for (auto& v : mlp) v = v / (1.0f + std::exp(-1.702f * v));   // QuickGELU
        linear_(mlp.data(), N, 4 * D, w.at(p + "mlp.c_proj.weight"), &w.at(p + "mlp.c_proj.bias"), tmp.data());
        for (size_t i = 0; i < t.size(); i++) t[i] += tmp[i];
    }
    std::vector<float> c(D);
    layernorm_(t.data(), 1, D, w.at("ln_post.weight"), w.at("ln_post.bias"), c.data());
    const auto& pr = w.at("proj"); const int E = (int)pr.shape[1];   // proj は (768, 512)
    std::vector<float> f(E, 0.0f);
    for (int i = 0; i < D; i++) for (int e = 0; e < E; e++) f[e] += c[i] * pr.f[(size_t)i * E + e];
    return f;
}

// ------------------------------------------------------------------ prompts.json（必要な分だけの JSON 読み）
struct Json {
    enum T { NUL, NUM, STR, ARR, OBJ, BOOL } type = NUL;
    double num = 0; std::string str; std::vector<Json> arr; std::vector<std::pair<std::string, Json>> obj;
    const Json& operator[](const char* k) const {
        for (auto& kv : obj) if (kv.first == k) return kv.second;
        throw std::runtime_error(std::string("json key missing: ") + k);
    }
    static Json parse(const std::string& s) { size_t i = 0; return val_(s, i); }
private:
    static void ws_(const std::string& s, size_t& i) { while (i < s.size() && std::isspace((unsigned char)s[i])) i++; }
    static std::string str_(const std::string& s, size_t& i) {
        std::string r; i++;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                char e = s[++i];
                if (e == 'u') {
                    unsigned cp = std::stoul(s.substr(i + 1, 4), nullptr, 16); i += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && s[i + 1] == '\\') { unsigned lo = std::stoul(s.substr(i + 3, 4), nullptr, 16); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
                    if (cp < 0x80) r += (char)cp;
                    else if (cp < 0x800) { r += (char)(0xC0 | cp >> 6); r += (char)(0x80 | (cp & 63)); }
                    else if (cp < 0x10000) { r += (char)(0xE0 | cp >> 12); r += (char)(0x80 | (cp >> 6 & 63)); r += (char)(0x80 | (cp & 63)); }
                    else { r += (char)(0xF0 | cp >> 18); r += (char)(0x80 | (cp >> 12 & 63)); r += (char)(0x80 | (cp >> 6 & 63)); r += (char)(0x80 | (cp & 63)); }
                } else r += e == 'n' ? '\n' : e == 't' ? '\t' : e;
                i++;
            } else r += s[i++];
        }
        i++; return r;
    }
    static Json val_(const std::string& s, size_t& i) {
        ws_(s, i); Json j;
        if (s[i] == '{') {
            j.type = OBJ; i++; ws_(s, i);
            if (s[i] == '}') { i++; return j; }
            for (;;) { ws_(s, i); std::string k = str_(s, i); ws_(s, i); i++; j.obj.emplace_back(k, val_(s, i)); ws_(s, i); if (s[i++] == '}') break; }
        } else if (s[i] == '[') {
            j.type = ARR; i++; ws_(s, i);
            if (s[i] == ']') { i++; return j; }
            for (;;) { j.arr.push_back(val_(s, i)); ws_(s, i); if (s[i++] == ']') break; }
        } else if (s[i] == '"') { j.type = STR; j.str = str_(s, i); }
        else if (s.compare(i, 4, "true") == 0) { j.type = BOOL; j.num = 1; i += 4; }
        else if (s.compare(i, 5, "false") == 0) { j.type = BOOL; i += 5; }
        else if (s.compare(i, 4, "null") == 0) { i += 4; }
        else { j.type = NUM; size_t n; j.num = std::stod(s.substr(i, 32), &n); i += n; }
        return j;
    }
};

}  // namespace clipvit
