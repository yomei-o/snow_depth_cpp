// snow_cli — 道路の写真から雪の段階を出す（python/snow.py の CLI と同じ）。
//   snow_cli feats   weights.bin labels.json datadir out.feats   ラベル付き写真の特徴を保存
//   snow_cli train   in.feats head.json [lam epochs lr]            層を学習して保存
//   snow_cli predict weights.bin head.json image... [--feat]       判定
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "snow.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <chrono>

static std::string slurp(const std::string& fn) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + fn);
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
static void spit(const std::string& fn, const std::string& s) {
    std::ofstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + fn);
    f << s;
}
static clipvit::Weights load_w(const std::string& fn) {
    std::string b = slurp(fn); clipvit::Weights w; w.load((const uint8_t*)b.data(), b.size()); return w;
}
static bool load_img(const std::string& fn, clipvit::Image& im) {
    int W, H, Cn; unsigned char* d = stbi_load(fn.c_str(), &W, &H, &Cn, 3);
    if (!d) return false;
    im.w = W; im.h = H; im.px.assign(d, d + (size_t)W * H * 3); stbi_image_free(d); return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: snow_cli feats|train|predict ...\n"); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "feats" && argc == 6) {
            clipvit::Weights w = load_w(argv[2]);
            clipvit::Json lab = clipvit::Json::parse(slurp(argv[3]));
            std::string dir = argv[4];
            std::vector<float> F; std::vector<int> road, around;
            for (auto& e : lab.arr) {
                clipvit::Image im;
                if (!load_img(dir + "/" + e["file"].str, im)) throw std::runtime_error("cannot load " + e["file"].str);
                auto f = clipvit::encode_image(w, clipvit::preprocess(im, true));
                F.insert(F.end(), f.begin(), f.end());
                road.push_back((int)e["road"].num); around.push_back((int)e["around"].num);
            }
            uint32_t N = (uint32_t)road.size(), D = 512;
            std::string o = "SNWF"; o.append((const char*)&N, 4); o.append((const char*)&D, 4);
            o.append((const char*)F.data(), F.size() * 4);
            o.append((const char*)road.data(), (size_t)N * 4); o.append((const char*)around.data(), (size_t)N * 4);
            spit(argv[5], o); std::printf("%u images -> %s\n", N, argv[5]);
        } else if (cmd == "train" && (argc == 4 || argc == 7)) {
            double lam = argc == 7 ? std::atof(argv[4]) : snow::LAM;
            int epochs = argc == 7 ? std::atoi(argv[5]) : snow::EPOCHS;
            double lr = argc == 7 ? std::atof(argv[6]) : snow::LR;
            snow::Feats fs = snow::load_feats(slurp(argv[2]));
            std::vector<double> X;
            for (int n = 0; n < fs.N; n++) {
                std::vector<float> f(fs.F.begin() + (size_t)n * fs.D, fs.F.begin() + (size_t)(n + 1) * fs.D);
                auto x = snow::normalize(f);
                X.insert(X.end(), x.begin(), x.end());
            }
            snow::Model m = snow::fit(X, fs.road, fs.around, fs.D, lam, epochs, lr);
            char note[128];
            std::snprintf(note, sizeof note, "trained by C++ on %d images, lam=%g epochs=%d lr=%g", fs.N, lam, epochs, lr);
            spit(argv[3], snow::save_model(m, note)); std::printf("%s -> %s\n", note, argv[3]);
        } else if (cmd == "predict" && argc >= 5) {
            clipvit::Weights w = load_w(argv[2]);
            snow::Model m = snow::load_model(slurp(argv[3]));
            bool feat = false;
            for (int i = 4; i < argc; i++) if (std::string(argv[i]) == "--feat") feat = true;
            for (int i = 4; i < argc; i++) {
                if (std::string(argv[i]).rfind("--", 0) == 0) continue;
                clipvit::Image im;
                if (!load_img(argv[i], im)) { std::fprintf(stderr, "cannot load %s\n", argv[i]); continue; }
                auto t0 = std::chrono::steady_clock::now();
                auto f = clipvit::encode_image(w, clipvit::preprocess(im, true));
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (feat) { for (size_t k = 0; k < f.size(); k++) std::printf(k ? " %.6f" : "%.6f", f[k]); std::printf("\n"); continue; }
                auto x = snow::normalize(f);
                auto r = snow::predict(m, m.road, x), a = snow::predict(m, m.around, x);
                std::printf("%s\troad=%d %.1fcm (", argv[i], r.level, r.cm);
                for (int c = 0; c < snow::C; c++) std::printf(c ? " %.3f" : "%.3f", r.p[c]);
                std::printf(")\taround=%d %.1fcm\t(%.0f ms)\n", a.level, a.cm, ms);
            }
        } else { std::fprintf(stderr, "bad arguments\n"); return 1; }
    } catch (std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    return 0;
}
