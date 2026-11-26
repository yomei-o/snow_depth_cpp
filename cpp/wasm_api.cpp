// ブラウザ用の入口。重みと head.json を受け取り、RGBA 画素から路面とまわりの雪の段階を返す。
#include "snow.hpp"
#include <emscripten/emscripten.h>
#include <string>

static clipvit::Weights g_w;
static snow::Model g_m;
static std::string g_json;

extern "C" {
EMSCRIPTEN_KEEPALIVE int sn_load_weights(const uint8_t* p, int n) {
    try { g_w = clipvit::Weights(); g_w.load(p, (size_t)n); return 1; } catch (...) { return 0; }
}
EMSCRIPTEN_KEEPALIVE int sn_load_head(const char* p, int n) {
    try { g_m = snow::load_model(std::string(p, (size_t)n)); return 1; } catch (...) { return 0; }
}
// rgba: w*h*4。戻り値は JSON（UTF-8、次の呼び出しまで有効）
EMSCRIPTEN_KEEPALIVE const char* sn_judge(const uint8_t* rgba, int w, int h) {
    clipvit::Image im; im.w = w; im.h = h; im.px.resize((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) for (int c = 0; c < 3; c++) im.px[i * 3 + c] = rgba[i * 4 + c];
    auto x = snow::normalize(clipvit::encode_image(g_w, clipvit::preprocess(im, true)));
    char b[64];
    g_json = "{\"levels\":[";
    for (int c = 0; c < snow::C; c++) {
        std::snprintf(b, sizeof b, "%g", g_m.levels[c].cm);
        g_json += (c ? ",[" : "[") + snow::json_str_(g_m.levels[c].label) + "," + b + "]";
    }
    g_json += "]";
    for (int k = 0; k < 2; k++) {
        auto r = snow::predict(g_m, k ? g_m.around : g_m.road, x);
        g_json += k ? ",\"around\":{" : ",\"road\":{";
        std::snprintf(b, sizeof b, "\"level\":%d,\"cm\":%.3f,\"p\":[", r.level, r.cm); g_json += b;
        for (int c = 0; c < snow::C; c++) { std::snprintf(b, sizeof b, c ? ",%.5f" : "%.5f", r.p[c]); g_json += b; }
        g_json += "]}";
    }
    g_json += "}";
    return g_json.c_str();
}
}
