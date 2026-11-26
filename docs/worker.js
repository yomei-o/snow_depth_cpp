// 推論はここ（UI を止めないため）。重みは 2 分割で取ってきて連結する。
importScripts("snow.js");
let M = null;

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " " + res.status);
  const reader = res.body.getReader();
  const chunks = []; let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); n += value.length; onBytes(value.length);
  }
  const out = new Uint8Array(n); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function init() {
  M = await SnowModule();
  const parts = ["model/w0.bin", "model/w1.bin"];
  const TOTAL = 96931781; let got = 0;
  const bufs = await Promise.all(parts.map((u) => fetchWithProgress(u, (k) => {
    got += k; postMessage({ type: "progress", got, total: TOTAL });
  })));
  const n = bufs.reduce((a, b) => a + b.length, 0);
  const p = M._malloc(n); let o = 0;
  for (const b of bufs) { M.HEAPU8.set(b, p + o); o += b.length; }
  const okW = M._sn_load_weights(p, n); M._free(p);
  const pj = new Uint8Array(await (await fetch("model/head.json")).arrayBuffer());
  const q = M._malloc(pj.length); M.HEAPU8.set(pj, q);
  const okP = M._sn_load_head(q, pj.length); M._free(q);
  if (!okW || !okP) throw new Error("モデルの読み込みに失敗しました");
  postMessage({ type: "ready" });
}

onmessage = (e) => {
  const { id, rgba, w, h } = e.data;
  try {
    const p = M._malloc(rgba.length); M.HEAPU8.set(rgba, p);
    const t0 = performance.now();
    const r = JSON.parse(M.UTF8ToString(M._sn_judge(p, w, h)));
    M._free(p);
    postMessage({ type: "result", id, r, ms: performance.now() - t0 });
  } catch (err) {
    postMessage({ type: "error", id, message: String(err) });
  }
};

init().catch((err) => postMessage({ type: "error", message: String(err) }));
