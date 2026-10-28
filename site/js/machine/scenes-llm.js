// Acts VIII–X: the real model, the phone, the whole picture, and a playground.
(function () {
"use strict";
const { A, X, Y, box, txt, line, arrow, frame, path, code } = M;
const range = n => [...Array(n).keys()];
const partial = (r, c) => A[r].slice(0, c + 1).reduce((s, v, i) => s + v * X[i], 0);
const fmtB = b => b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : b >= 1e6 ? (b / 1e6).toFixed(1) + " MB" : b >= 1e3 ? (b / 1e3).toFixed(1) + " KB" : b + " B";

// ---------------------------------------------------------------- the model

const PIPE = [
  ["prompt", "c-n", 64], ["tokens", "c-n", 104], ["embedding: look up a row", "c-n", 144],
  ["RMSNorm", "c-n", 206], ["Q, K, V projections", "c-a", 244], ["attention ← KV cache", "c-x", 282],
  ["O projection", "c-a", 320], ["RMSNorm", "c-n", 358], ["MLP: gate, up, down", "c-a", 396],
  ["logits: output matrix", "c-a", 452], ["next token", "c-r", 494]
];
M.scene("llm", {
  build: () => [
    PIPE.map(([s, cls, y], i) => [
      box(`pp${i}`, 180, y, s, cls, { w: 230, h: 28, fs: 12.5 }),
      i < PIPE.length - 1 && i !== 2 && i !== 8 ? arrow(`pa${i}`, 180, y + 14, 180, PIPE[i + 1][2] - 14, { cls: "faint" }) : []
    ]),
    frame("layer", 52, 184, 256, 234, "", { cls: "sub" }),
    txt("lyN", 306, 177, "× 24 layers", { anchor: "end", cls: "lbl b", fs: 12 }),
    arrow("pin", 180, 158, 180, 190, { cls: "faint" }),
    arrow("pout", 180, 418, 180, 438, { cls: "faint" }),
    line("zoom1", 296, 236, 400, 80, { cls: "zoom" }), line("zoom2", 296, 252, 400, 330, { cls: "zoom" }),
    frame("zp", 400, 80, 380, 250, "zoom: one copper box during decode", { cls: "hw" }),
    txt("z1", 420, 130, "Q projection, Qwen2.5-0.5B:", { anchor: "start", cls: "lbl b", fs: 13.5 }),
    txt("z2", 420, 154, "896 × 896 weights × this token's 896 numbers", { anchor: "start", cls: "lbl", fs: 13 }),
    txt("z3", 420, 186, "= a matrix × vector (GEMV)", { anchor: "start", cls: "lbl a b", fs: 13.5 }),
    txt("z4", 420, 218, "quantized weights → NEON sdot kernel:", { anchor: "start", cls: "lbl", fs: 13 }),
    range(4).map(j => box(`zl${j}`, 450 + 58 * j, 262, "", "c-r", { w: 52, h: 30, fs: 13, sub: j ? "" : "int32 lanes", subBelow: true })),
    txt("m1", 590, 380, "Copper boxes are matrix multiplies.", { cls: "lbl a b", fs: 13 }),
    txt("m2", 590, 404, "Measured, Phase 2 (scalar, stories15M):", { cls: "faint", fs: 12 }),
    txt("m3", 590, 426, "97.6 % of CPU cycles were in matmul.", { cls: "lbl b", fs: 13.5 }),
    txt("m4", 590, 466, "Attention grows with the context; norms and\nsampling are small. Not every step suits\nthe same processor.", { cls: "faint", fs: 12 })
  ],
  async play(a) {
    await a.wait(400);
    a.put(box("tok", 180, 64, "", "c-p", { w: 16, h: 16, rx: 8 }));
    const visit = async (i, ms = 260) => {
      await a.to("tok", { x: 60, y: PIPE[i][2] }, ms);
      if (PIPE[i][1] === "c-a") { a.pulse(`pp${i}`, 1.06, 140); if (i === 4) for (let j = 0; j < 4; j++) { a.to(`zl${j}`, { txt: [18, 19, 0, 0][j] }, 0); a.pulse(`zl${j}`, 1.12, 140); } }
    };
    for (const i of [0, 1, 2]) await visit(i);
    for (let L = 1; L <= 3; L++) {
      a.to("lyN", { txt: L < 3 ? `layer ${L} of 24` : "… 24 of 24" }, 0);
      for (let i = 3; i <= 8; i++) await visit(i, L === 1 ? 300 : 130);
    }
    await visit(9);
    await visit(10);
    await a.to("pp10", { txt: "next token: “fox”" }, 0);
    a.to("lyN", { txt: "× 24 layers" }, 0);
  }
});

const PRE = ["The", "quick", "brown", "fox"];
const DEC = ["jumps", "over", "the"];
M.scene("prefill", {
  init: { n: 0 },
  controls: [{
    key: "n", label: "Tokens processed together", type: "range", min: 0, max: 9,
    fmt: v => String(2 ** v), live: (a, st) => prefillRead(a, st)
  }],
  build: () => [
    txt("hp", 40, 62, "PREFILL: the whole prompt at once", { anchor: "start", cls: "lbl b", fs: 14 }),
    frame("d1", 40, 80, 140, 150, "DRAM", { cls: "hw" }),
    box("w1", 110, 150, "weight row", "c-a", { w: 110, h: 26, fs: 12 }),
    frame("c1", 220, 80, 140, 150, "core", { cls: "hw" }),
    PRE.map((t, i) => box(`pt${i}`, 440 + 86 * i, 110, t, "c-x", { w: 76, h: 30, fs: 13 })),
    PRE.map((_, i) => box(`pr${i}`, 440 + 86 * i, 180, "", "ghost", { w: 76, h: 30, fs: 13 })),
    txt("pc", 600, 235, "", { cls: "mono", fs: 13 }),
    line("mid", 30, 268, 770, 268, { cls: "faint dash" }),
    txt("hd", 40, 300, "DECODE: one new token per step", { anchor: "start", cls: "lbl b", fs: 14 }),
    frame("d2", 40, 318, 140, 150, "DRAM", { cls: "hw" }),
    box("w2", 110, 388, "weight row", "c-a", { w: 110, h: 26, fs: 12 }),
    frame("c2", 220, 318, 140, 150, "core", { cls: "hw" }),
    DEC.map((t, i) => box(`dt${i}`, 440 + 86 * i, 348, t, "c-x", { w: 76, h: 30, fs: 13, o: 0 })),
    DEC.map((_, i) => box(`dr${i}`, 440 + 86 * i, 418, "", "ghost", { w: 76, h: 30, fs: 13, o: 0 })),
    txt("dc", 600, 475, "", { cls: "mono", fs: 13 }),
    txt("cap", 400, 525, "Same weights, same arithmetic. Prefill reuses each weight; decode fetches it again every step.", { cls: "lbl", fs: 12.5 })
  ],
  after: (a, st) => prefillRead(a, st),
  async play(a) {
    await a.wait(400);
    await a.fly("w1m", "w1", 290, 150, 700);
    a.to("pc", { txt: "weight row loaded 1×" }, 0);
    await a.all(PRE.map((_, i) => a.fly(`pf${i}`, "w1m", 440 + 86 * i, 150, 500, { w: 30, h: 10, fs: 0.1 })));
    for (let i = 0; i < 4; i++) { a.del(`pf${i}`, 100); a.to(`pr${i}`, { txt: "·", cls: "c-p" }, 0); a.pulse(`pr${i}`, 1.15, 150); }
    await a.to("pc", { txt: "weight row loaded 1×, used 4×" }, 0);
    await a.wait(600);
    for (let i = 0; i < 3; i++) {
      await a.all([a.to(`dt${i}`, { o: 1 }, 300), a.to(`dr${i}`, { o: 1 }, 300)]);
      await a.fly(`w2m${i}`, "w2", 290, 388, 600);
      await a.fly(`df${i}`, `w2m${i}`, 440 + 86 * i, 388, 400, { w: 30, h: 10, fs: 0.1 });
      a.del(`df${i}`, 100);
      a.del(`w2m${i}`, 200);
      a.to(`dr${i}`, { txt: "·", cls: "c-p" }, 0);
      a.pulse(`dr${i}`, 1.15, 150);
      await a.to("dc", { txt: `weight row loaded ${i + 1}×, used ${i + 1}×` }, 0);
      await a.wait(250);
    }
  }
});
function prefillRead(a, st) {
  const n = 2 ** st.n;
  a.read(`<p><strong>${n} token${n > 1 ? "s" : ""} together:</strong> each weight fetched from memory feeds ${n} multiply-add${n > 1 ? "s" : ""}. ${n === 1
    ? "That is decode: the core mostly waits on memory, whichever processor it is."
    : n < 32 ? "The arithmetic per byte grows; memory matters less." : "Now the limit is how fast the hardware can multiply. Here GPUs and NPUs can pull ahead."}</p>
    <table><thead><tr><th>Qwen2.5-0.5B Q4_0, cores 2–7</th><th>prefill</th><th>decode</th></tr></thead><tbody>
    <tr><td>Aether</td><td>165 tok/s (512 tokens)</td><td>128.8 tok/s</td></tr>
    <tr><td>llama.cpp</td><td>631 tok/s (256 tokens)</td><td>125.0 tok/s</td></tr></tbody></table>
    <p><span class="tag m">measured</span> Phase 7. Decode is level; prefill has room to grow because it is arithmetic, not memory.</p>`);
}

const MODELS = {
  s15: { name: "stories15M", layers: 6, kv: 288, w: 60.8e6, wName: "weights, f32", cap: 256 },
  smol: { name: "SmolLM2-135M", layers: 30, kv: 192, w: 143.0e6, wName: "weights, Q8_0", cap: 8192 },
  qwen: { name: "Qwen2.5-0.5B", layers: 24, kv: 128, w: 278e6, wName: "weights, Q4_0", cap: 32768 }
};
const kvShown = 14;
function kvState(st) {
  const m = MODELS[st.model];
  const n = Math.min(2 ** st.ctx + st.extra, m.cap);
  const per = m.layers * m.kv * 2 * st.bytes;
  return { m, n, per, total: per * n };
}
M.scene("kv", {
  init: { model: "qwen", bytes: 4, ctx: 4, extra: 0 },
  controls: [
    { key: "model", label: "Model", opts: [["s15", "stories15M"], ["smol", "SmolLM2-135M"], ["qwen", "Qwen2.5-0.5B"]], action: (v, st) => { st.model = v; st.extra = 0; } },
    { key: "bytes", label: "Cache stored as", opts: [[4, "f32"], [2, "f16"]] },
    { key: "ctx", label: "Context length", type: "range", min: 0, max: 15, fmt: (v, st) => String(Math.min(2 ** v, MODELS[st.model].cap)) }
  ],
  buttons: [{ label: "Add a token", run: kvAdd }],
  build: st => {
    const { m, n, per, total } = kvState(st);
    const shown = Math.min(n, kvShown);
    const col = t => 60 + 44 * t;
    const big = Math.max(total, m.w);
    const bw = v => Math.max(2, (v / big) * 560);
    return [
      txt("kt", 40, 52, `${m.name}: ${m.layers} layers × ${m.kv} numbers per K (and per V) per token`, { anchor: "start", cls: "faint", fs: 12 }),
      txt("lk", 34, 140, "K", { cls: "lbl x b", fs: 13 }),
      txt("lv", 34, 182, "V", { cls: "lbl p b", fs: 13 }),
      range(shown).map(t => {
        if (n > kvShown && t === kvShown - 2) return [];
        const label = n > kvShown && t === kvShown - 1 ? n : t + 1;
        return [
          box(`kt${t}`, col(t), 92, label, "c-n", { w: 38, h: 26, fs: label > 999 ? 9 : 11 }),
          box(`kk${t}`, col(t), 136, "", "c-x", { w: 38, h: 34 }),
          box(`kv${t}`, col(t), 178, "", "c-p", { w: 38, h: 34 })
        ];
      }),
      n > kvShown ? txt("kmore", col(kvShown - 2), 162, "…", { cls: "lbl b", fs: 22 }) : [],
      txt("kn", 400, 232, `${n} token${n > 1 ? "s" : ""} in the cache${n === m.cap ? " (this model's limit)" : ""}`, { cls: "lbl b", fs: 14 }),
      txt("bt", 40, 290, "memory", { anchor: "start", cls: "faint b", fs: 12 }),
      txt("bkl", 40, 325, "KV cache", { anchor: "start", cls: "lbl b", fs: 13 }),
      box("bk", 160 + bw(total) / 2, 320, "", "c-x", { w: bw(total), h: 26, rx: 3 }),
      txt("bkv", 168 + bw(total), 325, fmtB(total), { anchor: "start", cls: "mono b", fs: 13 }),
      txt("bwl", 40, 365, m.wName, { anchor: "start", cls: "lbl", fs: 13 }),
      box("bw", 160 + bw(m.w) / 2, 360, "", "c-a", { w: bw(m.w), h: 26, rx: 3 }),
      txt("bwv", 168 + bw(m.w), 365, fmtB(m.w), { anchor: "start", cls: "mono", fs: 13 }),
      txt("per", 400, 420, `one token adds ${m.layers} × ${m.kv} × 2 × ${st.bytes} bytes = ${fmtB(per)}`, { cls: "mono", fs: 13.5 }),
      txt("rd", 400, 452, `each decode step reads the weights and the whole cache: ${fmtB(m.w + total)}`, { cls: "lbl", fs: 13.5 }),
      txt("src", 400, 520, "Calculated from each model's shapes; weight sizes are the measured file sizes (f32 for stories15M).", { cls: "faint", fs: 11.5 })
    ];
  }
});
async function kvAdd(a, st) {
  const { n } = kvState(st);
  const shown = Math.min(n, kvShown);
  const col = t => 60 + 44 * t;
  a.put(box("q", 760, 92, "q", "c-w", { w: 34, h: 26, fs: 13 }));
  a.put(txt("qn", 760, 66, "new token", { cls: "faint", fs: 11 }));
  await a.wait(300);
  for (let t = 0; t < shown; t++) a.put(line(`ql${t}`, 760, 106, col(t), 120, { cls: "c-w", o: 0.7 }));
  await a.wait(700);
  for (let t = 0; t < shown; t++) a.del(`ql${t}`, 200);
  a.put(box("nk", 760, 136, "", "c-x", { w: 38, h: 34 }));
  a.put(box("nv", 760, 178, "", "c-p", { w: 38, h: 34 }));
  const tx = col(Math.min(shown, kvShown - 1));
  await a.all([a.to("nk", { x: tx }, 600), a.to("nv", { x: tx }, 600), a.to("q", { x: tx, cls: "c-n", txt: "" }, 600)]);
  st.extra += 1;
  await a.show(M._scene("kv").build(st), 300);
}

// ---------------------------------------------------------------- the phone

const MODE = {
  cpu: { bars: [["c-p", 520, "CPU cores 2–7 do every step"]], route: [] },
  gpu: { bars: [["c-w", 70, "dispatch"], ["c-p", 300, "GPU compute"], ["c-w", 60, "sync"], ["c-p", 70, "CPU"]], route: ["gpu"] },
  npu: { bars: [["c-w", 80, "invoke"], ["c-p", 220, "NPU compute"], ["c-w", 60, "return"], ["c-p", 70, "CPU"]], route: ["npu"] }
};
const SPOT = { cpu: [185, 180], gpu: [375, 180], npu: [495, 180] };
M.scene("phone", {
  init: { mode: "cpu" },
  controls: [{ key: "mode", label: "Route", opts: [["cpu", "CPU only"], ["gpu", "CPU → GPU → CPU"], ["npu", "CPU → NPU → CPU"]] }],
  build: st => [
    frame("ph", 20, 30, 760, 500, "phone", { cls: "sub", rx: 28 }),
    frame("soc", 50, 66, 510, 290, "Snapdragon 8 Elite (SM8750)", { cls: "hw" }),
    frame("cl", 70, 100, 230, 156, "CPU · 8 Oryon cores", { cls: "sub" }),
    range(8).map(i => box(`core${i}`, 104 + 54 * (i % 4), 150 + 48 * Math.floor(i / 4), i, i < 6 ? "c-n" : "c-r", { w: 44, h: 36, fs: 14 })),
    txt("clN", 185, 248, "0–5 performance · 6–7 prime", { cls: "faint", fs: 10.5 }),
    frame("gpu", 316, 100, 118, 156, "Adreno GPU", { cls: "sub" }),
    path("gd", M.dotsD(8, 8, 11, 3), { x: 336, y: 136, cls: "dots" }),
    frame("npu", 446, 100, 100, 156, "Hexagon NPU", { cls: "sub" }),
    path("nd", range(5).map(j => range(5).map(i => `M${i * 14} ${j * 14}h11v11h-11Z`).join("")).join(""), { x: 461, y: 140, cls: "cellsP" }),
    frame("slc", 70, 276, 476, 60, "system-level cache (shared)", { cls: "sub" }),
    frame("dram", 600, 66, 160, 290, "LPDDR5X DRAM", { cls: "hw" }),
    txt("dramT", 680, 200, "weights,\nKV cache,\nactivations", { cls: "lbl", fs: 13 }),
    line("bus", 546, 306, 600, 306, { cls: "thick" }),
    txt("tl", 40, 392, "one decode step (illustration)", { anchor: "start", cls: "faint b", fs: 12 }),
    txt("facts", 400, 500, "Measured: one prime core reads DRAM at ~61 GB/s, all CPU cores ~67 GB/s. GPU and NPU read the same DRAM.", { cls: "faint", fs: 11.5 })
  ],
  async play(a, st) {
    const m = MODE[st.mode];
    await a.wait(400);
    a.put(box("job", 185, 180, "", "c-p", { w: 18, h: 18, rx: 9 }));
    let x = 40;
    const bar = async ([cls, w, label], i) => {
      a.put(box(`tb${i}`, x, 430, label, cls, { w: 0, h: 30, fs: 11.5, rx: 3 }));
      const p = a.to(`tb${i}`, { w: w - 2, x: x + w / 2 }, w * 3.5, true);
      x += w;
      return p;
    };
    if (st.mode === "cpu") {
      for (let i = 2; i < 8; i++) a.pulse(`core${i}`, 1.1, 300);
      await bar(m.bars[0], 0);
    } else {
      const [tx, ty] = SPOT[st.mode];
      const pb = bar(m.bars[0], 0);
      await a.to("job", { x: tx, y: ty, cls: "c-w" }, m.bars[0][1] * 3.5);
      await pb;
      a.to("job", { cls: "c-p" }, 0);
      await a.all([bar(m.bars[1], 1), a.to(st.mode === "gpu" ? "gd" : "nd", { o: 0.35 }, 300).then(() => a.to(st.mode === "gpu" ? "gd" : "nd", { o: 1 }, 300))]);
      const pr = bar(m.bars[2], 2);
      await a.to("job", { x: 185, y: 180, cls: "c-w" }, m.bars[2][1] * 3.5);
      await pr;
      a.to("job", { cls: "c-p" }, 0);
      await bar(m.bars[3], 3);
    }
    a.del("job", 300);
  }
});

const PROF = [["Embedding", 2], ["Attention", 18], ["Quantized GEMV", 62], ["KV cache", 8], ["Other", 10]];
const PCLS = ["c-n", "c-x", "c-a", "c-p", "c-n"];
M.scene("profile", {
  init: { f: {} },
  build: st => {
    const ms = PROF.map(([n, v]) => v / (st.f[n] || 1));
    const total = ms.reduce((s, v) => s + v, 0);
    let sx = 230;
    return [
      txt("ill", 40, 60, "Illustration: a made-up profile of one 100 ms inference step", { anchor: "start", cls: "warn", fs: 12 }),
      PROF.map(([n], i) => [
        txt(`pl${i}`, 40, 105 + 46 * i, n, { anchor: "start", cls: i === 2 ? "lbl b" : "lbl", fs: 13.5 }),
        box(`pb${i}`, 230 + ms[i] * 2.5, 100 + 46 * i, "", PCLS[i], { w: ms[i] * 5, h: 28, rx: 3 }),
        txt(`pv${i}`, 238 + ms[i] * 5, 105 + 46 * i, `${+ms[i].toFixed(1)} ms`, { anchor: "start", cls: "mono b", fs: 13 })
      ]),
      txt("tlb", 40, 365, "Total", { anchor: "start", cls: "lbl b", fs: 14 }),
      ms.map((v, i) => { const b = box(`ts${i}`, sx + v * 2.5, 360, "", PCLS[i], { w: Math.max(0, v * 5 - 1), h: 30, rx: 2 }); sx += v * 5; return b; }),
      txt("tv", sx + 10, 366, `${+total.toFixed(1)} ms`, { anchor: "start", cls: "mono b", fs: 15 }),
      txt("note", 400, 420, st.note || "", { cls: "cap", fs: 15 }),
      txt("m1", 400, 470, "On this phone: Phase 2's profile had matmul at 97.6 % of cycles. Phase 3's NEON kernels", { cls: "faint", fs: 12 }),
      txt("m2", 400, 488, "took decode from 88.6 to 422 tok/s (×4.8) on the same core. Measured.", { cls: "faint", fs: 12 }),
      txt("loop", 400, 528, "measure → find the biggest bar → improve it → measure again", { cls: "lbl p b", fs: 14 })
    ];
  },
  buttons: [
    { label: "GEMV 2× faster", run: (a, st) => prof(a, st, "Quantized GEMV") },
    { label: "Attention 2× faster", run: (a, st) => prof(a, st, "Attention") },
    { label: "Embedding 2× faster", run: (a, st) => prof(a, st, "Embedding") },
    { label: "Reset", run: (a, st) => { st.f = {}; st.note = ""; return a.show(M._scene("profile").build(st), 500); } }
  ],
  async play(a) {
    await a.wait(500);
    await a.add(txt("bn", 230 + 62 * 5 + 70, 197, "← bottleneck", { anchor: "start", cls: "warn b", fs: 13 }));
    await a.pulse("pb2", 1.05, 200);
  }
});
function prof(a, st, name) {
  const before = PROF.reduce((s, [n, v]) => s + v / (st.f[n] || 1), 0);
  const old = PROF.find(p => p[0] === name)[1] / (st.f[name] || 1);
  st.f[name] = (st.f[name] || 1) * 2;
  const after = before - old / 2;
  st.note = `${name}: ${+old.toFixed(1)} → ${+(old / 2).toFixed(1)} ms.  Total: ${+before.toFixed(1)} → ${+after.toFixed(1)} ms.`;
  return a.show(M._scene("profile").build(st), 600);
}

// ---------------------------------------------------------------- the whole picture

const CHAIN = ["LLM", "transformer layer", "matrix × vector", "dot product", "multiply + add"];
const BR = [
  ["CPU", "NEON lanes", "registers"],
  ["GPU", "kernel", "threads, workgroups"],
  ["NPU", "tensor op", "accelerator"]
];
const brX = i => 200 + 200 * i;
M.scene("final", {
  build: () => [
    CHAIN.map((s, i) => [
      box(`ch${i}`, 90 + 155 * i, 70, s, i === 4 ? "c-p" : "c-n", { w: 140, h: 36, fs: 13 }),
      i < 4 ? arrow(`cha${i}`, 160 + 155 * i, 70, 175 + 155 * i, 70, { cls: "faint" }) : []
    ]),
    txt("zi", 400, 112, "zoom in →", { cls: "faint", fs: 11.5 }),
    path("brp", "M710 88V140H400V170M400 196V214M200 214H600M200 214V236M400 214V236M600 214V236", { cls: "tree" }),
    box("op", 400, 184, "one operation", "c-p", { w: 150, h: 26, fs: 12.5 }),
    BR.map((b, i) => [
      box(`b0${i}`, brX(i), 254, b[0], "c-n", { w: 120, h: 34, fs: 15 }),
      arrow(`ba${i}`, brX(i), 271, brX(i), 292, { cls: "faint" }),
      box(`b1${i}`, brX(i), 308, b[1], "c-n", { w: 150, h: 30, fs: 12.5 }),
      arrow(`bb${i}`, brX(i), 323, brX(i), 344, { cls: "faint" }),
      box(`b2${i}`, brX(i), 360, b[2], "c-n", { w: 160, h: 30, fs: 12.5 })
    ]),
    path("join", "M200 375V400H600V375M400 375V400M400 400V420", { cls: "tree" }),
    box("res", 400, 438, "the same result", "c-r", { w: 150, h: 34, fs: 13.5 }),
    txt("fq", 400, 500, "", { cls: "cap", fs: 16 })
  ],
  async play(a) {
    await a.wait(400);
    for (let i = 0; i < 5; i++) { await a.pulse(`ch${i}`, 1.08, 160); }
    await a.pulse("op", 1.1, 160);
    for (let l = 0; l < 3; l++) {
      await a.all(range(3).map(i => a.pulse(`b${l}${i}`, 1.08, 160)));
    }
    await a.pulse("res", 1.12, 180);
    await a.wait(1600);
    // zoom back out: prefill and decode, and who decides
    await a.show([
      box("ml", 400, 70, "a mobile LLM", "c-p", { w: 170, h: 36, fs: 15 }),
      path("t1", "M400 88V110M230 110H570M230 110V128M570 110V128", { cls: "tree" }),
      box("pf", 230, 150, "PREFILL", "c-n", { w: 170, h: 36, fs: 15, sub: "lots of parallel arithmetic", subBelow: true }),
      box("dc", 570, 150, "DECODE", "c-n", { w: 170, h: 36, fs: 15, sub: "small repeated steps, memory-bound", subBelow: true }),
      path("t2", "M230 186V208H570V186M400 208V236", { cls: "tree" }),
      box("rt", 400, 256, "the runtime decides", "c-w", { w: 200, h: 36, fs: 14 }),
      path("t3", "M400 274V292M220 292H580M220 292V312M400 292V312M580 292V312", { cls: "tree" }),
      ["CPU", "GPU", "NPU"].map((s, i) => box(`fx${i}`, 220 + 180 * i, 332, s, "c-n", { w: 110, h: 36, fs: 15 })),
      txt("fq", 400, 420, "", { cls: "cap", fs: 17 })
    ], 900);
    await a.wait(600);
    await a.to("fq", { txt: "The hardware is not “running AI”. It is executing enormous numbers of\nsimple operations, moving data through memory, registers, SIMD lanes,\nthreads and specialised accelerators." }, 0);
  }
});

const SL = { memX: c => 74 + 34 * c, memWY: 74, memXY: 112, laneX: () => 182, v1y: 182, v2y: 228, v0y: 288, sumX: 300, sumY: 288, codeX: 40, codeY: 380, step: 34, cw: 30 };
const VL = { memX: c => 482 + 34 * c, memWY: 74, memXY: 112, laneX: j => 548 + 34 * j, v1y: 182, v2y: 228, v0y: 288, sumX: 728, sumY: 288, codeX: 430, codeY: 360, step: 34, cw: 30 };
const SCODE = ["let mut acc = 0.0f32;", "for i in 0..8 {", "    acc += weights[i] * acts[i];", "}"];
const VCODE = ["let mut acc = vdupq_n_f32(0.0);", "for c in (0..8).step_by(4) {", "    let w = vld1q_f32(&weights[c]);", "    let a = vld1q_f32(&acts[c]);", "    acc = vfmaq_f32(acc, w, a);", "}", "let sum = vaddvq_f32(acc);"];
M.scene("code", {
  build: () => {
    const out = [
      txt("hs", 200, 36, "scalar: one pair per step", { cls: "lbl b", fs: 14 }),
      txt("hv", 600, 36, "NEON: four pairs per step", { cls: "lbl p b", fs: 14 }),
      line("div", 400, 50, 400, 530, { cls: "faint dash" }),
      M.neonLayout(1, SL, false, "s", true),
      M.neonLayout(4, VL, false, "v", true),
      code("snc", SL.codeX, SL.codeY, SCODE, -1, { w: 330, fs: 11.5, lh: 18 }),
      code("vnc", VL.codeX, VL.codeY, VCODE, -1, { w: 340, fs: 11.5, lh: 18 }),
      txt("sc_ld", 40, 470, "", { anchor: "start", cls: "mono", fs: 12 }),
      txt("sc_fma", 40, 490, "", { anchor: "start", cls: "mono", fs: 12 }),
      txt("vc_ld", 430, 500, "", { anchor: "start", cls: "mono", fs: 12 }),
      txt("vc_fma", 430, 518, "", { anchor: "start", cls: "mono", fs: 12 }),
      txt("vc_red", 600, 500, "", { anchor: "start", cls: "mono", fs: 12 })
    ];
    return out;
  },
  play: a => a.all([
    M.neonRun(a, 1, SL, 0.8, "s"),
    M.neonRun(a, 4, VL, 0.8, "v")
  ])
});

// ---------------------------------------------------------------- playground

const MAXL = { f32: 4, f16: 8, i8: 16 };
const BYTES = { f32: 4, f16: 2, i8: 1 };
function plan(st) {
  if (st.hw === "gpu") return { over: [6, 3], ticks: 8, lanes: 1, note: "4 threads, one row each, all at once; dispatch and sync around them." };
  if (st.hw === "npu") return { over: [5, 3], ticks: 1, lanes: 32, note: st.dt === "f32" ? "NPUs are built for int8, int16 and fp16; an FP32 model is usually converted first." : "The whole 4 × 8 tile in one tensor operation." };
  const want = st.hw === "scalar" ? 1 : st.W;
  const lanes = Math.min(want, MAXL[st.dt]);
  const rows = Math.ceil(4 / st.T);
  const per = Math.ceil(8 / lanes) + (lanes > 1 ? 1 : 0);
  let note = st.hw === "scalar" ? "One pair per instruction." : `${lanes} ${st.dt.toUpperCase()} lanes per instruction.`;
  if (st.hw === "neon" && want > MAXL[st.dt]) note = `NEON registers are 128 bits: at most ${MAXL[st.dt]} × ${st.dt.toUpperCase()}. Using ${lanes}.`;
  if (st.hw === "neon" && st.dt === "i8" && lanes === 16) note = "sdot: 16 int8 products per instruction; only 8 are ours.";
  return { over: [0, 0], ticks: rows * per, lanes, rows, per, note };
}
M.scene("play", {
  init: { hw: "neon", dt: "f32", W: 4, T: 1 },
  controls: [
    { key: "hw", label: "Hardware", opts: [["scalar", "CPU scalar"], ["neon", "CPU NEON"], ["gpu", "GPU"], ["npu", "NPU"]] },
    { key: "dt", label: "Data type", opts: [["f32", "FP32"], ["f16", "FP16"], ["i8", "INT8"]] },
    { key: "W", label: "Vector width", opts: [[1, "1"], [4, "4"], [8, "8"], [16, "16"]] },
    { key: "T", label: "CPU cores", opts: [[1, "1"], [2, "2"], [4, "4"]] }
  ],
  build: st => {
    const p = plan(st);
    const cpu = st.hw === "scalar" || st.hw === "neon";
    const out = [
      range(4).map(r => range(8).map(c => box(`a${r}_${c}`, 46 + 26 * c, 104 + 26 * r, A[r][c], "c-a", { w: 23, h: 23, fs: 11, rx: 3 }))),
      X.map((v, c) => box(`x${c}`, 46 + 26 * c, 230, v, "c-x", { w: 23, h: 23, fs: 11, rx: 3 })),
      txt("lA", 137, 78, "A", { cls: "lbl a b", fs: 13 }), txt("lx", 137, 212, "x", { cls: "lbl x b", fs: 13 }),
      range(4).map(r => box(`y${r}`, 60 + 50 * r, 300, "", "ghost", { w: 44, h: 32, fs: 15 })),
      txt("ly", 30, 305, "y", { cls: "lbl r b", fs: 13 }),
      txt("pn", 40, 360, p.note, { anchor: "start", cls: "lbl", fs: 12.5 }),
      txt("bytes", 40, 386, `weights moved: 32 × ${BYTES[st.dt]} bytes = ${32 * BYTES[st.dt]} bytes`, { anchor: "start", cls: "faint mono", fs: 12 }),
      txt("tlT", 40, 440, "time, in steps (illustration)", { anchor: "start", cls: "faint b", fs: 12 }),
      line("tax", 40, 486, 760, 486, { cls: "faint" }),
      txt("tsum", 760, 440, "", { anchor: "end", cls: "mono b", fs: 13 })
    ];
    if (cpu) {
      const h = 280 / st.T;
      for (let k = 0; k < st.T; k++) {
        const y0 = 62 + h * k;
        out.push(frame(`core${k}`, 300, y0, 470, h - 10, `core ${k}`, { cls: "hw" }));
        const lw = Math.min(40, 400 / p.lanes);
        for (let j = 0; j < p.lanes; j++) out.push(box(`ln${k}_${j}`, 330 + lw * j + lw / 2, y0 + h / 2 + 2, "", "reg", { w: lw - 4, h: Math.min(30, h / 3), fs: Math.min(13, lw / 2.4) }));
        out.push(txt(`cr${k}`, 760, y0 + 20, "", { anchor: "end", cls: "mono faint", fs: 11.5 }));
      }
    } else if (st.hw === "gpu") {
      out.push(frame("gpu", 300, 62, 470, 280, "GPU", { cls: "hw" }));
      range(4).map(r => out.push(box(`th${r}`, 350, 124 + 62 * r, "", "c-p", { w: 50, h: 32, rx: 16, fs: 13, sub: `thread ${r}` })));
    } else {
      out.push(frame("npu", 300, 62, 470, 280, "NPU tensor unit · conceptual", { cls: "hw" }));
      range(4).map(r => range(8).map(c => out.push(box(`g${r}_${c}`, 380 + 44 * c, 126 + 50 * r, "", "reg", { w: 36, h: 34, fs: 13 }))));
    }
    return out;
  },
  async play(a, st) {
    const p = plan(st);
    const U = 26, TK = 260;
    let x = 40;
    const seg = (cls, n, label) => {
      const k = `t${x}`;
      a.put(box(k, x, 470, label, cls, { w: 0, h: 26, fs: 10.5, rx: 3 }));
      const pr = a.to(k, { w: n * U - 2, x: x + (n * U) / 2 }, n * TK, true);
      x += n * U;
      return pr;
    };
    await a.wait(300);
    if (p.over[0]) await seg("c-w", p.over[0], st.hw === "gpu" ? "dispatch" : "invoke");
    if (st.hw === "scalar" || st.hw === "neon") {
      const segP = seg("c-p", p.ticks, "compute");
      const L = p.lanes;
      await a.all(range(st.T).map(async k => {
        for (let r = k * p.rows; r < Math.min(4, (k + 1) * p.rows); r++) {
          a.to(`cr${k}`, { txt: `row ${r}` }, 0);
          let acc = 0;
          for (let c0 = 0; c0 < 8; c0 += L) {
            for (let j = 0; j < L; j++) {
              const c = c0 + j;
              a.to(`ln${k}_${j}`, { txt: c < 8 ? A[r][c] * X[c] : "", cls: c < 8 ? "c-p" : "ghost" }, 0);
              if (c < 8) { acc += A[r][c] * X[c]; a.pulse(`a${r}_${c}`, 1.2, 90); }
            }
            await a.wait(TK);
          }
          if (L > 1) await a.wait(TK);
          a.to(`y${r}`, { txt: acc, cls: "c-r" }, 0);
          a.pulse(`y${r}`, 1.15, 120);
        }
        a.to(`cr${k}`, { txt: "done" }, 0);
      }));
      await segP;
    } else if (st.hw === "gpu") {
      const segP = seg("c-p", p.ticks, "compute");
      for (let c = 0; c < 8; c++) {
        for (let r = 0; r < 4; r++) { a.to(`th${r}`, { txt: partial(r, c) }, 0); a.pulse(`a${r}_${c}`, 1.2, 90); }
        await a.wait(TK);
      }
      for (let r = 0; r < 4; r++) a.to(`y${r}`, { txt: Y[r], cls: "c-r" }, 0);
      await segP;
    } else {
      const segP = seg("c-p", p.ticks, "");
      for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) a.to(`g${r}_${c}`, { txt: A[r][c] * X[c], cls: "c-p" }, 0);
      await segP;
      for (let r = 0; r < 4; r++) a.to(`y${r}`, { txt: Y[r], cls: "c-r" }, 0);
    }
    if (p.over[1]) await seg("c-w", p.over[1], st.hw === "gpu" ? "sync" : "return");
    const total = p.over[0] + p.ticks + p.over[1];
    a.to("tsum", { txt: `${total} steps${p.over[0] ? ` (${p.over[0] + p.over[1]} of them overhead)` : ""}` }, 0);
  }
});
})();
