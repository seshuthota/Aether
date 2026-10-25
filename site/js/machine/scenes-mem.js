// Act IV: caches, then smaller numbers.
(function () {
"use strict";
const { A, X, box, txt, line, arrow, frame, path } = M;
const range = n => [...Array(n).keys()];

// ---------------------------------------------------------------- caches

const LV = [
  { k: "core", x: 24, w: 120, name: "core" },
  { k: "L1", x: 176, w: 124, name: "L1", note: "96 KB · per core" },
  { k: "L2", x: 330, w: 130, name: "L2", note: "12 MB · per cluster" },
  { k: "SLC", x: 492, w: 124, name: "system cache", note: "shared with GPU, NPU" },
  { k: "DRAM", x: 648, w: 132, name: "DRAM", note: "gigabytes · LPDDR5X" }
];
const lv = k => LV.find(l => l.k === k);
const mid = l => l.x + l.w / 2;
const SLOT = i => 240 + 48 * i;
const LINES = {
  x0: { txt: "x[0..15]", cls: "c-x", dram: 0, vals: X },
  far: { txt: "w[4096..4111]", cls: "c-a", dram: 2, vals: [7] }
};

M.scene("cache", {
  init: { lines: [] },
  build: st => [
    LV.map(l => [
      frame(`f_${l.k}`, l.x, 170, l.w, 220, l.name, { cls: l.k === "core" ? "hw" : "sub" }),
      l.note ? txt(`n_${l.k}`, mid(l), 410, l.note, { cls: "faint", fs: 11.5 }) : []
    ]),
    txt("near", 84, 150, "near, tiny", { cls: "faint", fs: 11.5 }),
    txt("far", 714, 150, "far, huge", { cls: "faint", fs: 11.5 }),
    arrow("dist", 150, 146, 650, 146, { cls: "faint dash" }),
    box("reg", 84, 280, st.reg == null ? "" : st.reg, st.reg == null ? "reg" : "c-x", { w: 74, h: 36, fs: 17, sub: "register" }),
    box("dl0", mid(lv("DRAM")), SLOT(0), LINES.x0.txt, "c-x", { w: 112, h: 24, fs: 11 }),
    box("dl1", mid(lv("DRAM")), SLOT(1), "…", "c-n", { w: 112, h: 24, fs: 11 }),
    box("dl2", mid(lv("DRAM")), SLOT(2), LINES.far.txt, "c-a", { w: 112, h: 24, fs: 11 }),
    st.lines.map((id, i) => ["L1", "L2", "SLC"].map(k =>
      box(`${k}_${id}`, mid(lv(k)), SLOT(i), LINES[id].txt, LINES[id].cls, { w: lv(k).w - 16, h: 24, fs: 11 }))),
    txt("stat", 400, 452, st.msg || "Use the load buttons under the picture.", { cls: "lbl b", fs: 15 }),
    txt("tlab", 40, 489, "time waiting", { anchor: "start", cls: "faint", fs: 12 }),
    box("tbar", 130 + (st.time || 0) / 2, 484, "", "c-w", { w: st.time || 0, h: 14, rx: 3 }),
    txt("tnote", 760, 489, "illustration", { anchor: "end", cls: "faint", fs: 11.5 }),
    txt("meas", 400, 530, "Measured streaming reads, one prime core: L1 ~102 GB/s, L2 ~67, DRAM ~61. Close, because the prefetcher fetches ahead.", { cls: "faint", fs: 11.5 })
  ],
  buttons: [
    { label: "Load x[0]", run: (a, st) => load(a, st, "x0", "x[0]", X[0]) },
    { label: "Load x[1]", run: (a, st) => load(a, st, "x0", "x[1]", X[1]) },
    { label: "Load a far-away weight", run: (a, st) => load(a, st, "far", "w[4096]", 7) },
    { label: "Empty the caches", run: (a, st) => { st.lines = []; st.reg = null; st.msg = ""; st.time = 0; return a.show(M._scene("cache").build(st), 400); } }
  ]
});

async function load(a, st, id, label, value) {
  const hit = st.lines.includes(id);
  for (const k of ["L1", "L2", "SLC"]) a.del(`miss_${k}`, 0);
  st.msg = `${label}: is it in L1?`;
  a.to("stat", { txt: st.msg }, 0);
  st.time = 0;
  a.to("tbar", { w: 0, x: 130 }, 150);
  a.put(box("req", 84, 280, "", "c-w", { w: 14, h: 14, rx: 7 }));
  const route = hit ? ["L1"] : ["L1", "L2", "SLC", "DRAM"];
  let t = 0;
  for (const k of route) {
    const l = lv(k);
    const ms = k === "DRAM" ? 700 : 380;
    t += k === "DRAM" ? 260 : hit ? 16 : 40;
    a.to("tbar", { w: t, x: 130 + t / 2 }, ms);
    await a.to("req", { x: mid(l), y: k === "DRAM" ? SLOT(LINES[id].dram) : 205 }, ms);
    if (k !== "DRAM" && !hit) {
      a.put(txt(`miss_${k}`, mid(l), 163, "miss", { cls: "warn b", fs: 12 }));
      await a.wait(200);
    }
  }
  let y;
  if (!hit) {
    if (st.lines.length >= 3) st.lines.shift();
    st.lines.push(id);
    y = SLOT(st.lines.indexOf(id));
    a.to("stat", { txt: "Miss. The whole 64-byte line comes back: 16 floats." }, 0);
    a.put(box("lineMove", mid(lv("DRAM")), SLOT(LINES[id].dram), LINES[id].txt, LINES[id].cls, { w: 112, h: 24, fs: 11 }));
    a.del("req", 150);
    for (const k of ["SLC", "L2", "L1"]) {
      t += 40;
      a.to("tbar", { w: t, x: 130 + t / 2 }, 420);
      await a.to("lineMove", { x: mid(lv(k)), y, w: lv(k).w - 16 }, 420);
      a.put(box(`${k}_${id}`, mid(lv(k)), y, LINES[id].txt, LINES[id].cls, { w: lv(k).w - 16, h: 24, fs: 11 }));
    }
    await a.del("lineMove", 0);
  } else {
    y = SLOT(st.lines.indexOf(id));
    await a.pulse(`L1_${id}`, 1.12, 150);
    a.del("req", 100);
  }
  a.put(box("val", mid(lv("L1")), y, value, LINES[id].cls, { w: 34, h: 28, fs: 15 }));
  await a.to("val", { x: 84, y: 280 }, hit ? 300 : 450);
  st.reg = value;
  await a.to("reg", { txt: value, cls: LINES[id].cls }, 0);
  await a.del("val", 0);
  st.time = t;
  st.msg = hit
    ? `Hit in L1. ${label} arrived earlier, in the same line as x[0].`
    : `Miss to DRAM: ${label} waited the longest, but its 15 neighbours came too.`;
  a.to("stat", { txt: st.msg }, 0);
}

// ---------------------------------------------------------------- bits

const DT = [
  { k: "f32", name: "FP32", bits: 32, lanes: "4 lanes" },
  { k: "f16", name: "FP16", bits: 16, lanes: "8 lanes" },
  { k: "i8", name: "INT8", bits: 8, lanes: "16 bytes → sdot" },
  { k: "q4", name: "4-bit", bits: 4, lanes: "unpacked to int8 first" }
];
const rects = (n, w, h, gap, pick) => {
  let d = "";
  for (let i = 0; i < n; i++) if (!pick || pick(i)) d += `M${i * (w + gap)} 0h${w}v${h}h${-w}Z`;
  return d;
};

M.scene("quant", {
  init: { dt: "f32" },
  controls: [{ key: "dt", label: "Number type", opts: DT.map(d => [d.k, d.name]) }],
  build: st => {
    const d = DT.find(t => t.k === st.dt);
    const perLine = 512 / d.bits, perReg = 128 / d.bits;
    const lw = 640 / perLine, rw = 320 / perReg;
    return [
      txt("bt", 40, 62, "bits per number", { anchor: "start", cls: "faint b", fs: 12 }),
      DT.map((t, i) => [
        txt(`bl${t.k}`, 40, 95 + 38 * i, `${t.name} · ${t.bits} bits`, { anchor: "start", cls: t.k === st.dt ? "lbl b" : "faint", fs: 13 }),
        path(`bp${t.k}`, rects(t.bits, 10, 20, 3), { x: 180, y: 84 + 38 * i, cls: t.k === st.dt ? "bits on" : "bits" })
      ]),
      txt("clT", 120, 278, "one 64-byte cache line holds", { anchor: "start", cls: "lbl", fs: 13 }),
      txt("clN", 760, 278, `${perLine} × ${d.name}`, { anchor: "end", cls: "lbl b", fs: 14 }),
      path(`clE${d.k}`, rects(perLine, lw - 1, 30, 1, i => i % 2 === 0), { x: 120, y: 290, cls: "cells even" }),
      path(`clO${d.k}`, rects(perLine, lw - 1, 30, 1, i => i % 2 === 1), { x: 120, y: 290, cls: "cells odd" }),
      txt("rgT", 120, 368, "one 128-bit NEON register holds", { anchor: "start", cls: "lbl", fs: 13 }),
      txt("rgN", 440, 368, `${perReg} × ${d.name}`, { anchor: "end", cls: "lbl b", fs: 14 }),
      path(`rgE${d.k}`, rects(perReg, rw - 1, 30, 1, i => i % 2 === 0), { x: 120, y: 380, cls: "cells even" }),
      path(`rgO${d.k}`, rects(perReg, rw - 1, 30, 1, i => i % 2 === 1), { x: 120, y: 380, cls: "cells odd" }),
      txt("rgL", 460, 400, d.lanes, { anchor: "start", cls: "lbl p b", fs: 13 }),
      txt("bwT", 400, 470, `At ~61 GB/s from DRAM, one core can be fed ${(61 / (d.bits / 8)).toFixed(0)} billion ${d.name} weights per second.`, { cls: "cap", fs: 15 }),
      txt("bwN", 400, 494, "Same bandwidth, smaller numbers: more weights per second.", { cls: "faint", fs: 12.5 })
    ];
  },
  after(a, st) {
    const rows = [
      ["f32", "FP32", "1.98 GB", "34", "—"],
      ["f16", "FP16", "0.99 GB", "68", "—"],
      ["i8", "Q8_0 (8.5 bits)", "525 MB", "128", "88.8"],
      ["q4", "Q4_0 (4.5 bits)", "278 MB", "241", "128.8"]
    ];
    a.read(`<table><thead><tr><th>Qwen2.5-0.5B as</th><th>bytes read per token</th><th>ceiling at 67 GB/s</th><th>measured decode</th></tr></thead><tbody>${rows.map(r =>
      `<tr${r[0] === st.dt ? ' class="on"' : ""}><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]} tok/s</td><td>${r[4] === "—" ? "—" : r[4] + " tok/s"}</td></tr>`).join("")}</tbody></table>
      <p><span class="tag c">calculated</span> ceiling = 67 GB/s ÷ bytes per token (FP32/FP16 from 494 M weights; Q8_0/Q4_0 file sizes). <span class="tag m">measured</span> decode, cores 2–7 (Phase 7).</p>`);
  }
});

// ---------------------------------------------------------------- scales

const qx = c => 150 + 56 * c;
M.scene("scale", {
  build: () => [
    txt("lr1", 40, 100, "real weights", { anchor: "start", cls: "faint b", fs: 12 }),
    A[0].map((v, c) => box(`rw${c}`, qx(c), 95, (v / 100).toFixed(2), "c-n", { w: 50, h: 30, fs: 13 })),
    arrow("ar1", qx(3.5), 116, qx(3.5), 168, { cls: "faint" }),
    txt("ar1l", qx(3.5) + 12, 146, "÷ 0.01, round", { anchor: "start", cls: "faint", fs: 12 }),
    txt("lr2", 40, 195, "int8, stored", { anchor: "start", cls: "lbl a b", fs: 12 }),
    A[0].map((v, c) => box(`a0_${c}`, qx(c), 190, v, "c-a", { w: 50, h: 36, fs: 18 })),
    box("scW", 660, 190, "scale 0.01", "c-w", { w: 130, h: 36, fs: 15, sub: "one per block" }),
    txt("lr3", 40, 285, "real inputs", { anchor: "start", cls: "faint b", fs: 12 }),
    X.map((v, c) => box(`rx${c}`, qx(c), 280, (v / 10).toFixed(1), "c-n", { w: 50, h: 30, fs: 13 })),
    arrow("ar2", qx(3.5), 301, qx(3.5), 353, { cls: "faint" }),
    txt("ar2l", qx(3.5) + 12, 331, "÷ 0.1, round", { anchor: "start", cls: "faint", fs: 12 }),
    txt("lr4", 40, 380, "int8, stored", { anchor: "start", cls: "lbl x b", fs: 12 }),
    X.map((v, c) => box(`x${c}`, qx(c), 375, v, "c-x", { w: 50, h: 36, fs: 18 })),
    box("scX", 660, 375, "scale 0.1", "c-w", { w: 130, h: 36, fs: 15, sub: "one per block" }),
    txt("rb", 400, 452, "", { cls: "mono b", fs: 15 }),
    txt("n1", 400, 492, "Real Q8_0 blocks: 32 int8 weights + one 16-bit scale = 34 bytes. The scale is the block's largest |w| ÷ 127,", { cls: "faint", fs: 11.5 }),
    txt("n2", 400, 510, "so the integers use the whole −127…127 range. Ours stay small so the arithmetic stays readable (illustration).", { cls: "faint", fs: 11.5 })
  ],
  async play(a) {
    await a.wait(600);
    await a.all([a.pulse("a0_3", 1.2, 200), a.pulse("scW", 1.1, 200)]);
    await a.to("rb", { txt: "Back to a real number: 4 × 0.01 = 0.04" }, 0);
    await a.pulse("rw3", 1.2, 200);
  }
});

// ---------------------------------------------------------------- sdot

const FLOW = ["int8 in memory", "vector load", "NEON registers", "sdot", "int32 accumulators", "× the two scales", "f32 output"];
const fy = k => 80 + 64 * k;
const bx = i => 226 + 31 * i;
const lx = g => bx(4 * g + 1.5);
const LANE = [18, 19, 0, 0];
M.scene("qdot", {
  build: () => [
    FLOW.map((f, k) => [
      box(`fl${k}`, 95, fy(k), f, "c-n", { w: 140, h: 34, fs: 12 }),
      k < 6 ? arrow(`fa${k}`, 95, fy(k) + 18, 95, fy(k + 1) - 18, { cls: "faint" }) : []
    ]),
    txt("mT", 210, 56, "16 bytes of weights and 16 of inputs (8 ours, 8 zero padding)", { anchor: "start", cls: "faint", fs: 11.5 }),
    range(16).map(i => i < 8
      ? [box(`a0_${i}`, bx(i), 82, A[0][i], "c-a", { w: 27, h: 28, fs: 13, rx: 4 }), box(`x${i}`, bx(i), 116, X[i], "c-x", { w: 27, h: 28, fs: 13, rx: 4 })]
      : [box(`wp${i}`, bx(i), 82, 0, "ghost", { w: 27, h: 28, fs: 13, rx: 4 }), box(`xp${i}`, bx(i), 116, 0, "ghost", { w: 27, h: 28, fs: 13, rx: 4 })]),
    frame("rv1", bx(0) - 17, 182, 31 * 16 + 3, 36, "", { cls: "reg", rx: 6 }),
    frame("rv2", bx(0) - 17, 222, 31 * 16 + 3, 36, "", { cls: "reg", rx: 6 }),
    txt("nv1", bx(15) + 24, 205, "v1", { anchor: "start", cls: "lbl", fs: 12 }),
    txt("nv2", bx(15) + 24, 245, "v2", { anchor: "start", cls: "lbl", fs: 12 }),
    range(4).map(g => box(`q0_${g}`, lx(g), 330, 0, "c-r", { w: 118, h: 34, fs: 16, sub: `int32 lane ${g}`, subBelow: true })),
    txt("nv0", bx(15) + 24, 335, "v0", { anchor: "start", cls: "lbl r", fs: 12 }),
    txt("qsum", 450, 410, "", { cls: "mono", fs: 15 }),
    txt("qout", 450, 444, "", { cls: "mono b", fs: 15 }),
    txt("meas", 450, 520, "Measured (Phase 6): Q8_0 GEMV ran at 12 GB/s when weights were turned back into f32 first, 32 GB/s with sdot.", { cls: "faint", fs: 11.5 })
  ],
  async play(a) {
    const step = async k => {
      for (let j = 0; j < FLOW.length; j++) a.to(`fl${j}`, { cls: j === k ? "c-p" : "c-n", s: j === k ? 1.06 : 1 }, 150);
      await a.wait(250);
    };
    await a.wait(400);
    await step(0);
    await a.wait(500);
    await step(1);
    await a.all(range(16).map(i => a.fly(`q1_${i}`, i < 8 ? `a0_${i}` : `wp${i}`, bx(i), 200, 650, { w: 27, h: 28 })));
    await a.all(range(16).map(i => a.fly(`q2_${i}`, i < 8 ? `x${i}` : `xp${i}`, bx(i), 240, 650, { w: 27, h: 28 })));
    await step(2);
    await a.wait(500);
    await step(3);
    for (let g = 0; g < 4; g++) {
      a.add(path(`br${g}`, `M${bx(4 * g) - 12} 262v8H${bx(4 * g + 3) + 12}v-8M${lx(g)} 270v38`, { cls: "bracket" }), 300);
      const terms = range(4).map(i => `${A[0][4 * g + i] || 0}·${X[4 * g + i] || 0}`).join("+");
      a.add(txt(`qh${g}`, lx(g) + 6, 292, g < 2 ? terms : "0", { anchor: "start", cls: "faint mono", fs: 10 }), 300);
    }
    await a.wait(700);
    for (let g = 0; g < 4; g++) { a.to(`q0_${g}`, { txt: LANE[g] }, 0); a.pulse(`q0_${g}`, 1.1, 150); }
    await a.wait(600);
    await step(4);
    await a.to("qsum", { txt: "18 + 19 + 0 + 0 = 37   (int32, exact)" }, 0);
    await a.wait(700);
    await step(5);
    await a.to("qout", { txt: "37 × 0.01 × 0.1 = 0.037   (f32)" }, 0);
    await a.wait(500);
    await step(6);
    await a.pulse("qout", 1.1, 200);
  }
});
})();
