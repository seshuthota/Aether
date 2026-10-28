// Acts V–VII: many workers (GPU), built for tensors (NPU), one op three ways.
(function () {
"use strict";
const { A, X, Y, box, txt, line, arrow, frame, path, code, cx, cy } = M;
const range = n => [...Array(n).keys()];
const partial = (r, c) => A[r].slice(0, c + 1).reduce((s, v, i) => s + v * X[i], 0);

// ---------------------------------------------------------------- GPU

M.scene("gpu-rows", {
  build: () => [
    M.matrix(),
    X.map((v, c) => box(`x${c}`, cx(c), 100, v, "c-x", { w: 38, h: 38 })),
    txt("lx", 104, 105, "x", { anchor: "end", cls: "lbl x b", fs: 14 }),
    txt("lx2", cx(3.5), 70, "every worker reads all of x", { cls: "faint", fs: 12 }),
    range(4).map(r => [
      arrow(`wa${r}`, 458, cy(r), 496, cy(r), { cls: "faint" }),
      box(`wk${r}`, 530, cy(r), "", "c-p", { w: 60, h: 34, rx: 17, fs: 15 }),
      arrow(`wb${r}`, 562, cy(r), 600, cy(r), { cls: "faint" }),
      box(`y${r}`, 630, cy(r), "?", "c-r", { w: 44, h: 38 })
    ]),
    txt("lwk", 530, 136, "workers", { cls: "lbl p b", fs: 13 }),
    txt("cap", 400, 430, "", { cls: "cap", fs: 16 })
  ],
  async play(a) {
    await a.wait(500);
    for (let c = 0; c < 8; c++) {
      a.pulse(`x${c}`, 1.15, 130);
      for (let r = 0; r < 4; r++) {
        a.pulse(`a${r}_${c}`, 1.18, 130);
        a.to(`wk${r}`, { txt: partial(r, c) }, 0);
      }
      await a.wait(380);
    }
    for (let r = 0; r < 4; r++) { a.to(`y${r}`, { txt: Y[r] }, 0); a.pulse(`y${r}`, 1.2, 150); }
    await a.to("cap", { txt: "Four workers, four rows, at the same time. Each still does its 8 steps in order." }, 0);
  }
});

const GRID = { cols: 76, rows: 64, gap: 7, x: 236, y: 64 };
M.scene("gpu-many", {
  build: () => [
    range(4).map(r => box(`wk${r}`, GRID.x + GRID.gap * r, GRID.y, "", "c-p", { w: 5, h: 5, rx: 2.5, fs: 1 })),
    path("grid", M.dotsD(GRID.cols, GRID.rows, GRID.gap, 2.3), { x: GRID.x, y: GRID.y, cls: "dots" }),
    txt("g1", 40, 96, "Qwen2.5-0.5B,\nMLP up projection:\n4864 rows × 896.", { anchor: "start", cls: "lbl b", fs: 14 }),
    txt("g2", 40, 200, "4864 dot products,\neach 896 long,\nnone waiting for\nanother.", { anchor: "start", cls: "lbl", fs: 14 }),
    txt("g3", 40, 330, "One dot here =\none output =\none GPU thread.", { anchor: "start", cls: "lbl p b", fs: 14 }),
    txt("g4", 40, 450, "4864 dots drawn.", { anchor: "start", cls: "faint", fs: 12 })
  ],
  async play(a) {
    await a.wait(700);
    for (let i = 0; i < 3; i++) {
      await a.to("grid", { o: 0.35 }, 300);
      await a.to("grid", { o: 1 }, 300);
    }
  }
});

function groupLayout(S) {
  const tc = S <= 64 ? 8 : 16, tr = S / tc, G = 4864 / S;
  for (let g = 6; g >= 1.5; g -= 0.25) {
    const gw = tc * g + 6, gh = tr * g + 6;
    const cols = Math.floor(420 / (gw + 4));
    const rows = Math.ceil(G / cols);
    if (rows * (gh + 4) <= 380) return { g, gw, gh, cols, rows, tc, tr, G };
  }
}
M.scene("gpu-groups", {
  init: { S: 64 },
  controls: [{ key: "S", label: "Threads per workgroup", opts: [[32, "32"], [64, "64"], [128, "128"], [256, "256"]] }],
  build: st => {
    const L = groupLayout(st.S);
    let outl = "", dots = "";
    const r = L.g * 0.32;
    for (let k = 0; k < L.G; k++) {
      const gx = (k % L.cols) * (L.gw + 4), gy = Math.floor(k / L.cols) * (L.gh + 4);
      outl += `M${gx} ${gy}h${L.gw}v${L.gh}h${-L.gw}Z`;
      for (let j = 0; j < L.tr; j++) for (let i = 0; i < L.tc; i++) {
        const x = gx + 3 + L.g * (i + 0.5), y = gy + 3 + L.g * (j + 0.5);
        dots += `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
      }
    }
    return [
      frame("gpu", 30, 50, 455, 470, "GPU · conceptual", { cls: "hw" }),
      path(`wgo${st.S}`, outl, { x: 48, y: 82, cls: "wg" }),
      path(`wgd${st.S}`, dots, { x: 48, y: 82, cls: "dots" }),
      box("wg0", 48 + L.gw / 2, 82 + L.gh / 2, "", "hotbox", { w: L.gw + 4, h: L.gh + 4, rx: 3 }),
      txt("wgN", 258, 505, `${L.G} workgroups × ${st.S} threads = 4864 threads`, { cls: "lbl b", fs: 13 }),
      frame("inset", 505, 50, 275, 470, "workgroup 0, up close", { cls: "sub" }),
      range(8).map(i => [
        box(`th${i}`, 532 + 31 * i, 112, `t${i}`, "c-p", { w: 26, h: 26, rx: 13, fs: 9.5 }),
        arrow(`tha${i}`, 532 + 31 * i, 128, 532 + 31 * i, 160, { cls: "faint" }),
        box(`ty${i}`, 532 + 31 * i, 176, `y${i}`, "c-r", { w: 26, h: 26, fs: 9.5 })
      ]),
      txt("thm", 642, 220, `thread i computes y[i]  ·  ${st.S} threads here`, { cls: "faint", fs: 11.5 }),
      path("wave", "M520 248h244", { cls: "bracket" }),
      txt("i1", 520, 280, "Threads in a workgroup can share fast\non-chip memory and wait for each other.", { anchor: "start", cls: "lbl", fs: 12.5 }),
      txt("i2", 520, 345, "The GPU runs them in waves: groups of\nthreads that execute the same instruction\ntogether (SIMT). Like SIMD lanes, but\neach lane is written as its own thread.", { anchor: "start", cls: "lbl", fs: 12.5 }),
      txt("i3", 520, 455, "Wave size depends on the GPU and\nthe compiler.", { anchor: "start", cls: "faint", fs: 11.5 })
    ];
  },
  after(a, st) {
    a.read(`<p>4864 rows ÷ ${st.S} threads per workgroup = <strong>${4864 / st.S} workgroups</strong>. Same 4864 threads, cut into different-sized groups.</p>`);
  },
  async play(a) {
    await a.wait(300);
    await a.pulse("wg0", 1.15, 250);
    for (let i = 0; i < 8; i++) a.pulse(`th${i}`, 1.2, 160);
  }
});

const KERNEL = ["i = thread index", "acc = 0", "for j in 0..cols:", "    acc += A[i][j] * x[j]", "y[i] = acc"];
const TL = {
  tiny: { cpu: [["c-w", 60, "submit"], ["c-n", 140, "wait"], ["c-x", 20, ""]], gpu: [["c-w", 80, "start-up"], ["c-p", 10, ""], ["c-w", 50, "signal"]] },
  big: { cpu: [["c-w", 60, "submit"], ["c-n", 570, "wait"], ["c-x", 20, ""]], gpu: [["c-w", 80, "start-up"], ["c-p", 440, "compute"], ["c-w", 50, "signal"]] }
};
M.scene("gpu-kernel", {
  init: { size: "tiny" },
  controls: [{ key: "size", label: "Work", opts: [["tiny", "our 4 × 8"], ["big", "4864 × 896"]] }],
  build: st => [
    frame("cpu", 30, 70, 230, 170, "CPU", { cls: "hw" }),
    txt("c1", 46, 116, `1. record: run gemv\n    on ${st.size === "tiny" ? 4 : 4864} threads\n2. submit to the driver\n3. wait for the fence`, { anchor: "start", cls: "lbl", fs: 13 }),
    code("kc", 300, 108, KERNEL, -1, { w: 240, fs: 11.5, lh: 17, title: "the kernel: one copy per thread" }),
    frame("gpu", 570, 70, 210, 230, "GPU", { cls: "hw" }),
    st.size === "tiny"
      ? path("gd_t", M.dotsD(4, 1, 30, 7), { x: 630, y: 185, cls: "dots" })
      : path("gd_b", M.dotsD(16, 14, 11, 3), { x: 592, y: 112, cls: "dots" }),
    st.size === "big" ? txt("gd_n", 675, 290, "…4864 threads", { cls: "faint", fs: 11 }) : [],
    frame("mem", 30, 330, 750, 64, "shared DRAM: on a phone, CPU and GPU use the same memory chips", { cls: "sub" }),
    box("mA", 250, 372, "A", "c-a", { w: 70, h: 24, fs: 12 }),
    box("mx", 360, 372, "x", "c-x", { w: 70, h: 24, fs: 12 }),
    box("my", 470, 372, "y", "c-r", { w: 70, h: 24, fs: 12 }),
    txt("tc", 40, 444, "CPU", { anchor: "start", cls: "lbl b", fs: 12 }),
    txt("tg", 40, 484, "GPU", { anchor: "start", cls: "lbl b", fs: 12 }),
    line("tax", 90, 500, 770, 500, { cls: "faint" }),
    txt("tlg", 400, 528, "amber: overhead · teal: the actual arithmetic · illustration, not measured", { cls: "faint", fs: 11.5 })
  ],
  async play(a, st) {
    const t = TL[st.size];
    await a.wait(400);
    let x = 90, n = 0;
    const seg = async (row, [cls, w, label], y) => {
      const k = `s${row}${n++}`;
      a.put(box(k, x, y, label, cls, { w: 0, h: 22, fs: 10.5, rx: 3 }));
      const p = a.to(k, { w, x: x + w / 2 }, w * 4, true);
      x += w;
      return p;
    };
    // CPU submits
    a.put(box("cmd", 250, 180, "gemv", "c-w", { w: 52, h: 24, fs: 11 }));
    const p1 = seg("c", t.cpu[0], 440);
    await a.to("cmd", { x: 570, y: 180 }, 240, true);
    await p1;
    a.del("cmd", 100);
    // CPU waits while the GPU starts, computes, signals
    const xw = x;
    a.put(box("cwait", xw, 440, "wait", "c-n", { w: 0, h: 22, fs: 10.5, rx: 3 }));
    const wTot = t.gpu.reduce((s, g) => s + g[1], 0);
    a.to("cwait", { w: wTot, x: xw + wTot / 2 }, wTot * 4, true);
    await seg("g", t.gpu[0], 480);
    M.codeTo(a, "kc", 300, 108, 3, { fs: 11.5, lh: 17 });
    const pulses = a.all([a.to(st.size === "tiny" ? "gd_t" : "gd_b", { o: 0.3 }, 120), a.wait(100)]);
    await pulses;
    a.to(st.size === "tiny" ? "gd_t" : "gd_b", { o: 1 }, 200);
    await seg("g", t.gpu[1], 480);
    M.codeTo(a, "kc", 300, 108, 4, { fs: 11.5, lh: 17 });
    a.put(box("res", 675, 300, "y", "c-r", { w: 30, h: 22, fs: 11 }));
    a.to("res", { x: 470, y: 372 }, 400);
    await seg("g", t.gpu[2], 480);
    a.del("res", 100);
    a.put(box("fence", 570, 220, "done", "c-w", { w: 52, h: 24, fs: 11 }));
    await a.to("fence", { x: 250, y: 220 }, 300);
    a.del("fence", 100);
    M.codeTo(a, "kc", 300, 108, -1, { fs: 11.5, lh: 17 });
    x = xw + wTot;
    await seg("c", t.cpu[2], 440);
    a.read(st.size === "tiny"
      ? "<p>32 multiply-adds: the GPU is busy for a sliver of the time. Starting it and learning it finished is most of the cost.</p>"
      : "<p>4.4 million multiply-adds: the arithmetic dominates, and the start-up cost is paid once.</p>");
  }
});

const CPUF = ["load 4 weights → v1", "load 4 inputs → v2", "fma: 4 lanes at once", "next chunk of the row", "add lanes → y[i]", "next row"];
const GPUF = ["CPU records and submits", "workgroups scheduled", "each thread loads its row", "kernel: acc += A·x", "y written to memory", "fence → CPU reads y"];
const FACT = ["workload size", "parallelism", "memory movement", "precision", "dispatch overhead", "how busy the hardware is"];
M.scene("cpu-vs-gpu", {
  build: () => [
    txt("hc", 210, 66, "CPU + NEON", { cls: "lbl b", fs: 15 }),
    txt("hg", 590, 66, "GPU", { cls: "lbl b", fs: 15 }),
    CPUF.map((s, k) => box(`cf${k}`, 210, 100 + 50 * k, s, "c-n", { w: 250, h: 34, fs: 13 })),
    GPUF.map((s, k) => box(`gf${k}`, 590, 100 + 50 * k, s, "c-n", { w: 250, h: 34, fs: 13 })),
    path("loop1", "M85 250C55 250 55 100 85 100", { cls: "loop" }),
    path("loop2", "M335 350C375 350 375 100 335 100", { cls: "loop" }),
    txt("hl", 400, 420, "Which one wins depends on:", { cls: "faint b", fs: 12 }),
    FACT.map((f, i) => box(`fc${i}`, 140 + 260 * (i % 3), 452 + 36 * Math.floor(i / 3), f, "c-w", { w: 220, h: 28, fs: 12.5, rx: 14 })),
    txt("cap", 400, 538, "For decode, both read every weight from the same DRAM, so both meet the same bandwidth limit.", { cls: "lbl", fs: 12.5 })
  ],
  async play(a) {
    await a.wait(400);
    for (let rep = 0; rep < 2; rep++) {
      for (let k = 0; k < 6; k++) {
        await a.all([a.to(`cf${k}`, { cls: "c-p" }, 0), a.to(`gf${k}`, { cls: "c-p" }, 0), a.wait(420)]);
        a.to(`cf${k}`, { cls: "c-n" }, 0);
        a.to(`gf${k}`, { cls: "c-n" }, 0);
      }
    }
  }
});

// ---------------------------------------------------------------- NPU

const PX = i => 20 + 156 * i;
M.scene("nested", {
  build: () => {
    const sq = (k, x, y, s, cls = "c-p") => box(k, x, y, "", cls, { w: s, h: s, rx: 2 });
    const panel = (i, title, c1, c2, o) => [
      frame(`pf${i}`, PX(i), 150, 140, 180, "", { cls: "sub", o }),
      txt(`pt${i}`, PX(i) + 70, 136, title, { cls: "lbl b", fs: 13, o }),
      txt(`pc${i}`, PX(i) + 70, 356, c1, { cls: "mono b", fs: 12.5, o }),
      txt(`pd${i}`, PX(i) + 70, 375, c2, { cls: "faint", fs: 11.5, o })
    ];
    return [
      panel(0, "multiply-add", "1", "", 1),
      box("mac", 90, 240, "×+", "c-p", { w: 70, h: 70, fs: 26 }),
      panel(1, "dot product", "896", "one Qwen row", 0),
      range(8).map(i => ({ ...sq(`d${i}`, 192 + 15 * i, 240, 12), o: 0 })),
      panel(2, "matrix × vector", "≈ 4.4 M", "4864 × 896 (MLP up)", 0),
      path("mgrid", M.dotsD(14, 11, 8, 2.6), { x: 350, y: 196, cls: "dots", o: 0 }),
      box("mrow", 402, 196, "", "hotbox", { w: 114, h: 9, rx: 2, o: 0 }),
      panel(3, "one layer", "≈ 14.9 M", "7 matrices", 0),
      ["q", "k", "v", "o"].map((n, i) => ({ ...box(`lq${i}`, 512 + 32 * i, 210, n, i ? "c-a" : "hotbox", { w: 28, h: 26, fs: 11 }), o: 0 })),
      ["gate", "up", "down"].map((n, i) => ({ ...box(`lm${i}`, 518 + 40 * i, 262, n, "c-a", { w: 36, h: 26, fs: 10 }), o: 0 })),
      panel(4, "the model", "≈ 494 M", "24 layers + output", 0),
      path("bars", range(24).map(i => `M0 ${i * 7}h108v5h-108Z`).join(""), { x: 660, y: 156, cls: "barsP", o: 0 }),
      box("bar0", 714, 158.5, "", "hotbox", { w: 112, h: 8, rx: 2, o: 0 }),
      line("z1a", 186, 234, PX(0) + 140, 150, { cls: "zoom", o: 0 }), line("z1b", 186, 246, PX(0) + 140, 330, { cls: "zoom", o: 0 }),
      line("z2a", 345, 192, PX(1) + 140, 150, { cls: "zoom", o: 0 }), line("z2b", 345, 200, PX(1) + 140, 330, { cls: "zoom", o: 0 }),
      line("z3a", 498, 197, PX(2) + 140, 150, { cls: "zoom", o: 0 }), line("z3b", 498, 223, PX(2) + 140, 330, { cls: "zoom", o: 0 }),
      line("z4a", 658, 155, PX(3) + 140, 150, { cls: "zoom", o: 0 }), line("z4b", 658, 162, PX(3) + 140, 330, { cls: "zoom", o: 0 }),
      txt("nc", 400, 445, "", { cls: "cap", fs: 16 }),
      txt("nd", 400, 472, "", { cls: "lbl", fs: 13.5 }),
      txt("ne", 400, 525, "Counts per token, calculated from Qwen2.5-0.5B's shapes.", { cls: "faint", fs: 11.5 })
    ];
  },
  async play(a) {
    const show = keys => a.all(keys.map(k => a.to(k, { o: 1 }, 450)));
    await a.wait(500);
    await show(["z1a", "z1b", "pf1", "pt1", "pc1", "pd1", ...range(8).map(i => `d${i}`)]);
    await a.wait(700);
    await show(["z2a", "z2b", "pf2", "pt2", "pc2", "pd2", "mgrid", "mrow"]);
    await a.wait(700);
    await show(["z3a", "z3b", "pf3", "pt3", "pc3", "pd3", ...range(4).map(i => `lq${i}`), ...range(3).map(i => `lm${i}`)]);
    await a.wait(700);
    await show(["z4a", "z4b", "pf4", "pt4", "pc4", "pd4", "bars", "bar0"]);
    await a.wait(500);
    await a.to("nc", { txt: "The same few shapes, repeated hundreds of millions of times per token," }, 0);
    await a.to("nd", { txt: "in an order fixed before the model runs. An NPU is built around that." }, 0);
  }
});

const NG = { x: c => 362 + 42 * c, y: r => 252 + 36 * r, xy: 208, yx: 730 };
M.scene("npu", {
  build: () => [
    frame("cpu", 30, 60, 220, 130, "CPU", { cls: "hw" }),
    txt("cpuT", 46, 104, "NPU runtime: load the\ncompiled graph once,\nthen ask it to run.", { anchor: "start", cls: "lbl", fs: 12.5 }),
    frame("dram", 30, 220, 220, 280, "DRAM", { cls: "hw" }),
    range(4).map(r => range(8).map(c => box(`a${r}_${c}`, 56 + 24 * c, 272 + 24 * r, A[r][c], "c-a", { w: 21, h: 21, fs: 10.5, rx: 3 }))),
    X.map((v, c) => box(`x${c}`, 56 + 24 * c, 392, v, "c-x", { w: 21, h: 21, fs: 10.5, rx: 3 })),
    range(4).map(r => box(`y${r}`, 92 + 30 * r, 452, "", "ghost", { w: 26, h: 22, fs: 11, rx: 3 })),
    txt("dy", 56, 457, "y", { anchor: "middle", cls: "lbl r b", fs: 12 }),
    frame("npu", 290, 60, 490, 440, "NPU · conceptual model", { cls: "hw" }),
    box("scal", 390, 116, "scalar (control)", "unit", { w: 160, h: 40, fs: 12.5 }),
    box("vec", 575, 116, "vector unit", "unit", { w: 160, h: 40, fs: 12.5 }),
    frame("tens", 305, 152, 460, 248, "tensor unit", { cls: "sub" }),
    frame("lmem", 305, 420, 460, 66, "on-chip memory, shared by the units", { cls: "sub" }),
    txt("illus", 535, 520, "Conceptual model: actual implementation varies by processor.", { cls: "warn", fs: 11.5 })
  ],
  async play(a) {
    await a.wait(400);
    a.put(box("run", 250, 125, "run", "c-w", { w: 44, h: 24, fs: 11 }));
    await a.to("run", { x: 310, y: 116 }, 500);
    a.del("run", 100);
    await a.pulse("scal", 1.08, 180);
    // DMA: the whole tile moves at once into on-chip memory
    await a.add(arrow("dma", 250, 360, 305, 450, { cls: "c-w" }), 200);
    const ps = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) ps.push(a.fly(`n${r}_${c}`, `a${r}_${c}`, 330 + 20 * c, 436 + 12 * r, 800, { w: 18, h: 10, fs: 0.1 }));
    X.forEach((_, c) => ps.push(a.fly(`nx${c}`, `x${c}`, 520 + 20 * c, 453, 800, { w: 18, h: 14, fs: 0.1 })));
    await a.all(ps);
    await a.wait(300);
    // into the tensor unit
    const ps2 = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) ps2.push(a.to(`n${r}_${c}`, { x: NG.x(c), y: NG.y(r), w: 36, h: 30, fs: 14 }, 800));
    X.forEach((_, c) => ps2.push(a.to(`nx${c}`, { x: NG.x(c), y: NG.xy, w: 36, h: 28, fs: 14 }, 800)));
    await a.all(ps2);
    await a.wait(500);
    // all 32 multiplies in one step
    for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) {
      a.to(`n${r}_${c}`, { txt: A[r][c] * X[c], cls: "c-p" }, 0);
      a.pulse(`n${r}_${c}`, 1.12, 160);
    }
    await a.wait(800);
    for (let r = 0; r < 4; r++) {
      a.put(box(`ty${r}`, NG.yx, NG.y(r), Y[r], "c-r", { w: 44, h: 30, fs: 15, o: 0 }));
      a.to(`ty${r}`, { o: 1 }, 400);
    }
    await a.all(range(4).map(r => a.all(range(8).map(c => a.to(`n${r}_${c}`, { x: NG.yx, s: 0.4, o: 0 }, 600)))));
    await a.wait(400);
    // results back out
    await a.all(range(4).map(r => a.to(`ty${r}`, { x: 92 + 30 * r, y: 452, w: 26, h: 22, fs: 11 }, 900)));
    for (let r = 0; r < 4; r++) a.to(`y${r}`, { txt: Y[r], cls: "c-r" }, 0);
    a.put(box("done", 140, 220, "done", "c-w", { w: 50, h: 24, fs: 11 }));
    await a.to("done", { y: 175 }, 400);
    a.del("done", 300);
  }
});

const W3 = {
  cpu: ["general-purpose", "instructions", "registers", "SIMD / NEON lanes", "flexible: runs any code"],
  gpu: ["massive parallelism", "threads", "workgroups", "kernels", "high throughput"],
  npu: ["neural-network structure", "operators in a graph", "tensor / vector units", "planned data movement", "inference, efficiently"]
};
const W3X = { cpu: 140, gpu: 400, npu: 660 };
M.scene("three", {
  build: () => [
    Object.entries(W3).map(([k, steps]) => [
      txt(`h${k}`, W3X[k], 66, k.toUpperCase(), { cls: "lbl b", fs: 16 }),
      steps.map((s, i) => [
        box(`${k}${i}`, W3X[k], 100 + 60 * i, s, i === 4 ? "c-p" : "c-n", { w: 210, h: 34, fs: 13 }),
        i < 4 ? arrow(`${k}a${i}`, W3X[k], 118 + 60 * i, W3X[k], 142 + 60 * i, { cls: "faint" }) : []
      ])
    ]),
    txt("ov1", 270, 210, "both: many lanes,\none instruction", { cls: "faint", fs: 11 }),
    txt("ov2", 530, 210, "both: work planned\nahead, handed over", { cls: "faint", fs: 11 }),
    frame("base", 30, 420, 740, 70, "", { cls: "sub" }),
    txt("baseT", 400, 452, "All three: multiply-add units, fed from one shared memory.", { cls: "cap", fs: 16 }),
    txt("baseU", 400, 475, "They overlap. Each is tuned for a different pattern of work.", { cls: "lbl", fs: 13 })
  ],
  async play(a) {
    await a.wait(400);
    for (let i = 0; i < 5; i++) {
      await a.all(["cpu", "gpu", "npu"].map(k => a.pulse(`${k}${i}`, 1.06, 160)));
      await a.wait(120);
    }
  }
});

// ---------------------------------------------------------------- one op, three ways

const SM = { x: c => 46 + 26 * c, y: r => 104 + 26 * r, xy: 230 };
const yX = r => 420 + 64 * r;
M.scene("same", {
  init: { hw: "cpu" },
  controls: [{ key: "hw", label: "Run on", opts: [["cpu", "CPU + NEON"], ["gpu", "GPU"], ["npu", "NPU"]] }],
  build: st => {
    const out = [
      txt("lA", SM.x(3.5), 78, "A", { cls: "lbl a b", fs: 13 }),
      range(4).map(r => range(8).map(c => box(`a${r}_${c}`, SM.x(c), SM.y(r), A[r][c], "c-a", { w: 23, h: 23, fs: 11, rx: 3 }))),
      txt("lx", SM.x(3.5), 212, "x", { cls: "lbl x b", fs: 13 }),
      X.map((v, c) => box(`x${c}`, SM.x(c), SM.xy, v, "c-x", { w: 23, h: 23, fs: 11, rx: 3 })),
      txt("ly", 370, 474, "y =", { anchor: "end", cls: "lbl r b", fs: 15 }),
      range(4).map(r => box(`y${r}`, yX(r), 469, "", "ghost", { w: 50, h: 36, fs: 17 })),
      txt("chk", 400, 520, "", { cls: "lbl p b", fs: 14 })
    ];
    if (st.hw === "cpu") out.push(
      frame("hw", 300, 60, 470, 360, "one CPU core · NEON", { cls: "hw" }),
      ["v1", "v2", "v0"].map((n, i) => [
        txt(`rn${i}`, 384, [150, 210, 300][i] + 5, n, { anchor: "end", cls: "lbl", fs: 12 }),
        range(4).map(j => box(`c${i}_${j}`, 420 + 50 * j, [150, 210, 300][i], i === 2 ? 0 : "", i === 2 ? "c-r" : "reg", { w: 44, h: 36, fs: 16 }))
      ]),
      box("csum", 690, 300, "", "ghost", { w: 56, h: 36, fs: 16, sub: "row sum" }),
      txt("crow", 535, 380, "", { cls: "mono", fs: 13 })
    );
    if (st.hw === "gpu") out.push(
      frame("cpuB", 300, 60, 130, 80, "CPU", { cls: "hw" }),
      frame("hw", 460, 60, 310, 360, "GPU", { cls: "hw" }),
      range(4).map(r => box(`th${r}`, 500, 130 + 74 * r, "", "c-p", { w: 50, h: 34, rx: 17, fs: 14, sub: `thread ${r}` }))
    );
    if (st.hw === "npu") out.push(
      frame("cpuB", 300, 60, 130, 80, "CPU · runtime", { cls: "hw" }),
      frame("hw", 460, 60, 310, 360, "NPU tensor unit · conceptual", { cls: "hw" })
    );
    return out;
  },
  async play(a, st) {
    await a.wait(400);
    for (let r = 0; r < 4; r++) a.to(`y${r}`, { txt: "", cls: "ghost" }, 0);
    if (st.hw === "cpu") {
      for (let r = 0; r < 4; r++) {
        a.to("crow", { txt: `row ${r}: 2 chunks of 4, then add the lanes` }, 0);
        const acc = [0, 0, 0, 0];
        for (const j of [0, 1, 2, 3]) a.to(`c2_${j}`, { txt: 0 }, 0);
        for (let ch = 0; ch < 2; ch++) {
          const k = r ? 0.5 : 1;
          await a.all(range(4).map(j => a.fly(`c0_${j}`, `a${r}_${4 * ch + j}`, 420 + 50 * j, 150, 450 * k, { w: 44, h: 36, fs: 16 })));
          await a.all(range(4).map(j => a.fly(`c1_${j}`, `x${4 * ch + j}`, 420 + 50 * j, 210, 450 * k, { w: 44, h: 36, fs: 16 })));
          for (let j = 0; j < 4; j++) {
            acc[j] += A[r][4 * ch + j] * X[4 * ch + j];
            a.to(`c2_${j}`, { txt: acc[j] }, 0);
            a.pulse(`c2_${j}`, 1.12, 120 * k);
          }
          await a.wait(350 * k);
        }
        await a.to("csum", { txt: Y[r], cls: "c-r" }, 0);
        await a.fly("ycp", "csum", yX(r), 469, 500, { w: 50, h: 36, fs: 17 });
        a.to(`y${r}`, { txt: Y[r], cls: "c-r" }, 0);
        a.del("ycp", 0);
      }
    }
    if (st.hw === "gpu") {
      a.put(box("cmd", 430, 100, "dispatch", "c-w", { w: 70, h: 24, fs: 11 }));
      await a.to("cmd", { x: 520, y: 80 }, 600);
      a.del("cmd", 150);
      await a.all(range(4).map(r => a.all(range(8).map(c =>
        a.fly(`g${r}_${c}`, `a${r}_${c}`, 556 + 26 * c, 130 + 74 * r, 600, { w: 23, h: 23, fs: 11 })))));
      for (let c = 0; c < 8; c++) {
        for (let r = 0; r < 4; r++) {
          a.pulse(`g${r}_${c}`, 1.25, 120);
          a.to(`th${r}`, { txt: partial(r, c) }, 0);
        }
        await a.wait(300);
      }
      await a.all(range(4).map(r => a.fly(`gy${r}`, `th${r}`, yX(r), 469, 600, { w: 50, h: 36, rx: 6, fs: 17 })));
      for (let r = 0; r < 4; r++) { a.to(`y${r}`, { txt: Y[r], cls: "c-r" }, 0); a.del(`gy${r}`, 0); }
      a.put(box("fence", 520, 80, "done", "c-w", { w: 52, h: 24, fs: 11 }));
      await a.to("fence", { x: 400, y: 100 }, 500);
      a.del("fence", 150);
    }
    if (st.hw === "npu") {
      a.put(box("cmd", 430, 100, "run graph", "c-w", { w: 80, h: 24, fs: 11 }));
      await a.to("cmd", { x: 530, y: 80 }, 600);
      a.del("cmd", 150);
      await a.all(range(4).map(r => a.all(range(8).map(c =>
        a.fly(`n${r}_${c}`, `a${r}_${c}`, 500 + 30 * c, 190 + 44 * r, 800, { w: 27, h: 27, fs: 12 })))));
      await a.all(range(8).map(c => a.fly(`nx${c}`, `x${c}`, 500 + 30 * c, 140, 800, { w: 27, h: 27, fs: 12 })));
      await a.wait(300);
      for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) {
        a.to(`n${r}_${c}`, { txt: A[r][c] * X[c], cls: "c-p" }, 0);
        a.pulse(`n${r}_${c}`, 1.15, 160);
      }
      await a.wait(800);
      await a.all(range(4).map(r => a.all(range(8).map(c => a.to(`n${r}_${c}`, { x: yX(r), y: 469, o: 0, s: 0.5 }, 700)))));
      for (let r = 0; r < 4; r++) a.to(`y${r}`, { txt: Y[r], cls: "c-r" }, 0);
    }
    await a.to("chk", { txt: "y = [37, 34, 22, 36] on every processor. Only the route changed." }, 0);
  }
});
})();
