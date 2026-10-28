// Acts I–III: the arithmetic, one core, SIMD lanes.
(function () {
"use strict";
const { A, X, Y, P0, box, txt, line, arrow, frame, code } = M;
const range = n => [...Array(n).keys()];

// The matrix's home position: cell centres.
const cx = c => 133 + 42 * c;
const cy = r => 171 + 42 * r;
M.cx = cx;
M.cy = cy;

function matrix(focus = null) {
  const out = [];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++)
    out.push(box(`a${r}_${c}`, cx(c), cy(r), A[r][c], "c-a", { w: 38, h: 38, o: focus == null || focus === r ? 1 : 0.22 }));
  return out;
}
M.matrix = matrix;
const xColumn = () => X.map((v, i) => box(`x${i}`, 520, 87 + 42 * i, v, "c-x", { w: 38, h: 38 }));
const yColumn = (vals, x = 616, y0 = null, gap = 42) =>
  vals.map((v, r) => box(`y${r}`, x, y0 == null ? cy(r) : y0 + gap * r, v, "c-r", { w: 44, h: 38 }));

// ---------------------------------------------------------------- Act I

M.scene("problem", {
  build: () => [
    matrix(),
    xColumn(),
    yColumn(["?", "?", "?", "?"]),
    txt("sign1", 474, 242, "×", { fs: 30 }),
    txt("sign2", 568, 242, "=", { fs: 30 }),
    txt("lA", 280, 136, "A · 4 × 8 weights", { cls: "lbl a b", fs: 14 }),
    txt("lx", 520, 52, "x · 8 inputs", { cls: "lbl x b", fs: 14 }),
    txt("ly", 616, 136, "y · 4 outputs", { cls: "lbl r b", fs: 14 }),
    txt("cap", 400, 470, "Four numbers to find. Each needs eight multiplications.", { cls: "cap", fs: 17 })
  ]
});

const rowScene = (withProducts, extra = []) => [
  A[0].map((v, c) => box(`a0_${c}`, cx(c), 110, v, "c-a", { w: 38, h: 38 })),
  X.map((v, c) => box(`x${c}`, cx(c), 172, v, "c-x", { w: 38, h: 38 })),
  range(8).map(c => txt(`m${c}`, cx(c), 146, "×", { fs: 13, cls: "faint" })),
  [1, 2, 3].map(r => A[r].map((v, c) => box(`a${r}_${c}`, cx(c), 498 + 24 * (r - 1), v, "c-a", { w: 22, h: 22, fs: 11, o: 0.2 }))),
  txt("lrow", 104, 115, "A row 0", { anchor: "end", cls: "lbl a b", fs: 13 }),
  txt("lxr", 104, 177, "x", { anchor: "end", cls: "lbl x b", fs: 13 }),
  txt("lrest", 104, 527, "rows 1–3", { anchor: "end", cls: "faint", fs: 12 }),
  yColumn(["?", "?", "?", "?"], 712, 110, 46),
  txt("ly", 712, 80, "y", { cls: "lbl r b", fs: 14 }),
  withProducts ? P0.map((v, c) => box(`p${c}`, cx(c), 252, v, "c-p", { w: 38, h: 38 })) : [],
  withProducts ? txt("lp", 104, 257, "products", { anchor: "end", cls: "lbl p b", fs: 13 }) : [],
  extra
];

M.scene("products", {
  build: () => rowScene(false),
  async play(a) {
    await a.wait(300);
    for (let c = 0; c < 8; c++) {
      const fast = c > 1;
      await a.all([a.to(`a0_${c}`, { s: 1.15 }, 150), a.to(`x${c}`, { s: 1.15 }, 150)]);
      a.put(box(`p${c}`, cx(c), 172, `${A[0][c]}×${X[c]}`, "c-p", { fs: 12, w: 38, h: 38, o: 0 }));
      await a.to(`p${c}`, { y: 252, o: 1 }, fast ? 320 : 600);
      await a.to(`p${c}`, { txt: P0[c], fs: 19 }, 0);
      a.to(`a0_${c}`, { s: 1 }, 150);
      a.to(`x${c}`, { s: 1 }, 150);
      await a.wait(fast ? 90 : 350);
    }
    await a.add(txt("lp", 104, 257, "products", { anchor: "end", cls: "lbl p b", fs: 13 }));
  }
});

M.scene("accumulate", {
  build: () => rowScene(true, [
    box("acc", 290, 350, 0, "c-r", { w: 90, h: 46, fs: 22, sub: "running sum" }),
    txt("eq", 290, 410, "", { cls: "mono", fs: 15 })
  ]),
  async play(a) {
    await a.wait(300);
    let s = 0;
    for (let c = 0; c < 8; c++) {
      const fast = c > 1;
      await a.to(`p${c}`, { x: 290, y: 350, s: 0.6 }, fast ? 300 : 550);
      a.to(`p${c}`, { o: 0 }, 120);
      a.to("eq", { txt: `${s} + ${P0[c]} = ${s + P0[c]}` }, 0);
      s += P0[c];
      await a.to("acc", { txt: s }, 0);
      await a.pulse("acc", 1.1, 110);
      await a.wait(fast ? 60 : 300);
    }
    await a.wait(300);
    await a.fly("acc2", "acc", 712, 110, 750, { sub: "", w: 44, h: 38 });
    await a.to("y0", { txt: 37 }, 0);
    await a.del("acc2", 0);
    await a.pulse("y0", 1.2, 160);
    a.del("lp", 300);
    await a.add(txt("dp", 400, 470, "Multiply the pairs, add the products: a dot product.", { cls: "cap", fs: 17 }));
  }
});

M.scene("allrows", {
  build: () => [
    matrix(),
    xColumn(),
    yColumn([37, "?", "?", "?"]),
    txt("sign1", 474, 242, "×", { fs: 30 }),
    txt("sign2", 568, 242, "=", { fs: 30 }),
    txt("ly", 616, 136, "y", { cls: "lbl r b", fs: 14 }),
    txt("macs", 400, 450, "multiply-adds so far: 8", { cls: "mono", fs: 15 })
  ],
  async play(a) {
    await a.wait(300);
    let n = 8;
    for (let r = 1; r < 4; r++) {
      await a.all(range(4).map(q => {
        const ps = [];
        for (let c = 0; c < 8; c++) ps.push(a.to(`a${q}_${c}`, { o: q === r ? 1 : 0.22 }, 250));
        return a.all(ps);
      }));
      for (let c = 0; c < 8; c++) {
        a.pulse(`x${c}`, 1.15, 70);
        await a.pulse(`a${r}_${c}`, 1.18, 70);
        n++;
        a.to("macs", { txt: `multiply-adds so far: ${n}` }, 0);
      }
      await a.to(`y${r}`, { txt: Y[r] }, 0);
      await a.pulse(`y${r}`, 1.2, 140);
    }
    for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) a.to(`a${r}_${c}`, { o: 1 }, 300);
    await a.wait(400);
    await a.to("macs", { txt: "32 multiply-adds. That is all a matrix × vector is." }, 0);
    await a.add(txt("scale1", 400, 490, "stories15M: 15.2 million of them per token.  Qwen2.5-0.5B: about 494 million.", { cls: "faint", fs: 13 }));
  }
});

// ---------------------------------------------------------------- Act II

const hex = n => "0x" + n.toString(16).toUpperCase();
const MEM = { ay: i => 106 + 24 * i, xy: i => 346 + 24 * i, vx: 252 };

function memory(focus = -1) {
  return [
    frame("mem", 40, 50, 280, 470, "memory (DRAM)", { cls: "hw" }),
    txt("memA", 62, 90, "A, row 0 (weights)", { anchor: "start", cls: "lbl a b", fs: 12 }),
    txt("memR", 62, 306, "…rows 1–3 follow", { anchor: "start", cls: "faint", fs: 12 }),
    txt("memX", 62, 330, "x (activations)", { anchor: "start", cls: "lbl x b", fs: 12 }),
    range(8).map(i => [
      txt(`adA${i}`, 62, MEM.ay(i) + 4, hex(0x1000 + 4 * i), { anchor: "start", cls: "mono faint", fs: 12.5 }),
      box(`a0_${i}`, MEM.vx, MEM.ay(i), A[0][i], "c-a", { w: 64, h: 21, fs: 14, rx: 4, s: i === focus ? 1.12 : 1 }),
      txt(`adX${i}`, 62, MEM.xy(i) + 4, hex(0x2000 + 4 * i), { anchor: "start", cls: "mono faint", fs: 12.5 }),
      box(`x${i}`, MEM.vx, MEM.xy(i), X[i], "c-x", { w: 64, h: 21, fs: 14, rx: 4, s: i === focus ? 1.12 : 1 })
    ])
  ];
}

M.scene("memory", {
  build: () => [
    memory(),
    txt("b1", 560, 100, "The weight 2, stored as an f32:", { cls: "lbl", fs: 14 }),
    box("big2", 560, 150, "2.0", "c-a", { w: 70, h: 46, fs: 20 }),
    arrow("b2", 560, 178, 560, 222, { cls: "faint" }),
    ["00", "00", "00", "40"].map((b, i) =>
      box(`byte${i}`, 470 + 60 * i, 250, b, "c-n", { w: 52, h: 32, fs: 15, sub: hex(0x1000 + i), subBelow: true })),
    txt("b3", 560, 330, "32 bits = 4 bytes, so the next number", { cls: "lbl", fs: 13.5 }),
    txt("b4", 560, 350, "starts 4 addresses later.", { cls: "lbl", fs: 13.5 }),
    txt("b5", 560, 395, "(2.0 is 0x40000000 in f32; the phone stores", { cls: "faint", fs: 12 }),
    txt("b6", 560, 412, "the lowest byte first.)", { cls: "faint", fs: 12 })
  ]
});

// One CPU core: registers on the left, two arithmetic units on the right.
const REG = { x: 478, r1: 132, r2: 192, r3: 252, r0: 312 };
const UNIT = { mx: 682, my: 152, ay: 282 };
function core(ins, insx, regs = {}) {
  const reg = (k, y, name, v, cls) =>
    box(k, REG.x, y, v == null ? "" : v, v == null ? "reg" : cls, { w: 110, h: 34, fs: 17, sub: name });
  return [
    frame("core", 350, 50, 420, 470, "one CPU core", { cls: "hw" }),
    frame("regs", 392, 66, 172, 278, "registers", { cls: "sub" }),
    reg("r1", REG.r1, "s1", regs.r1, "c-a"),
    reg("r2", REG.r2, "s2", regs.r2, "c-x"),
    reg("r3", REG.r3, "s3", regs.r3, "c-p"),
    box("r0", REG.x, REG.r0, regs.r0 == null ? 0 : regs.r0, "c-r", { w: 110, h: 34, fs: 17, sub: "s0 · accumulator" }),
    box("mul", UNIT.mx, UNIT.my, "×", "unit", { w: 100, h: 62, fs: 28, sub: "multiplier" }),
    box("add", UNIT.mx, UNIT.ay, "+", "unit", { w: 100, h: 62, fs: 28, sub: "adder" }),
    txt("ins", 372, 378, ins, { anchor: "start", cls: "mono b", fs: 14 }),
    txt("insx", 372, 400, insx, { anchor: "start", cls: "faint", fs: 12.5 })
  ];
}

// One multiply-add, instruction by instruction. k < 1 speeds it up.
async function mac(a, i, before, k = 1) {
  const t = ms => ms * k;
  const say = (s, x) => { a.to("ins", { txt: s }, 0); a.to("insx", { txt: x }, 0); };
  say("LDR  s1, [w + 4·i]", "load: copy w[i] from memory into register s1");
  await a.fly("tk", `a0_${i}`, REG.x, REG.r1, t(700), { w: 110, h: 34, fs: 17, s: 1 });
  await a.to("r1", { txt: A[0][i], cls: "c-a" }, 0);
  await a.del("tk", 0);
  say("LDR  s2, [x + 4·i]", "load: copy x[i] into register s2");
  await a.fly("tk", `x${i}`, REG.x, REG.r2, t(700), { w: 110, h: 34, fs: 17, s: 1 });
  await a.to("r2", { txt: X[i], cls: "c-x" }, 0);
  await a.del("tk", 0);
  say("FMUL s3, s1, s2", "multiply: s3 = s1 × s2");
  await a.all([
    a.fly("tk", "r1", UNIT.mx - 22, UNIT.my, t(450), { w: 36, h: 30, sub: "" }),
    a.fly("tk2", "r2", UNIT.mx + 22, UNIT.my, t(450), { w: 36, h: 30, sub: "" })
  ]);
  await a.all([a.del("tk", 0), a.del("tk2", 0), a.pulse("mul", 1.08, t(120))]);
  a.put(box("tk", UNIT.mx, UNIT.my, A[0][i] * X[i], "c-p", { w: 40, h: 30, fs: 17 }));
  await a.to("tk", { x: REG.x, y: REG.r3, w: 110, h: 34 }, t(450));
  await a.to("r3", { txt: A[0][i] * X[i], cls: "c-p" }, 0);
  await a.del("tk", 0);
  say("FADD s0, s0, s3", "add: accumulator = accumulator + product");
  await a.all([
    a.fly("tk", "r3", UNIT.mx - 22, UNIT.ay, t(450), { w: 36, h: 30, sub: "" }),
    a.fly("tk2", "r0", UNIT.mx + 22, UNIT.ay, t(450), { w: 36, h: 30, sub: "" })
  ]);
  await a.all([a.del("tk", 0), a.del("tk2", 0), a.pulse("add", 1.08, t(120))]);
  const sum = before + A[0][i] * X[i];
  a.put(box("tk", UNIT.mx, UNIT.ay, sum, "c-r", { w: 40, h: 30, fs: 17 }));
  await a.to("tk", { x: REG.x, y: REG.r0, w: 110, h: 34 }, t(450));
  await a.to("r0", { txt: sum }, 0);
  await a.del("tk", 0);
  return sum;
}

M.scene("load", {
  build: () => [memory(0), core("", "")],
  async play(a) {
    await a.wait(400);
    await mac(a, 0, 0, 1);
    a.to("ins", { txt: "4 instructions for one multiply-add." }, 0);
    a.to("insx", { txt: "Two loads, a multiply, an add." }, 0);
  }
});

const LOOP = ["for i in 0..8 {", "    acc += w[i] * x[i];", "}"];
M.scene("loop", {
  build: () => [
    memory(),
    core("", ""),
    code("lc", 372, 452, LOOP, 1, { w: 380, title: "the Rust loop" }),
    txt("iv", 730, 452, "i = 0", { anchor: "end", cls: "mono code", fs: 12.5 }),
    txt("cnt", 750, 352, "instructions: 0", { anchor: "end", cls: "mono", fs: 12.5 })
  ],
  async play(a) {
    await a.wait(300);
    let s = 0, n = 0;
    for (let i = 0; i < 8; i++) {
      a.to("iv", { txt: `i = ${i}` }, 0);
      a.pulse(`a0_${i}`, 1.15, 120);
      a.pulse(`x${i}`, 1.15, 120);
      s = await mac(a, i, s, i === 0 ? 0.8 : i < 3 ? 0.45 : 0.22);
      n += 4;
      a.to("cnt", { txt: `instructions: ${n}` }, 0);
    }
    a.to("ins", { txt: "s0 = 37 after 32 instructions," }, 0);
    a.to("insx", { txt: "plus the loop's own: add 1 to i, compare, branch." }, 0);
  }
});

const TILES = [["LDR", "load w[i]", 0], ["LDR", "load x[i]", 0], ["FMUL", "multiply", 1], ["FADD", "add to acc", 1],
  ["ADD", "i += 1", 2], ["CMP", "i < 8 ?", 2], ["B.NE", "loop back", 3]];
const UNITS = ["load / store", "FP / SIMD", "integer ALU", "branch"];
const tileX = j => 152 + 82 * j;
const unitX = u => 230 + 150 * u;
M.scene("layers", {
  build: () => [
    txt("L1", 40, 62, "MATH", { anchor: "start", cls: "faint b", fs: 11.5 }),
    txt("m1", 420, 92, "y[0] = A[0][0]·x[0] + A[0][1]·x[1] + … + A[0][7]·x[7]", { cls: "cap", fs: 17 }),
    arrow("ar1", 420, 108, 420, 140, { cls: "faint" }),
    txt("L2", 40, 152, "RUST", { anchor: "start", cls: "faint b", fs: 11.5 }),
    txt("m2", 420, 168, "for i in 0..8 { acc += w[i] * x[i]; }", { cls: "mono", fs: 15 }),
    arrow("ar2", 420, 184, 420, 214, { cls: "faint" }),
    txt("ar2l", 432, 204, "compiles to, per pass:", { anchor: "start", cls: "faint", fs: 11.5 }),
    txt("L3", 40, 232, "INSTRUCTIONS", { anchor: "start", cls: "faint b", fs: 11.5 }),
    TILES.map(([op, what], j) => box(`tl${j}`, tileX(j), 268, op, "c-n mono", { w: 72, h: 36, fs: 14, sub: what, subBelow: true })),
    txt("L4", 40, 352, "HARDWARE", { anchor: "start", cls: "faint b", fs: 11.5 }),
    frame("hwc", 140, 370, 620, 110, "core · conceptual: real cores have several of each unit", { cls: "hw" }),
    UNITS.map((u, i) => box(`un${i}`, unitX(i), 432, u, "unit", { w: 130, h: 46, fs: 13 })),
    txt("pass", 760, 232, "", { anchor: "end", cls: "mono", fs: 12.5 }),
    txt("meas", 400, 520, "Measured here (Phase 2, scalar loop): 1.64 instructions per cycle, about 1.77 cycles per multiply-add.", { cls: "faint", fs: 12 })
  ],
  async play(a) {
    await a.wait(400);
    for (let p = 0; p < 3; p++) {
      a.to("pass", { txt: `pass i = ${p}` }, 0);
      for (let j = 0; j < TILES.length; j++) {
        const u = TILES[j][2], k = p ? 0.45 : 1;
        await a.to(`tl${j}`, { cls: "c-p mono", s: 1.1 }, 140 * k);
        await a.add(arrow("wire", tileX(j), 300, unitX(u), 408, { cls: "c-p" }), 0);
        await a.pulse(`un${u}`, 1.06, 140 * k);
        await a.to(`tl${j}`, { cls: "c-n mono", s: 1 }, 120 * k);
        await a.del("wire", 0);
      }
    }
    a.to("pass", { txt: "… 8 passes" }, 0);
  }
});

// ---------------------------------------------------------------- Act III

M.scene("simd", {
  build: () => [
    frame("core", 30, 50, 740, 460, "one core, one clock", { cls: "hw" }),
    line("div", 400, 80, 400, 470, { cls: "faint dash" }),
    txt("tl", 215, 102, "scalar: one pair per instruction", { cls: "lbl b", fs: 14 }),
    txt("tr", 590, 102, "SIMD: one instruction, four pairs", { cls: "lbl p b", fs: 14 }),
    range(4).map(j => {
      const y = 160 + 72 * j;
      return [
        txt(`tk${j}`, 66, y + 5, `tick ${j + 1}`, { anchor: "start", cls: "faint mono", fs: 12 }),
        box(`a0_${j}`, 150, y, A[0][j], "c-a", { w: 40, h: 38 }),
        txt(`sx${j}`, 186, y + 6, "×", { fs: 16, cls: "faint" }),
        box(`x${j}`, 222, y, X[j], "c-x", { w: 40, h: 38 }),
        txt(`se${j}`, 262, y + 6, "=", { fs: 16, cls: "faint" }),
        box(`sp${j}`, 304, y, "", "ghost", { w: 40, h: 38 })
      ];
    }),
    frame("vr1", 482, 148, 212, 46, "", { cls: "reg" }),
    frame("vr2", 482, 232, 212, 46, "", { cls: "reg" }),
    frame("vr0", 482, 336, 212, 46, "", { cls: "reg" }),
    range(4).map(j => [
      box(`va${j}`, 513 + 50 * j, 171, A[0][j], "c-a", { w: 44, h: 38, from: `a0_${j}`, fromO: 0 }),
      box(`vx${j}`, 513 + 50 * j, 255, X[j], "c-x", { w: 44, h: 38, from: `x${j}`, fromO: 0 }),
      box(`vp${j}`, 513 + 50 * j, 359, "", "ghost", { w: 44, h: 38 })
    ]),
    txt("vm", 588, 218, "×", { fs: 18, cls: "faint" }),
    arrow("va", 588, 284, 588, 328, { cls: "faint" }),
    txt("vt", 588, 412, "", { cls: "faint mono", fs: 12 }),
    txt("clock", 400, 498, "", { cls: "mono b", fs: 14 })
  ],
  async play(a) {
    await a.wait(500);
    for (let j = 0; j < 4; j++) {
      a.to("clock", { txt: `tick ${j + 1}` }, 0);
      a.to(`tk${j}`, { cls: "mono b" }, 0);
      const ps = [a.to(`sp${j}`, { txt: P0[j], cls: "c-p" }, 0), a.pulse(`sp${j}`, 1.2, 160)];
      if (j === 0) {
        for (let q = 0; q < 4; q++) {
          ps.push(a.to(`vp${q}`, { txt: P0[q], cls: "c-p" }, 0));
          ps.push(a.pulse(`vp${q}`, 1.2, 160));
        }
        ps.push(a.to("vt", { txt: "done at tick 1" }, 0));
      }
      await a.all(ps);
      await a.wait(650);
    }
    a.to("clock", { txt: "Same core, same clock: four results per instruction instead of one." }, 0);
  }
});

// The NEON dot product, shared by the "neon" and "code" scenes.
const NEON_CODE = {
  1: { lines: ["let mut acc = 0.0f32;", "for i in 0..8 {", "    acc += weights[i] * acts[i]; // one pair", "}"], init: 0, load: 2, fma: 2, red: -1 },
  4: {
    lines: [
      "let mut acc = vdupq_n_f32(0.0);       // [0, 0, 0, 0]",
      "for c in (0..8).step_by(4) {",
      "    let w = vld1q_f32(&weights[c]);   // load 4 weights",
      "    let a = vld1q_f32(&acts[c]);      // load 4 inputs",
      "    acc = vfmaq_f32(acc, w, a);       // acc += w × a",
      "}",
      "let sum = vaddvq_f32(acc);            // add the 4 lanes"
    ], init: 0, load: 2, load2: 3, fma: 4, red: 6
  },
  8: {
    lines: [
      "// 8 f32 lanes need 256-bit registers (x86 AVX2).",
      "// NEON here is 128-bit: 8 lanes only for f16.",
      "acc = fma(acc, w[0..8], a[0..8]);  // one instruction",
      "sum = add_lanes(acc);"
    ], init: -1, load: 2, fma: 2, red: 3
  }
};
M.NEON_CODE = NEON_CODE;

// L: layout. memX(c), memWY, memXY, laneX(j), v1y, v2y, v0y, sumX, sumY, codeX, codeY
async function neonRun(a, W, L, slow = 1, P = "") {
  const C = NEON_CODE[W];
  const hl = line => (L.codeX != null ? M.codeTo(a, P + "nc", L.codeX, L.codeY, line, { lh: 18, fs: 12 }) : null);
  const acc = new Array(W).fill(0);
  let loads = 0, fmas = 0;
  const count = () => {
    a.to(P + "c_ld", { txt: `loads: ${loads}` }, 0);
    a.to(P + "c_fma", { txt: `multiply-add instructions: ${fmas}` }, 0);
  };
  hl(C.init);
  await a.wait(500 * slow);
  const chunks = 8 / W;
  for (let ch = 0; ch < chunks; ch++) {
    const k = (ch === 0 ? 1 : W === 1 ? 0.4 : 0.8) * slow;
    const c0 = ch * W;
    await a.add(frame(P + "chunk", L.memX(c0) - (L.step || 46) / 2 + 1, L.memWY - 25, (L.step || 46) * W - 2, L.memXY - L.memWY + 50, "", { cls: "hot", rx: 6 }), 200 * k);
    hl(C.load);
    await a.all(range(W).map(j => a.fly(`${P}l1_${j}`, `${P}a0_${c0 + j}`, L.laneX(j), L.v1y, 600 * k, { w: L.cw || 40, h: 34, fs: 16, s: 1 })));
    if (C.load2 != null) hl(C.load2);
    await a.all(range(W).map(j => a.fly(`${P}l2_${j}`, `${P}x${c0 + j}`, L.laneX(j), L.v2y, 600 * k, { w: L.cw || 40, h: 34, fs: 16, s: 1 })));
    loads += 2;
    count();
    hl(C.fma);
    await a.all(range(W).map(j => {
      const p = A[0][c0 + j] * X[c0 + j];
      a.put(box(`${P}pp${j}`, L.laneX(j), (L.v2y + L.v0y) / 2, `+${p}`, "c-p", { w: 36, h: 24, fs: 13, o: 0 }));
      return a.to(`${P}pp${j}`, { o: 1 }, 250 * k);
    }));
    await a.all(range(W).map(j => a.to(`${P}pp${j}`, { y: L.v0y, o: 0 }, 350 * k)));
    for (let j = 0; j < W; j++) {
      acc[j] += A[0][c0 + j] * X[c0 + j];
      a.to(`${P}l0_${j}`, { txt: acc[j] }, 0);
      a.pulse(`${P}l0_${j}`, 1.12, 120 * k);
    }
    fmas += 1;
    count();
    await a.wait(350 * k);
  }
  await a.del(P + "chunk", 200);
  const sum = acc.reduce((s, v) => s + v, 0);
  if (W > 1) {
    hl(C.red);
    await a.wait(300 * slow);
    await a.all(range(W).map(j => a.fly(`${P}rd${j}`, `${P}l0_${j}`, L.sumX, L.sumY, 700 * slow, { w: 30, h: 30 })));
    for (let j = 0; j < W; j++) a.del(`${P}rd${j}`, 0);
    a.to(P + "c_red", { txt: "add across lanes: 1" }, 0);
  } else {
    await a.fly(P + "rd0", P + "l0_0", L.sumX, L.sumY, 600 * slow, { w: 30, h: 30 });
    a.del(P + "rd0", 0);
  }
  await a.to(P + "s_out", { txt: sum, cls: "c-r" }, 0);
  await a.pulse(P + "s_out", 1.2, 160);
}
M.neonRun = neonRun;

const NL = {
  memX: c => 214 + 46 * c, memWY: 78, memXY: 122,
  laneX: j => 214 + 46 * j, v1y: 200, v2y: 250, v0y: 320,
  sumX: 690, sumY: 320, codeX: 52, codeY: 414
};
function neonLayout(W, L, regName, P = "", short = false) {
  const lane = (k, j, y, v, cls) => box(`${P}${k}_${j}`, L.laneX(j), y, v, cls, { w: L.cw || 40, h: 34, fs: 16 });
  const names = { 1: ["s1 · 32-bit", "s2", "s0 · acc"], 4: ["v1 · 128-bit NEON", "v2", "v0 · 4 accumulators"], 8: ["v1 · 256-bit", "v2", "v0 · 8 accumulators"] }[W];
  const rf = (k, y) => frame(P + k, L.laneX(0) - (L.step || 46) / 2, y - 21, (L.step || 46) * W, 42, "", { cls: "reg", rx: 6 });
  return [
    range(8).map(c => box(`${P}a0_${c}`, L.memX(c), L.memWY, A[0][c], "c-a", { w: L.cw || 40, h: 34, fs: 16 })),
    range(8).map(c => box(`${P}x${c}`, L.memX(c), L.memXY, X[c], "c-x", { w: L.cw || 40, h: 34, fs: 16 })),
    txt(P + "mw", L.memX(0) - 32, L.memWY + 5, short ? "w" : "weights in memory", { anchor: "end", cls: "lbl a", fs: 12 }),
    txt(P + "mx", L.memX(0) - 32, L.memXY + 5, short ? "x" : "activations", { anchor: "end", cls: "lbl x", fs: 12 }),
    rf("rf1", L.v1y), rf("rf2", L.v2y), rf("rf0", L.v0y),
    range(W).map(j => [lane("l1", j, L.v1y, "", "reg"), lane("l2", j, L.v2y, "", "reg"), lane("l0", j, L.v0y, 0, "c-r")]),
    W > 1 ? range(W).map(j => txt(`${P}ln${j}`, L.laneX(j), L.v1y - 28, `lane ${j}`, { cls: "faint", fs: 10 })) : [],
    txt(P + "n1", L.laneX(0) - 32, L.v1y + 5, regName ? names[0] : "", { anchor: "end", cls: "lbl", fs: 12 }),
    txt(P + "n2", L.laneX(0) - 32, L.v2y + 5, regName ? names[1] : "", { anchor: "end", cls: "lbl", fs: 12 }),
    txt(P + "n0", L.laneX(0) - 32, L.v0y + 5, regName ? names[2] : "", { anchor: "end", cls: "lbl r", fs: 12 }),
    txt(P + "vmul", L.laneX(W - 1) + 34, (L.v1y + L.v2y) / 2 + 5, "×", { anchor: "start", cls: "faint", fs: 16 }),
    box(P + "s_out", L.sumX, L.sumY, "?", "ghost", { w: 60, h: 40, fs: 20, sub: "sum" })
  ];
}
M.neonLayout = neonLayout;

M.scene("neon", {
  init: { W: 4 },
  controls: [{ key: "W", label: "Lanes", opts: [[1, "1 (scalar)"], [4, "4 (NEON f32)"], [8, "8"]] }],
  build: st => [
    neonLayout(st.W, NL, true),
    txt("c_ld", 566, 200, "loads: 0", { anchor: "start", cls: "mono", fs: 13 }),
    txt("c_fma", 566, 224, "multiply-add instructions: 0", { anchor: "start", cls: "mono", fs: 13 }),
    txt("c_red", 566, 248, "", { anchor: "start", cls: "mono", fs: 13 }),
    txt("c_per", 566, 272, `values per instruction: ${st.W}`, { anchor: "start", cls: "mono b", fs: 13 }),
    code("nc", NL.codeX, NL.codeY, NEON_CODE[st.W].lines, -1, { w: 540, fs: 12, lh: 18, title: st.W === 4 ? "Rust with NEON intrinsics (inside unsafe)" : st.W === 1 ? "Rust, scalar" : "not on this phone" })
  ],
  play: (a, st) => neonRun(a, st.W, NL)
});

const GEMV = {
  l2: { name: "matrix in L2: 288 × 288 (324 KB)", v: [5.3, 17.8, 41.6, 53.5], ceil: 67, ceilName: "L2 read, one prime core: ~67 GB/s" },
  dram: { name: "matrix in DRAM: 8192 × 8192 (256 MB)", v: [3.0, 9.3, 36.0, 40.6], ceil: 61, ceilName: "DRAM read, one prime core: ~61 GB/s" }
};
const GROWS = ["scalar loop", "NEON, 1 accumulator", "NEON, 4 accumulators", "NEON, 4 rows at once"];
const gx = v => 250 + v * 6.6;
const CHIPS = ["arithmetic", "memory loads", "cache behaviour", "dependency chains", "instruction throughput", "data layout", "loop overhead"];
M.scene("notmagic", {
  init: { where: "l2" },
  controls: [{ key: "where", label: "Matrix", opts: [["l2", "fits in L2"], ["dram", "in DRAM"]] }],
  build: st => {
    const g = GEMV[st.where];
    return [
      txt("claim", 400, 68, "“NEON = 4× faster”", { cls: "cap", fs: 24 }),
      txt("gt", 400, 108, `f32 GEMV, GB/s of weights read · ${g.name}`, { cls: "faint", fs: 12.5 }),
      GROWS.map((name, i) => [
        txt(`gl${i}`, 40, 155 + 50 * i, name, { anchor: "start", cls: "lbl", fs: 13 }),
        box(`gb${i}`, 250, 150 + 50 * i, "", i ? "c-p" : "c-n", { w: 0, h: 28, rx: 3 }),
        txt(`gv${i}`, 258, 155 + 50 * i, "", { anchor: "start", cls: "mono b", fs: 13 })
      ]),
      line("ceil", gx(g.ceil), 128, gx(g.ceil), 340, { cls: "warn dash" }),
      txt("ceilL", gx(g.ceil) - 4, 356, g.ceilName, { anchor: "end", cls: "warn", fs: 11.5 }),
      txt("src", 400, 540, "Measured on this phone, one prime core (results/phase-3.md, Phase 1).", { cls: "faint", fs: 11.5 })
    ];
  },
  async play(a, st) {
    const g = GEMV[st.where];
    await a.wait(300);
    for (let i = 0; i < 4; i++) {
      const w = gx(g.v[i]) - 250;
      a.to(`gb${i}`, { w, x: 250 + w / 2 }, 700);
      await a.wait(350);
      await a.to(`gv${i}`, { x: 258 + w, txt: `${g.v[i]}` }, 0);
    }
    await a.wait(300);
    await a.add(line("strike", 268, 62, 268, 62, { cls: "warn thick" }), 0);
    await a.to("strike", { x2: 532 }, 500);
    await a.add(txt("ratio", 400, 392, `1 accumulator: ×${(g.v[1] / g.v[0]).toFixed(1)}, not ×4. Every multiply-add waited for the one before it.`, { cls: "lbl", fs: 13 }));
    await a.add(txt("chipT", 400, 428, "What the speed depends on:", { cls: "faint b", fs: 12 }), 200);
    const rows = [CHIPS.slice(0, 4), CHIPS.slice(4)];
    let i = 0;
    for (let r = 0; r < 2; r++) {
      const widths = rows[r].map(c => c.length * 6.8 + 24);
      let x = 400 - (widths.reduce((s, w) => s + w + 8, -8)) / 2;
      for (let j = 0; j < rows[r].length; j++, i++) {
        a.add(box(`chip${i}`, x + widths[j] / 2, 458 + 36 * r, rows[r][j], "c-w", { w: widths[j], h: 28, fs: 12.5, rx: 14 }), 300);
        x += widths[j] + 8;
        await a.wait(110);
      }
    }
  }
});

// Gantt rows: [kind, units]. L = load, W = wait, F = multiply-add.
const ROWS = [
  { name: "data already in L1", seq: [["L", 0.25], ["F", 1]] },
  { name: "from DRAM, one request at a time", seq: [["L", 0.25], ["W", 6], ["F", 1]] },
  { name: "from DRAM, prefetcher fetching ahead", seq: [["W", 0.8], ["F", 1]] }
];
const GU = 29, G0 = 60, GEND = 760;
M.scene("waiting", {
  build: () => [
    ROWS.map((r, i) => [
      txt(`gn${i}`, 40, 110 + 120 * i, r.name, { anchor: "start", cls: "lbl b", fs: 13 }),
      line(`gax${i}`, G0, 150 + 120 * i, GEND, 150 + 120 * i, { cls: "faint" }),
      txt(`gc${i}`, GEND, 110 + 120 * i, "", { anchor: "end", cls: "mono", fs: 13 })
    ]),
    box("lgL", 70, 495, "", "c-x", { w: 18, h: 14, rx: 2 }), txt("lgLt", 86, 500, "load", { anchor: "start", cls: "faint", fs: 12 }),
    box("lgW", 150, 495, "", "c-w", { w: 18, h: 14, rx: 2 }), txt("lgWt", 166, 500, "waiting for memory", { anchor: "start", cls: "faint", fs: 12 }),
    box("lgF", 320, 495, "", "c-p", { w: 18, h: 14, rx: 2 }), txt("lgFt", 336, 500, "multiply-add (SIMD)", { anchor: "start", cls: "faint", fs: 12 }),
    txt("illus", 760, 500, "illustration, not to scale", { anchor: "end", cls: "faint", fs: 12 }),
    line("cur", G0, 90, G0, 470, { cls: "faint dash", o: 0 })
  ],
  async play(a) {
    await a.wait(300);
    const unit = 210;
    a.to("cur", { o: 1 }, 100);
    a.to("cur", { x1: GEND, x2: GEND }, ((GEND - G0) / GU) * unit, true);
    await a.all(ROWS.map(async (r, i) => {
      let x = G0, n = 0, chunks = 0;
      for (;;) {
        for (const [kind, u] of r.seq) {
          const w = u * GU;
          if (x + w > GEND) { a.to(`gc${i}`, { txt: `${chunks} chunks done` }, 0); return; }
          const k = `g${i}_${n++}`;
          const cls = kind === "L" ? "c-x" : kind === "W" ? "c-w" : "c-p";
          a.put(box(k, x, 150 + 120 * i, "", cls, { w: 0, h: 34, rx: 3, o: 1 }));
          await a.to(k, { w: w - 2, x: x + w / 2 }, u * unit, true);
          x += w;
          if (kind === "F") { chunks++; a.to(`gc${i}`, { txt: `${chunks}` }, 0); }
        }
      }
    }));
    a.to("cur", { o: 0 }, 200);
  }
});
})();
