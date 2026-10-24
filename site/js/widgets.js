// Interactive widgets for Phase 2 onwards. Loaded after site.js (it uses
// site.js's el() and fmtBytes()). Each init returns early when its element
// is not on the page, like the widgets in site.js.

(function () {
  initFileMap();
  initBpe();
  initRmsNorm();
  initRope();
  initAttn();
  initSoftmaxOverflow();
  initKvCalc();
  initSilu();
  initSampler2();
  initFloatOrder();
  initChains();
  initGemvBars();
  initTails();
  initReuse();
  initTile();
  initAbTest();
  initHandoff();
  initSplitSim();
  initSpinPark();
  initCoreSets();
  initQuantBlock();
  initPplCalc();
})();

function stats(box, rows) {
  box.innerHTML = "";
  for (const [v, l] of rows) box.append(el("div", { class: "stat" }, [el("b", { text: String(v) }), el("span", { text: l })]));
}

// ---------------------------------------------------------------- Phase 2: model file map

function initFileMap() {
  const root = document.querySelector("[data-filemap]");
  if (!root) return;
  // stories15M: dim 288, hidden 768, 6 layers, 6 heads (head size 48), vocab 32000, seq_len 256.
  const d = 288, h = 768, L = 6, V = 32000, S = 256, hs = 48;
  const parts = [
    ["header", "7 × i32", 7, "k-misc", "dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len. 28 bytes that fix every shape after them."],
    ["token_embedding", "[32000, 288]", V * d, "", "One row of 288 numbers per vocabulary token. Also used as the output classifier, because vocab_size in the header is positive (tied)."],
    ["rms_att", "6 × [288]", L * d, "k-norm", "The gain of the RMSNorm before attention, one vector per layer."],
    ["wq", "6 × [288, 288]", L * d * d, "k-att", "Query projection, one 288 × 288 matrix per layer."],
    ["wk", "6 × [288, 288]", L * d * d, "k-att", "Key projection."],
    ["wv", "6 × [288, 288]", L * d * d, "k-att", "Value projection."],
    ["wo", "6 × [288, 288]", L * d * d, "k-att", "Output projection: mixes the heads back into the residual stream."],
    ["rms_ffn", "6 × [288]", L * d, "k-norm", "The gain of the RMSNorm before the MLP."],
    ["w1 (gate)", "6 × [768, 288]", L * h * d, "k-mlp", "MLP gate: its output goes through SiLU."],
    ["w2 (down)", "6 × [288, 768]", L * d * h, "k-mlp", "MLP down projection, back to 288."],
    ["w3 (up)", "6 × [768, 288]", L * h * d, "k-mlp", "MLP up projection, multiplied by the gate."],
    ["rms_final", "[288]", d, "k-norm", "The gain of the last RMSNorm, before the classifier."],
    ["freq_cis", "2 × [256, 24]", S * hs, "k-misc", "A precomputed RoPE table left over from older llama2.c versions. Aether skips it and computes the angles itself."]
  ];
  const total = 28 + parts.slice(1).reduce((a, p) => a + p[2] * 4, 0);
  const bar = root.querySelector("[data-bar]"), out = root.querySelector("[data-out]"), sbox = root.querySelector("[data-stats]");
  let offset = 0;
  const rows = parts.map((p, i) => {
    const bytes = i === 0 ? 28 : p[2] * 4;
    const r = { name: p[0], shape: p[1], bytes, cls: p[3], says: p[4], at: offset };
    offset += bytes;
    return r;
  });
  rows.forEach((r, i) => {
    const b = el("button", { type: "button", class: r.cls, title: r.name, style: "flex:" + Math.max(r.bytes / total, 0.006), text: r.bytes / total > 0.06 ? r.name : "" });
    b.addEventListener("click", () => show(i));
    bar.append(b);
  });
  function show(i) {
    [...bar.children].forEach((c, j) => c.classList.toggle("on", j === i));
    const r = rows[i];
    out.innerHTML = "";
    out.append(el("b", { text: r.name + " " }), el("code", { text: r.shape }), document.createTextNode(" — " + r.says));
    stats(sbox, [[fmtBytes(r.bytes), "bytes"], [(100 * r.bytes / total).toFixed(2) + " %", "of the file"], ["byte " + r.at.toLocaleString(), "starts at"], [fmtBytes(total), "whole file"]]);
  }
  show(1);
}

// ---------------------------------------------------------------- Phase 2: BPE merges

function initBpe() {
  const root = document.querySelector("[data-bpe]");
  if (!root) return;
  // Traced from stories15M's tokenizer.bin; the final ids equal llama2.c's encode().
  const DATA = [{"text":"Once upon a time","ids":[1,9038,2501,263,931],"steps":[{"t":[" ","O","n","c","e"," ","u","p","o","n"," ","a"," ","t","i","m","e"],"m":-1},{"t":[" ","O","n","c","e"," ","u","p","o","n"," ","a"," t","i","m","e"],"m":12,"s":-1,"id":260},{"t":[" ","O","n","c","e"," ","u","p","o","n"," a"," t","i","m","e"],"m":10,"s":-4,"id":263},{"t":[" ","O","n","c","e"," ","u","p","on"," a"," t","i","m","e"],"m":8,"s":-6,"id":265},{"t":[" ","O","n","c","e"," u","p","on"," a"," t","i","m","e"],"m":5,"s":-59,"id":318},{"t":[" ","O","n","c","e"," u","p","on"," a"," t","im","e"],"m":10,"s":-67,"id":326},{"t":[" ","O","n","ce"," u","p","on"," a"," t","im","e"],"m":3,"s":-87,"id":346},{"t":[" O","n","ce"," u","p","on"," a"," t","im","e"],"m":0,"s":-179,"id":438},{"t":[" O","n","ce"," u","p","on"," a"," t","ime"],"m":8,"s":-344,"id":603},{"t":[" O","n","ce"," up","on"," a"," t","ime"],"m":3,"s":-442,"id":701},{"t":[" O","n","ce"," up","on"," a"," time"],"m":6,"s":-672,"id":931},{"t":[" On","ce"," up","on"," a"," time"],"m":0,"s":-1292,"id":1551},{"t":[" On","ce"," upon"," a"," time"],"m":2,"s":-2242,"id":2501},{"t":[" Once"," upon"," a"," time"],"m":0,"s":-8779,"id":9038}]},{"text":"The robot smiled","ids":[1,450,19964,25156],"steps":[{"t":[" ","T","h","e"," ","r","o","b","o","t"," ","s","m","i","l","e","d"],"m":-1},{"t":[" ","T","h","e"," ","r","o","b","o","t"," s","m","i","l","e","d"],"m":10,"s":-10,"id":269},{"t":[" ","T","h","e"," ","r","o","b","o","t"," s","m","i","le","d"],"m":13,"s":-21,"id":280},{"t":[" ","T","h","e"," ","ro","b","o","t"," s","m","i","le","d"],"m":5,"s":-48,"id":307},{"t":[" T","h","e"," ","ro","b","o","t"," s","m","i","le","d"],"m":0,"s":-64,"id":323},{"t":[" T","h","e"," ","ro","b","ot"," s","m","i","le","d"],"m":6,"s":-68,"id":327},{"t":[" T","he"," ","ro","b","ot"," s","m","i","le","d"],"m":1,"s":-95,"id":354},{"t":[" The"," ","ro","b","ot"," s","m","i","le","d"],"m":0,"s":-191,"id":450},{"t":[" The"," ","ro","b","ot"," s","m","ile","d"],"m":7,"s":-229,"id":488},{"t":[" The"," ro","b","ot"," s","m","ile","d"],"m":1,"s":-437,"id":696},{"t":[" The"," ro","b","ot"," sm","ile","d"],"m":4,"s":-1301,"id":1560},{"t":[" The"," ro","b","ot"," sm","iled"],"m":5,"s":-2097,"id":2356},{"t":[" The"," ro","bot"," sm","iled"],"m":2,"s":-7192,"id":7451},{"t":[" The"," robot"," sm","iled"],"m":1,"s":-19705,"id":19964},{"t":[" The"," robot"," smiled"],"m":2,"s":-24897,"id":25156}]},{"text":"happily","ids":[1,2250,2354],"steps":[{"t":[" ","h","a","p","p","i","l","y"],"m":-1},{"t":[" h","a","p","p","i","l","y"],"m":0,"s":-39,"id":298},{"t":[" h","a","p","p","il","y"],"m":4,"s":-50,"id":309},{"t":[" h","a","pp","il","y"],"m":2,"s":-148,"id":407},{"t":[" ha","pp","il","y"],"m":0,"s":-188,"id":447},{"t":[" happ","il","y"],"m":0,"s":-1991,"id":2250},{"t":[" happ","ily"],"m":1,"s":-2095,"id":2354}]}];
  const pick = root.querySelector("[data-pick]"), chips = root.querySelector("[data-chips]"), note = root.querySelector("[data-note]");
  const stepBtn = root.querySelector("[data-step]"), playBtn = root.querySelector("[data-play]"), resetBtn = root.querySelector("[data-reset]");
  DATA.forEach((d, i) => pick.append(el("option", { value: String(i), text: "“" + d.text + "”" })));
  let step = 0, timer = 0;
  const show = (t) => t.replace(/ /g, "␣");
  function draw() {
    const d = DATA[Number(pick.value)], s = d.steps[step], last = step === d.steps.length - 1;
    chips.innerHTML = "";
    chips.append(el("span", { class: "chip p", text: "<s>" }));
    s.t.forEach((t, i) => chips.append(el("span", { class: "chip" + (i === s.m ? " merge" : last ? " g" : "") }, [document.createTextNode(show(t))])));
    if (step === 0) note.textContent = "Start: BOS (<s>), the dummy-prefix space, then one token per character. " + s.t.length + " pieces. Press Step to merge the best pair.";
    else if (!last) note.textContent = "Merge " + step + ": “" + show(s.t[s.m]) + "” (id " + s.id + ", score " + s.s + ") had the highest score of every adjacent pair that exists in the vocabulary. " + s.t.length + " pieces left.";
    else note.textContent = "Done after " + step + " merges: no adjacent pair is in the vocabulary any more. Ids: " + d.ids.join(" ") + " — exactly what llama2.c's encoder gives.";
    stepBtn.disabled = last;
  }
  function stop() { clearInterval(timer); timer = 0; playBtn.textContent = "Play"; }
  stepBtn.addEventListener("click", () => { if (step < DATA[Number(pick.value)].steps.length - 1) { step++; draw(); } });
  playBtn.addEventListener("click", () => {
    if (timer) return stop();
    const n = DATA[Number(pick.value)].steps.length - 1;
    if (step >= n) step = 0;
    playBtn.textContent = "Pause";
    timer = setInterval(() => { step++; draw(); if (step >= n) stop(); }, 850);
  });
  resetBtn.addEventListener("click", () => { stop(); step = 0; draw(); });
  pick.addEventListener("change", () => { stop(); step = 0; draw(); });
  draw();
}

// ---------------------------------------------------------------- Phase 2: RMSNorm

function initRmsNorm() {
  const root = document.querySelector("[data-rmsnorm]");
  if (!root) return;
  const n = 6;
  const sl = root.querySelector("[data-sliders]"), inB = root.querySelector("[data-in]"), outB = root.querySelector("[data-out-bars]");
  const sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const vals = [1.5, -0.5, 3, 0.8, -2, 0.4];
  const inputs = vals.map((v, i) => {
    const r = el("input", { type: "range", min: "-6", max: "6", step: "0.1", value: String(v), "aria-label": "x" + i });
    sl.append(el("label", { class: "field" }, [document.createTextNode("x" + i), r]));
    return r;
  });
  function bars(box, xs, max) {
    box.innerHTML = "";
    for (const v of xs) {
      const h = (50 * Math.min(Math.abs(v) / max, 1)).toFixed(1) + "%";
      box.append(el("span", {}, [el("i", { class: v < 0 ? "neg" : "", style: (v < 0 ? "top:50%;" : "bottom:50%;") + "height:" + h })]));
    }
  }
  function render() {
    const x = inputs.map((r) => Number(r.value));
    const ms = x.reduce((a, v) => a + v * v, 0) / n;
    const scale = 1 / Math.sqrt(ms + 1e-5);
    const y = x.map((v) => v * scale);
    bars(inB, x, 6); bars(outB, y, 3);
    stats(sbox, [[Math.sqrt(ms).toFixed(3), "RMS of x"], [scale.toFixed(3), "scale = 1/RMS"], [Math.sqrt(y.reduce((a, v) => a + v * v, 0) / n).toFixed(3), "RMS of output"]]);
    note.textContent = "Output = x × " + scale.toFixed(3) + ". Every element is multiplied by the same number, so the pattern (which elements are big, which negative) is kept; only the overall size is reset to RMS 1.";
  }
  root.querySelector("[data-scale]").addEventListener("click", () => { inputs.forEach((r) => (r.value = String(Math.max(-6, Math.min(6, Number(r.value) * 2))))); render(); });
  root.querySelector("[data-reset]").addEventListener("click", () => { inputs.forEach((r, i) => (r.value = String(vals[i]))); render(); });
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 2: RoPE

function initRope() {
  const root = document.querySelector("[data-rope]");
  if (!root) return;
  const svg = root.querySelector("svg"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const mS = root.querySelector("[data-m]"), nS = root.querySelector("[data-n]"), pair = root.querySelector("[data-pair]");
  const q0 = [0.9, 0.35], k0 = [0.55, 0.8];
  const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
  function arrow(v, color, label) {
    const cx = 130, cy = 130, R = 100, x = cx + v[0] * R, y = cy - v[1] * R;
    return '<line x1="' + cx + '" y1="' + cy + '" x2="' + x.toFixed(1) + '" y2="' + y.toFixed(1) + '" stroke="' + color + '" stroke-width="3" stroke-linecap="round"/>' +
      '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="4.5" fill="' + color + '"/>' +
      '<text x="' + (x + (v[0] >= 0 ? 8 : -8)).toFixed(1) + '" y="' + (y - 6).toFixed(1) + '" text-anchor="' + (v[0] >= 0 ? "start" : "end") + '" font-size="13" fill="' + color + '">' + label + "</text>";
  }
  function render() {
    const m = Number(mS.value), n = Number(nS.value), p = Number(pair.value);
    const freq = Math.pow(10000, -2 * p / 48);
    const q = rot(q0, m * freq), k = rot(k0, n * freq);
    const dot = q[0] * k[0] + q[1] * k[1];
    root.querySelector("[data-m-label]").textContent = m;
    root.querySelector("[data-n-label]").textContent = n;
    svg.innerHTML = '<circle cx="130" cy="130" r="100" fill="none" stroke="var(--line)"/><line x1="20" y1="130" x2="240" y2="130" stroke="var(--line)"/><line x1="130" y1="20" x2="130" y2="240" stroke="var(--line)"/>' +
      arrow(q, "var(--accent)", "q at " + m) + arrow(k, "var(--copper)", "k at " + n);
    stats(sbox, [[(freq).toPrecision(3), "radians per position"], [(m * freq % (2 * Math.PI)).toFixed(2), "q turned (rad)"], [(n * freq % (2 * Math.PI)).toFixed(2), "k turned (rad)"], [m - n, "distance m − n"], [dot.toFixed(4), "q · k"]]);
    note.textContent = "q · k depends only on the angle between the arrows, and that angle only on m − n. Press “move both” and watch the dot product stay put while both arrows turn.";
  }
  root.querySelector("[data-both]").addEventListener("click", () => {
    const step = 7;
    if (Number(mS.value) + step <= Number(mS.max) && Number(nS.value) + step <= Number(nS.max)) { mS.value = String(Number(mS.value) + step); nS.value = String(Number(nS.value) + step); render(); }
  });
  root.addEventListener("input", render);
  root.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- Phase 2: attention toy

function initAttn() {
  const root = document.querySelector("[data-attn]");
  if (!root) return;
  // Illustration: 2-number keys and values for five earlier tokens.
  const toks = ["The", "cat", "sat", "on", "the"];
  const K = [[0.2, 0.1], [1.6, 0.3], [0.3, 1.5], [-0.4, 0.5], [0.25, 0.05]];
  const Vv = [[0.1, 0.0], [0.9, 0.2], [0.1, 0.8], [-0.2, 0.3], [0.1, 0.1]];
  const ang = root.querySelector("[data-angle]"), len = root.querySelector("[data-len]"), upto = root.querySelector("[data-upto]");
  const cands = root.querySelector("[data-weights]"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  function render() {
    const a = Number(ang.value) * Math.PI / 180, L = Number(len.value), P = Number(upto.value);
    const q = [L * Math.cos(a), L * Math.sin(a)];
    const scores = K.slice(0, P + 1).map((k) => (q[0] * k[0] + q[1] * k[1]) / Math.sqrt(2));
    const mx = Math.max(...scores), ex = scores.map((s) => Math.exp(s - mx)), sum = ex.reduce((x, y) => x + y, 0);
    const w = ex.map((e) => e / sum);
    const out = [0, 1].map((j) => w.reduce((acc, wi, i) => acc + wi * Vv[i][j], 0));
    cands.innerHTML = "";
    toks.forEach((t, i) => {
      const live = i <= P;
      cands.append(el("div", { class: "cand" + (live ? (w[i] === Math.max(...w) ? " pick" : "") : " drop") }, [
        el("span", { text: "“" + t + "” " + (live ? scores[i].toFixed(2) : "future") }),
        el("i", { style: "width:" + (live ? (w[i] * 100).toFixed(0) : 0) + "%" }),
        el("b", { text: live ? (w[i] * 100).toFixed(0) + "%" : "—" })
      ]));
    });
    root.querySelector("[data-angle-label]").textContent = ang.value + "°";
    root.querySelector("[data-len-label]").textContent = L.toFixed(1);
    root.querySelector("[data-upto-label]").textContent = "“" + toks[P] + "” (position " + P + ")";
    stats(sbox, [["[" + q.map((v) => v.toFixed(2)).join(", ") + "]", "query q"], ["[" + out.map((v) => v.toFixed(2)).join(", ") + "]", "output = Σ weight × value"], [(Math.max(...w) * 100).toFixed(0) + " %", "largest weight"]]);
    note.textContent = L < 1 ? "A short query gives similar scores everywhere, so softmax spreads the weight: the output is a blur of all values." :
      L > 4 ? "A long query makes the score gaps large, so softmax puts almost all weight on one token: the output is nearly that token's value." :
      "Turn the query towards a key and that token's weight grows. Positions after the current one are hidden: a token may only look back.";
  }
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 2: softmax overflow

function initSoftmaxOverflow() {
  const root = document.querySelector("[data-smax]");
  if (!root) return;
  const ins = [...root.querySelectorAll("[data-s]")], out = root.querySelector("[data-out]");
  const f = Math.fround;
  const fmt = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : v.toFixed(4)) : String(v));
  function render() {
    const s = ins.map((i) => Number(i.value));
    // f32 exp: anything above ~88.72 overflows to Infinity.
    const naiveE = s.map((v) => f(Math.exp(v))), naiveSum = naiveE.reduce((a, b) => f(a + b), 0);
    const naive = naiveE.map((e) => f(e / naiveSum));
    const mx = Math.max(...s), stE = s.map((v) => f(Math.exp(v - mx))), stSum = stE.reduce((a, b) => f(a + b), 0);
    const stable = stE.map((e) => f(e / stSum));
    out.textContent =
      "scores            " + s.map(fmt).join("   ") + "\n\n" +
      "naive  exp(s)     " + naiveE.map(fmt).join("   ") + "\n" +
      "naive  softmax    " + naive.map(fmt).join("   ") + "\n\n" +
      "stable exp(s−max) " + stE.map(fmt).join("   ") + "\n" +
      "stable softmax    " + stable.map(fmt).join("   ");
  }
  root.querySelector("[data-big]").addEventListener("click", () => { ins.forEach((i, k) => (i.value = String([95, 92, 90][k]))); render(); });
  root.querySelector("[data-reset]").addEventListener("click", () => { ins.forEach((i, k) => (i.value = String([2, 1, 0.1][k]))); render(); });
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 2: KV cache calculator

function initKvCalc() {
  const root = document.querySelector("[data-kvcalc]");
  if (!root) return;
  const models = {
    s15: { name: "stories15M", layers: 6, kvDim: 288, max: 256, weights: 60.8e6 },
    smol: { name: "SmolLM2-135M", layers: 30, kvDim: 192, max: 8192, weights: 538.1e6 },
    qwen: { name: "Qwen2.5-0.5B", layers: 24, kvDim: 128, max: 32768, weights: 278e6 }
  };
  const pick = root.querySelector("[data-model]"), ctx = root.querySelector("[data-ctx]"), bytes = root.querySelector("[data-bytes]");
  const out = root.querySelector("[data-out]"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  function render() {
    const m = models[pick.value];
    ctx.max = String(m.max);
    if (Number(ctx.value) > m.max) ctx.value = String(m.max);
    const T = Number(ctx.value), b = Number(bytes.value);
    const perTok = m.layers * m.kvDim * 2 * b, total = perTok * T;
    root.querySelector("[data-ctx-label]").textContent = T.toLocaleString() + " tokens";
    out.textContent = m.layers + " layers × " + T.toLocaleString() + " tokens × " + m.kvDim + " (K width) × 2 (K and V) × " + b + " bytes = " + fmtBytes(total);
    stats(sbox, [[fmtBytes(perTok), "per token"], [fmtBytes(total), "whole cache"], [(100 * total / m.weights).toFixed(1) + " %", "of the weights' size"]]);
    note.textContent = m.name === "Qwen2.5-0.5B" ? "Weights compared at the Q4_0 size the app uses (278 MB). Qwen keeps the cache small with only 2 key/value heads (grouped-query attention, Phase 5)." :
      m.name === "SmolLM2-135M" ? "Weights compared at f32 (538 MB). 3 key/value heads for 9 query heads keep each token's entry at 192 numbers per layer." :
      "stories15M can only see 256 tokens: the model was trained with that context, and the cache is sized for it.";
  }
  root.addEventListener("input", render);
  root.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- Phase 2: SiLU and SwiGLU

function initSilu() {
  const root = document.querySelector("[data-silu]");
  if (!root) return;
  const svg = root.querySelector("svg"), g = root.querySelector("[data-g]"), u = root.querySelector("[data-u]"), sbox = root.querySelector("[data-stats]");
  const silu = (x) => x / (1 + Math.exp(-x));
  const X = (x) => 30 + (x + 6) * 40, Y = (y) => 150 - y * 22;
  function render() {
    const gv = Number(g.value), uv = Number(u.value);
    root.querySelector("[data-g-label]").textContent = gv.toFixed(1);
    root.querySelector("[data-u-label]").textContent = uv.toFixed(1);
    let s = '<line x1="30" y1="150" x2="510" y2="150" stroke="var(--line)"/><line x1="270" y1="10" x2="270" y2="200" stroke="var(--line)"/>';
    const path = (fn) => { let p = ""; for (let x = -6; x <= 6.001; x += 0.1) p += (p ? " L" : "M") + X(x).toFixed(1) + "," + Y(fn(x)).toFixed(1); return p; };
    s += '<path d="' + path((x) => Math.max(0, x)) + '" fill="none" stroke="var(--faint)" stroke-width="2" stroke-dasharray="5 4"/>';
    s += '<path d="' + path(silu) + '" fill="none" stroke="var(--accent)" stroke-width="2.6"/>';
    s += '<circle cx="' + X(gv) + '" cy="' + Y(silu(gv)) + '" r="5" fill="var(--copper)"/>';
    s += '<text x="40" y="24" font-size="12" fill="var(--accent)">SiLU(x) = x / (1 + e^−x)</text><text x="40" y="40" font-size="12" fill="var(--faint)">ReLU(x) = max(0, x), for comparison</text>';
    for (const t of [-6, -3, 0, 3, 6]) s += '<text x="' + X(t) + '" y="166" font-size="10" text-anchor="middle" class="muted">' + t + "</text>";
    svg.innerHTML = s;
    stats(sbox, [[silu(gv).toFixed(3), "SiLU(gate)"], [uv.toFixed(2), "up"], [(silu(gv) * uv).toFixed(3), "SiLU(gate) × up"]]);
  }
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 2: sampling

function initSampler2() {
  const root = document.querySelector("[data-sampler2]");
  if (!root) return;
  // Illustration: logits for the word after "Once upon a time, there was a little".
  const words = ["girl", "boy", "dog", "cat", "bird", "house", "tree", "robot"];
  const logits = [5.2, 4.9, 4.1, 4.0, 3.1, 1.2, 0.9, 0.4];
  const T = root.querySelector("[data-t]"), K = root.querySelector("[data-k]"), P = root.querySelector("[data-p]");
  const box = root.querySelector("[data-weights]"), chips = root.querySelector("[data-chips]"), note = root.querySelector("[data-note]");
  let last = [];
  function dist() {
    const t = Number(T.value), k = Number(K.value), p = Number(P.value);
    if (t === 0) { const best = logits.indexOf(Math.max(...logits)); return words.map((_, i) => ({ i, p: i === best ? 1 : 0, keep: i === best })); }
    const z = logits.map((l) => l / t), mx = Math.max(...z), e = z.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0);
    let items = e.map((v, i) => ({ i, p: v / s, keep: true })).sort((a, b) => b.p - a.p);
    if (k > 0) items.forEach((it, r) => { if (r >= k) it.keep = false; });
    if (p > 0 && p < 1) { let m = 0, cut = false; items.forEach((it) => { if (!it.keep) return; if (cut) it.keep = false; m += it.p; if (m > p) cut = true; }); }
    const mass = items.filter((x) => x.keep).reduce((a, b) => a + b.p, 0);
    items.forEach((it) => (it.final = it.keep ? it.p / mass : 0));
    return items.sort((a, b) => a.i - b.i);
  }
  function render() {
    const d = dist(), t = Number(T.value);
    last = d;
    root.querySelector("[data-t-label]").textContent = t === 0 ? "0 (greedy)" : t.toFixed(2);
    root.querySelector("[data-k-label]").textContent = Number(K.value) === 0 ? "off" : K.value;
    root.querySelector("[data-p-label]").textContent = Number(P.value) >= 1 || Number(P.value) <= 0 ? "off" : Number(P.value).toFixed(2);
    box.innerHTML = "";
    d.forEach((it) => {
      const shown = t === 0 ? it.p : it.final;
      box.append(el("div", { class: "cand" + (it.keep ? "" : " drop") + (shown === Math.max(...d.map((x) => (t === 0 ? x.p : x.final))) ? " pick" : "") }, [
        el("span", { text: "“" + words[it.i] + "” " + logits[it.i].toFixed(1) }), el("i", { style: "width:" + (shown * 100).toFixed(0) + "%" }), el("b", { text: (shown * 100).toFixed(0) + "%" })
      ]));
    });
    const kept = d.filter((x) => x.keep).length;
    note.textContent = t === 0 ? "Temperature 0 means greedy: always “girl”. Deterministic, which is why every correctness test uses it." :
      kept + " of 8 words can be drawn. " + (t < 0.7 ? "Low temperature sharpens: the top words take nearly everything." : t > 1.3 ? "High temperature flattens: unlikely words get a real chance." : "");
  }
  root.querySelector("[data-draw]").addEventListener("click", () => {
    const t = Number(T.value);
    let r = Math.random(), pickI = 0;
    for (const it of last) { const pr = t === 0 ? it.p : it.final; if (r < pr) { pickI = it.i; break; } r -= pr; pickI = it.i; }
    chips.append(el("span", { class: "chip g", text: words[pickI] }));
    if (chips.children.length > 24) chips.firstChild.remove();
  });
  root.querySelector("[data-clear]").addEventListener("click", () => { chips.innerHTML = ""; });
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 2: float summation order

function initFloatOrder() {
  const root = document.querySelector("[data-floatorder]");
  if (!root) return;
  const f = Math.fround;
  // Products like those in a dot product: mixed signs and sizes.
  const xs = [-0.07610889, 0.9858316, -0.0111042, 0.849659, 2.595785, 5.757236, -0.02401409, 0.04495488, -9.482862, -1.724546, 0.03033164, -0.06967886].map(f);
  const out = root.querySelector("[data-out]"), note = root.querySelector("[data-note]");
  const bits = (v) => { const b = new DataView(new ArrayBuffer(4)); b.setFloat32(0, v); return "0x" + b.getUint32(0).toString(16).padStart(8, "0"); };
  const orders = {
    ltr: ["left to right (llama2.c, scalar Aether)", () => xs.reduce((a, v) => f(a + v), 0)],
    rtl: ["right to left", () => [...xs].reverse().reduce((a, v) => f(a + v), 0)],
    four: ["4 accumulators, then combined (NEON style)", () => { const a = [0, 0, 0, 0]; xs.forEach((v, i) => (a[i % 4] = f(a[i % 4] + v))); return f(f(a[0] + a[1]) + f(a[2] + a[3])); }],
    sorted: ["smallest magnitude first", () => [...xs].sort((p, q) => Math.abs(p) - Math.abs(q)).reduce((a, v) => f(a + v), 0)]
  };
  const exact = xs.reduce((a, v) => a + v, 0);
  function render() {
    let s = "";
    const seen = new Set();
    for (const [k, [label, fn]] of Object.entries(orders)) {
      const v = fn();
      seen.add(bits(v));
      s += label.padEnd(44) + v.toPrecision(9).padStart(14) + "  " + bits(v) + "\n";
    }
    s += "\n" + "exact (float64), for reference".padEnd(44) + exact.toPrecision(9).padStart(14);
    out.textContent = s;
    note.textContent = seen.size + " different f32 results from the same 12 numbers. None is wrong; each order rounds at different moments. Matching llama2.c bit for bit therefore means matching its order exactly.";
  }
  render();
}

// ---------------------------------------------------------------- Phase 3: dependency chains

function initChains() {
  const root = document.querySelector("[data-chains]");
  if (!root) return;
  // Illustration: one FMA pipe, a new FMA can start every cycle, each takes LAT cycles.
  const LAT = 4, CYC = 32;
  const grid = root.querySelector("[data-grid]"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  function render() {
    const acc = Number(root.querySelector("[data-acc]:checked").value);
    grid.innerHTML = "";
    grid.style.gridTemplateColumns = "repeat(" + CYC + ", minmax(0, 1fr))";
    // FMA j goes to accumulator j % acc; it can start when the previous FMA on that accumulator finished.
    const ready = new Array(acc).fill(0);
    const starts = [];
    let cyc = 0, j = 0;
    while (true) {
      const a = j % acc, s = Math.max(ready[a], cyc);
      if (s + LAT > CYC) break;
      starts.push([s, a]);
      ready[a] = s + LAT; cyc = s + 1; j++;
    }
    for (let row = 0; row < Math.min(acc, 4); row++) {
      for (let c = 0; c < CYC; c++) {
        const hit = starts.find(([s, a]) => a === row && c >= s && c < s + LAT);
        grid.append(el("span", { class: hit ? "a" + row : "wait", text: hit && c === hit[0] ? "+" : "" }));
      }
    }
    stats(sbox, [[starts.length, "FMAs finished in " + CYC + " cycles"], [(starts.length / CYC).toFixed(2), "FMAs per cycle"], [acc + " × 4 lanes", "independent chains"], [(starts.length * 4 / CYC).toFixed(2), "multiply-adds per cycle (4 lanes)"]]);
    note.textContent = acc === 1 ? "One accumulator: every FMA needs the previous FMA's result, so a new one can only start when the last finishes, every " + LAT + " cycles. Only one FMA is ever in flight; the unit could start one every cycle." :
      "With " + acc + " accumulators, FMA j adds into register j mod " + acc + ". While one waits for its result, the others start. " + (acc >= LAT ? "With as many chains as cycles of latency, the unit starts a new FMA every cycle." : "Still not enough chains to hide all " + LAT + " cycles.");
  }
  root.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- Phase 3: GEMV steps, measured

function initGemvBars() {
  const root = document.querySelector("[data-gemvbars]");
  if (!root) return;
  // results/phase-3/raw/gemv-bench-core7-v2.csv, median GB/s, one prime core (7), battery caps.
  const M = {
    wq: ["stories15M wq, 288 × 288 (324 KB, fits L2)", [5.34, 17.84, 41.63, 53.53]],
    w1: ["stories15M w1, 768 × 288 (864 KB, fits L2)", [5.32, 17.84, 41.43, 52.92]],
    cls: ["stories15M classifier, 32000 × 288 (36.9 MB)", [5.28, 17.58, 41.41, 28.95]],
    up: ["Qwen2.5 up, 4864 × 896 (17.4 MB)", [3.47, 13.25, 40.17, 32.92]],
    down: ["Qwen2.5 down, 896 × 4864 (17.4 MB)", [3.06, 9.54, 36.16, 48.59]],
    big: ["synthetic 8192 × 8192 (256 MB, DRAM)", [3.03, 9.25, 36.01, 40.59]]
  };
  const names = ["scalar", "NEON, 1 accumulator", "NEON, 4 accumulators per row", "NEON, 4 rows × 2 accumulators (gemv)"];
  const pick = root.querySelector("[data-pick]"), box = root.querySelector("[data-bars]"), note = root.querySelector("[data-note]");
  for (const [k, [label]] of Object.entries(M)) pick.append(el("option", { value: k, text: label }));
  function render() {
    const [, v] = M[pick.value];
    box.innerHTML = "";
    v.forEach((g, i) => box.append(el("div", { class: "bar" + (i === 3 && g < v[2] ? " alt" : "") }, [el("span", { text: names[i] }), el("i", { style: "--w:" + (g / 70 * 100).toFixed(1) + "%" }), el("b", { text: g.toFixed(1) + " GB/s" })])));
    box.append(el("div", { class: "bar alt" }, [el("span", { text: "pure read, same core (Phase 1)" }), el("i", { style: "--w:" + (pick.value === "big" ? 61 / 70 * 100 : 67 / 70 * 100).toFixed(1) + "%" }), el("b", { text: pick.value === "big" ? "~61 GB/s (DRAM)" : "~67 GB/s (L2)" })]));
    note.textContent = v[3] < v[2] ? "Here the 4-row kernel loses to the simpler one: short rows (" + (pick.value === "up" ? "896" : "288") + " floats) mean four streams that each jump after a few KB. Likely the prefetcher, not yet measured." :
      "Each step removes one limit: scalar is one dependency chain, one accumulator still waits on FMA latency, four accumulators hide the latency, and four rows at once reuse each loaded piece of x four times.";
  }
  pick.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- Phase 3: tails

function initTails() {
  const root = document.querySelector("[data-tails]");
  if (!root) return;
  const n = root.querySelector("[data-n]"), cells = root.querySelector("[data-cells]"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  function render() {
    const N = Number(n.value), n16 = Math.floor(N / 16) * 16, n4 = Math.floor(N / 4) * 4;
    root.querySelector("[data-n-label]").textContent = N;
    cells.innerHTML = "";
    for (let i = 0; i < N; i++) cells.append(el("span", { class: i < n16 ? "c16" : i < n4 ? "c4" : "c1", title: "element " + i }));
    stats(sbox, [[n16, "in 16-float steps (4 registers)"], [n4 - n16, "in 4-float steps"], [N - n4, "one at a time (scalar tail)"]]);
    note.textContent = N - n4 === 0 && n4 === n16 ? "A multiple of 16: the fast loop does everything. Real sizes (288, 896, 4864) are like this, which is exactly why tests must also try odd ones." :
      "The vector loops stop before the end; the leftovers need their own code. Get a bound wrong by one and the kernel reads past the array or skips the last numbers.";
  }
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 3: reuse with prompt length

function initReuse() {
  const root = document.querySelector("[data-reuse]");
  if (!root) return;
  const t = root.querySelector("[data-t]"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]"), box = root.querySelector("[data-bars]");
  const W = 60.8e6, BW = 61e9; // stories15M weight bytes; one prime core's DRAM read (Phase 1, measured)
  function render() {
    const T = Number(t.value);
    root.querySelector("[data-t-label]").textContent = T + (T === 1 ? " token (decode)" : " tokens (prefill)");
    const perTok = W / T;
    stats(sbox, [[T, "uses of each loaded weight"], [(T / 2).toFixed(1), "FLOP per byte of weights"], [fmtBytes(perTok), "weight bytes per token"], [(perTok / BW * 1e3).toFixed(3) + " ms", "memory floor per token at 61 GB/s"]]);
    box.innerHTML = "";
    box.append(el("div", { class: "bar" }, [el("span", { text: "bytes per token" }), el("i", { style: "--w:" + (100 / T).toFixed(2) + "%" }), el("b", { text: fmtBytes(perTok) })]));
    note.textContent = T === 1 ? "Decode: each 4-byte weight feeds one multiply-add. Memory sets the pace." :
      T < 16 ? "Each weight now feeds " + T + " multiply-adds. The memory cost per token falls as 1/T." :
      "At this reuse the memory floor is tiny; the arithmetic itself becomes the limit, so the kernel's FLOPs per cycle matter (the GEMM micro-kernel).";
  }
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 3: GEMM tile

function initTile() {
  const root = document.querySelector("[data-tile]");
  if (!root) return;
  const R = root.querySelector("[data-r]"), C = root.querySelector("[data-c]"), grid = root.querySelector("[data-grid]");
  const sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  function render() {
    const r = Number(R.value), c = Number(C.value);
    root.querySelector("[data-r-label]").textContent = r;
    root.querySelector("[data-c-label]").textContent = c;
    grid.innerHTML = "";
    grid.style.gridTemplateColumns = "repeat(" + (c + 1) + ", 1.6rem)";
    grid.append(el("span", { class: "blank" }));
    for (let j = 0; j < c; j++) grid.append(el("span", { class: "x", text: "x" + j }));
    for (let i = 0; i < r; i++) {
      grid.append(el("span", { class: "w", text: "w" + i }));
      for (let j = 0; j < c; j++) grid.append(el("span", { class: "acc", text: i + "," + j }));
    }
    const loads = r + c, fmas = r * c, regs = r * c + r + c;
    stats(sbox, [[loads, "vector loads per step"], [fmas, "FMAs per step"], [(fmas / loads).toFixed(2), "FMAs per load"], [regs + " / 32", "NEON registers needed"]]);
    note.textContent = regs > 32 ? "Too many: there are only 32 NEON registers, so accumulators would spill to memory and the gain disappears." :
      r === 4 && c === 4 ? "Aether's tile: 8 loads feed 16 FMAs, 2 per load, using 24 of the 32 registers." :
      "Each step loads " + r + " weight vectors and " + c + " token vectors, and every pair gets its own accumulator. Bigger tiles reuse each load more, until the registers run out.";
  }
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 3: A/B under drift

function initAbTest() {
  const root = document.querySelector("[data-abtest]");
  if (!root) return;
  // Illustration: the phone's speed drifts down as it warms; B is truly 10 % faster than A.
  const out = root.querySelector("[data-out]"), svg = root.querySelector("svg"), note = root.querySelector("[data-note]");
  const base = (i) => 110 * (1 - 0.03 * i) + 1.5 * Math.sin(i * 2.1);
  const ORDER = { seq: "AAAABBBB", alt: "ABBABAAB" };
  function run(mode) {
    const runs = [...ORDER[mode]].map((who, i) => [who, base(i) * (who === "B" ? 1.10 : 1)]);
    const mean = (w) => runs.filter((r) => r[0] === w).reduce((a, r) => a + r[1], 0) / 4;
    const a = mean("A"), b = mean("B");
    let s = '<line x1="10" y1="130" x2="470" y2="130" stroke="var(--line)"/>';
    runs.forEach(([w, v], i) => {
      const x = 38 + i * 57, h = (v - 60) * 1.6;
      s += '<rect x="' + (x - 18) + '" y="' + (130 - h) + '" width="36" height="' + h + '" rx="3" fill="' + (w === "A" ? "var(--copper)" : "var(--accent)") + '"/>';
      s += '<text x="' + x + '" y="146" text-anchor="middle" font-size="11" class="muted">' + w + " " + (i + 1) + "</text>";
      s += '<text x="' + x + '" y="' + (124 - h) + '" text-anchor="middle" font-size="11">' + v.toFixed(0) + "</text>";
    });
    svg.innerHTML = s;
    out.textContent = "order " + ORDER[mode] + ": mean A " + a.toFixed(1) + " tok/s, mean B " + b.toFixed(1) + " tok/s → B looks " + Math.abs((b / a - 1) * 100).toFixed(1) + " % " + (b > a ? "faster" : "slower") + " (truth: 10 % faster)";
    note.textContent = mode === "seq" ? "All A first, then all B: B ran on a warmer, slower phone, so the drift is charged to B and a real 10 % gain shows up as a loss." :
      "A and B take turns in a balanced order (each has the same average position in time), so the drift falls on both equally and cancels. What is left is ordinary noise.";
  }
  root.querySelector("[data-seq]").addEventListener("click", () => run("seq"));
  root.querySelector("[data-alt]").addEventListener("click", () => run("alt"));
  run("seq");
}

// ---------------------------------------------------------------- Phase 4: hand-out cost per token

function initHandoff() {
  const root = document.querySelector("[data-handoff]");
  if (!root) return;
  const pick = root.querySelector("[data-model]"), bar = root.querySelector("[data-bar]");
  const sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  function render() {
    const [jobs, tps] = pick.value.split(",").map(Number);
    const cost = Number(root.querySelector("[data-ho]:checked").value); // µs per job, phone, cores 2-7
    const token = 1000 / tps, extra = jobs * cost / 1000; // ms
    const total = token + extra;
    bar.innerHTML = "";
    bar.append(el("i", { style: "width:" + (100 * token / total) + "%;background:var(--accent)" }), el("i", { style: "width:" + (100 * extra / total) + "%;background:var(--copper)" }));
    stats(sbox, [[token.toFixed(2) + " ms", "token, measured"], [extra.toFixed(extra < 0.1 ? 3 : 2) + " ms", jobs + " jobs × " + cost + " µs"], [(100 * extra / token).toFixed(extra / token < 0.1 ? 1 : 0) + " %", "added to every token"], [(1000 / total).toFixed(0), "tok/s if this were the only change"]]);
    note.textContent = cost > 100 ? "Starting threads per job, or waking them through a lock, costs far more than the job itself: the hand-out alone takes many times the token's whole measured time." :
      cost > 10 ? "Waking sleeping workers through the kernel every job would add a large slice to every token." :
      "Watching a counter keeps the hand-out to about a tenth of the token. The remaining costs of threading are elsewhere: waiting for the slowest thread, and the serial work between jobs.";
  }
  root.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- Phase 4: equal shares vs blocks

function initSplitSim() {
  const root = document.querySelector("[data-splitsim]");
  if (!root) return;
  // Illustration: speed in rows per unit time, from the Phase 1 one-core DRAM reads (prime 61, perf 43 GB/s).
  const ROWS = 1536, ALIGN = 16;
  const lanes = root.querySelector("[data-lanes]"), slow = root.querySelector("[data-slow]");
  const sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  // pool.rs split(): blocks of ALIGN dealt out evenly, the first blocks % n shares get one more.
  function evenShare(share, n) {
    const blocks = Math.ceil(ROWS / ALIGN), base = Math.floor(blocks / n), extra = blocks % n;
    const start = share * base + Math.min(share, extra), mine = base + (share < extra ? 1 : 0);
    return Math.min((start + mine) * ALIGN, ROWS) - Math.min(start * ALIGN, ROWS);
  }
  function render() {
    const cores = root.querySelector("[data-set]:checked").value.split(",").map(Number); // caller (fastest) first
    const blocksMode = root.querySelector("[data-mode]:checked").value === "blocks";
    const sf = Number(slow.value) / 100;
    root.querySelector("[data-slow-label]").textContent = sf === 1 ? "no extra load" : Math.round(sf * 100) + " % of its speed";
    const slowCore = cores.includes(0) ? 0 : cores.find((c) => c < 6); // one performance core with other work
    const speed = cores.map((c) => (c >= 6 ? 61 : 43) * (c === slowCore ? sf : 1));
    const n = cores.length;
    const segs = cores.map(() => []);
    if (!blocksMode) {
      cores.forEach((c, i) => segs[i].push(evenShare(i, n) / speed[i]));
    } else {
      // pool.rs block_size(): about 8 blocks per thread, a multiple of ALIGN.
      const block = Math.max(1, Math.ceil(Math.floor(ROWS / (n * 8)) / ALIGN)) * ALIGN;
      const free = new Array(n).fill(0);
      for (let start = 0; start < ROWS; start += block) {
        let t = 0;
        for (let i = 1; i < n; i++) if (free[i] < free[t] - 1e-9) t = i; // the thread that comes back first takes it
        const dt = (Math.min(start + block, ROWS) - start) / speed[t];
        segs[t].push(dt);
        free[t] += dt;
      }
    }
    const busy = segs.map((s) => s.reduce((a, b) => a + b, 0));
    const T = Math.max(...busy), ideal = ROWS / speed.reduce((a, b) => a + b, 0);
    lanes.innerHTML = "";
    cores.forEach((c, i) => {
      const bar = el("div", { class: "split" });
      segs[i].forEach((d, k) => bar.append(el("i", { class: (c >= 6 ? "pr" : "pf") + (k % 2 ? " alt" : ""), style: "width:" + (100 * d / T) + "%" })));
      if (T - busy[i] > 1e-9) bar.append(el("i", { class: "hatch", style: "width:" + (100 * (T - busy[i]) / T) + "%" }));
      const tag = "core " + c + (i === 0 ? " (caller)" : "") + (c === slowCore && sf < 1 ? " · loaded" : "");
      lanes.append(el("span", { text: tag }), bar);
    });
    const idle = 1 - busy.reduce((a, b) => a + b, 0) / (n * T);
    // Two prime cores with equal shares, as a yardstick.
    const ref = Math.max(evenShare(0, 2), evenShare(1, 2)) / 61;
    stats(sbox, [[(T / ideal).toFixed(2) + " ×", "job time vs a perfect split"], [Math.round(100 * idle) + " %", "of thread time spent waiting"], [(ref / T).toFixed(2) + " ×", "speed vs 2 prime cores, equal shares"], [blocksMode ? segs.reduce((a, s) => a + s.length, 0) + " blocks" : n + " shares", blocksMode ? "handed out from the counter" : "fixed in advance"]]);
    note.textContent = !blocksMode && n > 2 ? "Equal shares: every thread got the same number of rows, so the job ends when the slowest one does, and the others wait in the stripes." :
      !blocksMode ? "Two equal cores with equal shares finish together; little waiting. Add a slower core and that changes." :
      "Blocks: a thread that comes back early takes the next block. The loaded core simply does fewer, and the waiting at the end is at most about one block.";
  }
  root.addEventListener("change", render);
  slow.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 4: spin, then park

function initSpinPark() {
  const root = document.querySelector("[data-spinpark]");
  if (!root) return;
  // Assumptions: spin window 40 µs (20,000 polls, not measured); wake-up 1.3 µs spinning (the pool's hand-out on 6 threads), 48 µs parked (the lock-and-condvar hand-out on 2 threads), both phone figures for a whole hand-out, not for one wake-up alone.
  const W = 40, FAST = 1.3, SLOW = 48;
  const gap = root.querySelector("[data-gap]"), bar = root.querySelector("[data-bar]");
  const sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const fmt = (us) => us >= 1e6 ? (us / 1e6).toFixed(1) + " s" : us >= 1000 ? (us / 1000).toFixed(us >= 1e4 ? 0 : 1) + " ms" : (us >= 10 ? us.toFixed(0) : us.toFixed(1)) + " µs";
  function render() {
    const g = Math.pow(10, Number(gap.value) / 100 * 7); // 1 µs .. 10 s
    root.querySelector("[data-gap-label]").textContent = fmt(g);
    const mode = root.querySelector("[data-sp]:checked").value;
    const spin = mode === "spin" ? g : mode === "park" ? 0 : Math.min(g, W);
    const parked = g - spin, wake = parked > 0 ? SLOW : FAST;
    // Log-scaled segment widths so 1 µs and 10 s both show.
    const lg = (x) => (x > 0 ? Math.log10(1 + x) : 0);
    const tot = lg(spin) + lg(parked) + lg(wake);
    bar.innerHTML = "";
    bar.append(el("i", { style: "width:" + (100 * lg(spin) / tot) + "%;background:var(--copper)" }), el("i", { class: "hatch", style: "width:" + (100 * lg(parked) / tot) + "%" }), el("i", { style: "width:" + (100 * lg(wake) / tot) + "%;background:var(--blue)" }));
    stats(sbox, [[fmt(wake), "delay before the job starts"], [fmt(spin), "core kept busy spinning"], [Math.round(100 * spin / g) + " %", "of the gap burned"]]);
    note.textContent = mode === "spin" && g > 1e4 ? "Spinning through a long gap: the job starts instantly, but the core burned the whole gap doing nothing. Between chat messages that is seconds per core." :
      mode === "park" && g < 100 ? "Parking for a short gap: the core is free, but every job now starts tens of microseconds late. During decode that is added about 240 times per token." :
      mode === "both" && g <= W ? "A decode-length gap: the new job arrives while the worker is still spinning, so it starts at once." :
      mode === "both" ? "A long gap: the worker spun for its window, then parked. It wastes at most the window, and pays the wake-up only when the wait was long anyway." :
      "Try a gap of a few microseconds (decode) and of several seconds (between messages).";
  }
  root.addEventListener("change", render);
  gap.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- Phase 4: core sets, measured

function initCoreSets() {
  const root = document.querySelector("[data-coresets]");
  if (!root) return;
  // SmolLM2-135M Q4_0 decode, tok/s, two passes (second in reverse order), on battery, caps 2.40/2.44 GHz:
  // results/phase-4.md, clean sweep.
  const MEAS = {
    "7": [160.0, 158.9], "6,7": [262.8, 261.3], "5,6,7": [276.2, 279.9], "4,5,6,7": [310.6, 299.0],
    "3,4,5,6,7": [326.2, 327.1], "2,3,4,5,6,7": [326.4, 343.4], "1,2,3,4,5,6,7": [347.8, 338.0],
    "0,1,2,3,4,5,6,7": [340.6, 310.5], "0": [83.3, 81.8], "0,1": [135.7, 144.8], "2,3,4,5": [219.3, 200.2], "0,1,2,3,4,5": [247.5, 184.4],
  };
  const mean = (k) => (MEAS[k][0] + MEAS[k][1]) / 2;
  const MB = 75.8, ALL = 67, top = 360;
  const box = root.querySelector("[data-cores]"), bars = root.querySelector("[data-bars]");
  const sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const on = new Set([2, 3, 4, 5, 6, 7]);
  const btns = [];
  for (let c = 0; c < 8; c++) {
    const prime = c >= 6;
    const b = el("button", { type: "button", class: "core" + (prime ? " prime" : ""), "aria-pressed": "false" }, [el("b", { text: "cpu" + c }), el("small", { text: prime ? "prime" : "perf" })]);
    b.addEventListener("click", () => {
      if (on.has(c)) on.delete(c); else on.add(c);
      root.querySelectorAll("[data-quick]").forEach((r) => (r.checked = false));
      render();
    });
    box.append(b);
    btns.push(b);
  }
  root.querySelectorAll("[data-quick]").forEach((r) => r.addEventListener("change", () => {
    on.clear();
    r.value.split(",").forEach((c) => on.add(Number(c)));
    render();
  }));
  const name = (k) => {
    const l = k.split(",").map(Number);
    return l.length > 1 && l[l.length - 1] - l[0] === l.length - 1 ? l[0] + "–" + l[l.length - 1] : l.join(", ");
  };
  function render() {
    btns.forEach((b, c) => b.setAttribute("aria-pressed", String(on.has(c))));
    const list = [...on].sort((a, b) => a - b), key = list.join(",");
    bars.innerHTML = "";
    for (const k of ["7", "6,7", "5,6,7", "4,5,6,7", "3,4,5,6,7", "2,3,4,5,6,7", "1,2,3,4,5,6,7", "0,1,2,3,4,5,6,7", "0", "0,1", "2,3,4,5", "0,1,2,3,4,5"]) {
      const v = mean(k);
      bars.append(el("div", { class: "bar" + (k === key ? "" : " alt") }, [el("span", { text: "cores " + name(k) }), el("i", { style: "--w:" + (100 * v / top) + "%" }), el("b", { text: v.toFixed(0) })]));
    }
    // The pool's caller: the fastest core by top clock (prime first), ties in the order given.
    const caller = list.length ? (list.find((c) => c >= 6) ?? list[0]) : null;
    const v = MEAS[key] ? mean(key) : 0;
    stats(sbox, [[list.length ? name(key) : "none", list.length + (list.length === 1 ? " thread" : " threads")], [caller === null ? "—" : "core " + caller, "runs the caller"], [v ? v.toFixed(0) + " tok/s" : "not measured", v ? "decode, mean of passes " + MEAS[key][0].toFixed(0) + " and " + MEAS[key][1].toFixed(0) : "decode"], [v ? (MB * v / 1000).toFixed(1) + " GB/s · " + Math.round(100 * MB * v / 1000 / ALL) + " %" : "—", "weights read · share of 67 GB/s"]]);
    note.textContent = !list.length ? "Pick at least one core." : v ? "The highlighted bar is this set. The two passes of one set can differ by 10 % or more, so small gaps between bars mean nothing. GB/s is calculated: 75.8 MB of weights read once per token." :
      "Not measured: the sweep ran the twelve sets shown as bars.";
  }
  render();
}

// ---------------------------------------------------------------- Phase 6: block quantization

// Phase 6: block quantization. Same math as src/quant.rs (quantize_q8_0,
// quantize_q4_0); the scale d is kept exact here, not rounded to f16.
function initQuantBlock() {
  const root = document.querySelector("[data-quantblock]");
  if (!root) return;
  const OUT = 0; // index of the outlier
  const fmt = root.querySelector("[data-fmt]"), rand = root.querySelector("[data-rand]");
  const outBox = root.querySelector("[data-outlier]"), wrap = root.querySelector("[data-table]");
  const scaleLine = root.querySelector("[data-scale]"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const xs = [], inputs = [], qCells = [], dCells = [], eCells = [];
  let saved = null; // value at OUT before the outlier replaced it
  const roundAway = (v) => Math.sign(v) * Math.floor(Math.abs(v) + 0.5); // f32::round: halves away from 0
  function quantize(vals, f) {
    let amax = 0, max = 0;
    if (f === "q8") {
      amax = Math.max(...vals.map(Math.abs));
      const d = amax / 127, id = d !== 0 ? 1 / d : 0;
      const q = vals.map((v) => Math.max(-127, Math.min(127, roundAway(v * id))));
      return { d, max: amax, q, deq: q.map((k) => k * d) };
    }
    // Q4_0: the element of largest magnitude (first on ties), sign kept.
    for (const v of vals) if (amax < Math.abs(v)) { amax = Math.abs(v); max = v; }
    const d = max / -8, id = d !== 0 ? 1 / d : 0;
    const q = vals.map((v) => Math.min(15, Math.floor(Math.max(0, v * id + 8.5))));
    return { d, max, q, deq: q.map((k) => (k - 8) * d) };
  }
  function setOutlier() {
    let others = 0;
    xs.forEach((v, i) => { if (i !== OUT) others = Math.max(others, Math.abs(v)); });
    xs[OUT] = 8 * others;
    inputs[OUT].value = xs[OUT].toFixed(2);
  }
  function randomise() {
    for (let i = 0; i < 32; i++) {
      xs[i] = Math.round((Math.random() * 2 - 1) * 100) / 100;
      inputs[i].value = xs[i].toFixed(2);
    }
    saved = xs[OUT];
    if (outBox.checked) setOutlier();
    render();
  }
  function render() {
    const f = fmt.value, r = quantize(xs, f), q8 = f === "q8";
    let maxErr = 0, sumErr = 0;
    for (let i = 0; i < 32; i++) {
      const e = r.deq[i] - xs[i];
      maxErr = Math.max(maxErr, Math.abs(e)); sumErr += Math.abs(e);
      qCells[i].textContent = String(r.q[i]);
      dCells[i].textContent = r.deq[i].toFixed(3);
      eCells[i].textContent = e.toFixed(3);
    }
    const bytes = q8 ? 34 : 18;
    scaleLine.textContent = q8 ? "d = max |x| ÷ 127 = " + r.d.toPrecision(3) + ", so the largest value gets q = ±127."
      : "d = max ÷ −8 = " + r.d.toPrecision(3) + ", where max = " + r.max.toFixed(2) + " is the value with the largest magnitude.";
    stats(sbox, [[r.d.toPrecision(3), "scale d"], [maxErr.toFixed(3), "max error"], [(sumErr / 32).toFixed(3), "mean abs error"], [bytes + " B", "for 32 weights (f32: 128 B)"]]);
    note.textContent = (q8 ? "Q8_0: q = round(x / d), value = q · d. 2 bytes of d plus 32 int8 values = 34 bytes."
      : "Q4_0: q = min(15, trunc(x / d + 8.5)), value = (q − 8) · d. 2 bytes of d plus 16 bytes of nibbles = 18 bytes.")
      + " d is kept exact here; the file stores it as f16, which moves each value slightly.";
  }
  // Build the table once; render() only rewrites the computed cells.
  const row = (label, cells) => el("tr", {}, [el("th", { text: label, scope: "row" }), ...cells]);
  const head = el("tr", {}, [el("th", { text: "i" }), ...Array.from({ length: 32 }, (_, i) => el("th", { text: String(i) }))]);
  for (let i = 0; i < 32; i++) {
    xs[i] = Math.round((Math.random() * 2 - 1) * 100) / 100;
    const inp = el("input", { type: "number", step: "0.01", "aria-label": "value " + i });
    inp.style.width = "4.4rem";
    inp.value = xs[i].toFixed(2);
    inp.addEventListener("input", () => { xs[i] = parseFloat(inp.value) || 0; if (i === OUT) saved = null; render(); });
    inputs.push(inp);
    qCells.push(el("td", { class: "num" }));
    dCells.push(el("td", { class: "num" }));
    eCells.push(el("td", { class: "num" }));
  }
  const body = [
    row("x", inputs.map((inp) => el("td", {}, [inp]))),
    row("q", qCells), row("x̂", dCells), row("error", eCells)
  ];
  const table = el("table", {}, [el("thead", {}, [head]), el("tbody", {}, body)]);
  wrap.style.overflowX = "auto";
  wrap.append(table);
  fmt.addEventListener("change", render);
  rand.addEventListener("click", randomise);
  outBox.addEventListener("change", () => {
    if (outBox.checked) { saved = xs[OUT]; setOutlier(); }
    else if (saved !== null) { xs[OUT] = saved; inputs[OUT].value = saved.toFixed(2); saved = null; }
    render();
  });
  render();
}

// ---------------------------------------------------------------- Phase 6: perplexity

// Phase 6: perplexity from five next-token guesses. Illustration only, not a
// measurement. Needs el() from site.js and stats() from widgets.js.

function initPplCalc() {
  const root = document.querySelector("[data-pplcalc]");
  if (!root) return;
  // "The cat sat on the mat": the five words the model had to guess, with
  // the probability it gave each one (made-up values for the example).
  const TOKENS = [
    { ctx: "The", word: "cat", p: 0.4 },
    { ctx: "The cat", word: "sat", p: 0.25 },
    { ctx: "The cat sat", word: "on", p: 0.6 },
    { ctx: "The cat sat on", word: "the", p: 0.8 },
    { ctx: "The cat sat on the", word: "mat", p: 0.1 },
  ];
  const box = root.querySelector("[data-sliders]"), sbox = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const rows = TOKENS.map((t) => {
    const r = el("input", { type: "range", min: "0.01", max: "0.99", step: "0.01", value: String(t.p), "aria-label": "probability of " + t.word });
    const out = el("span", { class: "pv" });
    box.append(el("label", { class: "field" }, [el("span", { text: "after “" + t.ctx + "”: " + t.word }), r, out]));
    return { r, out, start: t.p };
  });
  function render() {
    const p = rows.map((x) => Number(x.r.value));
    const s = p.map((v) => -Math.log(v));
    const mean = s.reduce((a, v) => a + v, 0) / s.length;
    const ppl = Math.exp(mean);
    rows.forEach((x, i) => (x.out.textContent = "p " + p[i].toFixed(2) + " → surprise " + s[i].toFixed(3)));
    stats(sbox, [[mean.toFixed(3), "mean surprise, nats"], [ppl.toFixed(2), "perplexity = exp(mean)"]]);
    const n = Math.max(1, Math.round(ppl));
    note.textContent = "Perplexity " + ppl.toFixed(2) + ": as uncertain as choosing uniformly among about " + n + " tokens. A higher p on any token lowers its surprise, and the mean and the perplexity follow.";
  }
  const reset = root.querySelector("[data-reset]");
  if (reset) reset.addEventListener("click", () => { rows.forEach((x) => (x.r.value = String(x.start))); render(); });
  root.addEventListener("input", render);
  render();
}
