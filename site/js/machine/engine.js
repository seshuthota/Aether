// The stage for start/machine.html.
//
// One SVG whose shapes are kept by key, so a number can glide from the
// matrix into memory, into a register, into a GPU thread, without being
// redrawn. Each step of the story lists the shapes it wants (show): shapes
// whose key is already on stage move to their new place, new ones fade in,
// missing ones fade out. A step can then play a short script; scrolling to
// another step cancels it.
(function () {
"use strict";

const NS = "http://www.w3.org/2000/svg";
const M = (window.M = {});

// ---------------------------------------------------------------- data

// The one example used everywhere: y = A × x.
M.A = [
  [2, 1, 3, 4, 0, 2, 1, 5],
  [1, 2, 0, 3, 2, 1, 4, 1],
  [3, 0, 2, 1, 1, 4, 2, 0],
  [1, 3, 1, 0, 2, 2, 1, 4]
];
M.X = [2, 3, 1, 2, 4, 1, 2, 3];
M.Y = M.A.map(r => r.reduce((s, v, i) => s + v * M.X[i], 0)); // 37 34 22 36
M.P0 = M.A[0].map((v, i) => v * M.X[i]);

// ---------------------------------------------------------------- shapes

const DEF = {
  box: { w: 40, h: 40, rx: 6, fs: 19, o: 1, s: 1, cls: "c-n" },
  text: { fs: 15, o: 1, s: 1, cls: "lbl", anchor: "middle" },
  line: { o: 1, cls: "" },
  frame: { o: 1, cls: "", rx: 10, x: 0, y: 0, w: 100, h: 100 },
  path: { o: 1, s: 1, cls: "", x: 0, y: 0 },
  dots: { o: 1, s: 1, cls: "", x: 0, y: 0 }
};
const LAYER = { frame: 0, path: 1, dots: 1, line: 2, box: 3, text: 4 };
const NUM = ["x", "y", "w", "h", "o", "s", "x1", "y1", "x2", "y2", "fs", "cx", "cy", "zoom"];
const STATIC = ["txt", "sub", "label", "cls", "anchor", "arrow", "d", "rx", "subBelow", "la"];

let svg, world, layers, ctlEl, readEl;
const nodes = new Map();
const tweens = new Set();
let raf = 0;
let speed = 1;
let epoch = 0;
const CANCEL = { cancelled: true };
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const T = ms => (reduced.matches ? 0 : ms / speed);
const ease = p => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);

function mk(tag, parent, attrs) {
  const e = document.createElementNS(NS, tag);
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

const KIND = {
  box: {
    make(g) {
      return { r: mk("rect", g), t: mk("text", g, { class: "v" }), sub: mk("text", g, { class: "sub" }) };
    },
    stat(n) {
      const s = n.st.txt == null ? "" : String(n.st.txt);
      n.e.t.textContent = s;
      n.e.t.setAttribute("class", /[a-z]{2,}/i.test(s) && !/\[/.test(s) ? "v word" : "v");
      n.e.sub.textContent = n.st.sub || "";
    },
    draw(n) {
      const c = n.cur, e = n.e;
      n.g.setAttribute("transform", `translate(${c.x} ${c.y}) scale(${c.s})`);
      e.r.setAttribute("x", -c.w / 2);
      e.r.setAttribute("y", -c.h / 2);
      e.r.setAttribute("width", Math.max(0, c.w));
      e.r.setAttribute("height", Math.max(0, c.h));
      e.r.setAttribute("rx", n.st.rx);
      e.t.setAttribute("y", c.fs * 0.35);
      e.t.setAttribute("font-size", c.fs);
      e.sub.setAttribute("y", n.st.subBelow ? c.h / 2 + 14 : -c.h / 2 - 6);
    }
  },
  text: {
    make(g) {
      const t = mk("text", g);
      t.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve");
      return { t };
    },
    stat(n) {
      const t = n.e.t;
      t.setAttribute("text-anchor", n.st.anchor);
      const lines = String(n.st.txt == null ? "" : n.st.txt).split("\n");
      t.textContent = "";
      lines.forEach((s, i) => {
        const sp = mk("tspan", t, { x: 0 });
        if (i) sp.setAttribute("dy", "1.25em");
        sp.textContent = s;
      });
    },
    draw(n) {
      const c = n.cur;
      n.g.setAttribute("transform", `translate(${c.x} ${c.y}) scale(${c.s})`);
      n.e.t.setAttribute("font-size", c.fs);
    }
  },
  line: {
    make(g) {
      return { l: mk("line", g), h: mk("path", g) };
    },
    stat(n) {
      n.e.h.style.display = n.st.arrow ? "" : "none";
    },
    draw(n) {
      const c = n.cur, l = n.e.l;
      let x2 = c.x2, y2 = c.y2;
      const a = Math.atan2(c.y2 - c.y1, c.x2 - c.x1);
      if (n.st.arrow) { x2 -= Math.cos(a) * 6; y2 -= Math.sin(a) * 6; }
      l.setAttribute("x1", c.x1); l.setAttribute("y1", c.y1);
      l.setAttribute("x2", x2); l.setAttribute("y2", y2);
      if (n.st.arrow) {
        const s = 8, p = (d, r) => `${c.x2 + Math.cos(a + d) * r} ${c.y2 + Math.sin(a + d) * r}`;
        n.e.h.setAttribute("d", `M${c.x2} ${c.y2} L${p(Math.PI - 0.42, s)} L${p(Math.PI + 0.42, s)} Z`);
      }
    }
  },
  frame: {
    make(g) {
      return { r: mk("rect", g), t: mk("text", g) };
    },
    stat(n) {
      n.e.t.textContent = (n.st.label || "").toUpperCase();
    },
    draw(n) {
      const c = n.cur, e = n.e;
      e.r.setAttribute("x", c.x); e.r.setAttribute("y", c.y);
      e.r.setAttribute("width", Math.max(0, c.w)); e.r.setAttribute("height", Math.max(0, c.h));
      e.r.setAttribute("rx", n.st.rx);
      e.t.setAttribute("x", c.x + (n.st.la ? 2 : 12));
      e.t.setAttribute("y", n.st.la ? c.y - 8 : c.y + 19);
    }
  },
  path: {
    make(g) {
      return { p: mk("path", g) };
    },
    stat(n) {
      n.e.p.setAttribute("d", n.st.d || "");
    },
    draw(n) {
      const c = n.cur;
      n.g.setAttribute("transform", `translate(${c.x} ${c.y}) scale(${c.s})`);
    }
  }
};
KIND.dots = KIND.path;

function make(kind, key, layer) {
  const g = mk("g", layers[layer]);
  const n = { k: key, t: kind, g, cur: {}, st: {} };
  n.e = KIND[kind].make(g);
  nodes.set(key, n);
  return n;
}

function applyStatic(n) {
  n.g.setAttribute("class", `node ${n.t} ${n.st.cls || ""}`);
  KIND[n.t].stat(n);
}

function draw(n) {
  if (n === cam) {
    const c = cam.cur;
    world.setAttribute("transform", `translate(400 280) scale(${c.zoom}) translate(${-c.cx} ${-c.cy})`);
    return;
  }
  n.g.setAttribute("opacity", Math.max(0, Math.min(1, n.cur.o)));
  KIND[n.t].draw(n);
}

function numeric(p) {
  const o = {};
  for (const k of NUM) if (typeof p[k] === "number") o[k] = p[k];
  return o;
}

function tick(now) {
  for (const t of tweens) {
    const p = Math.min(1, (now - t.t0) / t.ms);
    const e = t.lin ? p : ease(p);
    for (const k in t.to) t.n.cur[k] = t.from[k] + (t.to[k] - t.from[k]) * e;
    draw(t.n);
    if (p >= 1) { tweens.delete(t); t.res(); }
  }
  raf = tweens.size ? requestAnimationFrame(tick) : 0;
}

function tween(n, to, ms, lin) {
  ms = T(ms);
  for (const t of tweens) if (t.n === n) for (const k in to) delete t.to[k];
  const from = {};
  let moved = false;
  for (const k in to) {
    from[k] = n.cur[k] == null ? to[k] : n.cur[k];
    if (from[k] !== to[k]) moved = true;
  }
  if (!moved || ms <= 0) {
    Object.assign(n.cur, to);
    draw(n);
    return Promise.resolve();
  }
  return new Promise(res => {
    tweens.add({ n, from, to: Object.assign({}, to), t0: performance.now(), ms, res, lin });
    if (!raf) raf = requestAnimationFrame(tick);
  });
}

function upsert(sp, ms) {
  let n = nodes.get(sp.k);
  const tgt = Object.assign({}, DEF[sp.t], sp);
  if (n && n.t !== sp.t) { n.g.remove(); nodes.delete(sp.k); n = null; }
  if (!n) {
    n = make(sp.t, sp.k, sp.lay == null ? LAYER[sp.t] : sp.lay);
    const start = Object.assign({}, tgt, { o: 0 });
    if (sp.from != null) {
      const src = typeof sp.from === "string" ? (nodes.get(sp.from) || {}).cur : sp.from;
      if (src) for (const p of ["x", "y", "s", "w", "h", "fs"]) if (typeof src[p] === "number") start[p] = src[p];
      start.o = sp.fromO != null ? sp.fromO : typeof sp.from === "string" ? 1 : 0;
    }
    n.cur = start;
  }
  n.dying = false;
  n.st = tgt;
  applyStatic(n);
  draw(n);
  return tween(n, numeric(tgt), ms);
}

function remove(n, ms) {
  n.dying = true;
  return tween(n, { o: 0 }, ms).then(() => {
    if (n.dying) { n.g.remove(); nodes.delete(n.k); }
  });
}

const cam = { k: "cam", cur: { cx: 400, cy: 280, zoom: 1 } };

function flat(list, out = []) {
  for (const x of list) {
    if (Array.isArray(x)) flat(x, out);
    else if (x) out.push(x);
  }
  return out;
}

function show(list, ms = 750) {
  const want = new Set();
  const ps = [];
  let c = { cx: 400, cy: 280, zoom: 1 };
  for (const sp of flat(list)) {
    if (sp.t === "cam") { c = numeric(sp); continue; }
    want.add(sp.k);
    ps.push(upsert(sp, ms));
  }
  ps.push(tween(cam, c, ms));
  for (const [k, n] of nodes) if (!want.has(k) && !n.dying) ps.push(remove(n, ms * 0.6));
  return Promise.all(ps);
}

// ---------------------------------------------------------------- builders

M.box = (k, x, y, txt, cls = "c-n", o = {}) => Object.assign({ k, t: "box", x, y, txt, cls }, o);
M.txt = (k, x, y, txt, o = {}) => Object.assign({ k, t: "text", x, y, txt }, o);
M.line = (k, x1, y1, x2, y2, o = {}) => Object.assign({ k, t: "line", x1, y1, x2, y2 }, o);
M.arrow = (k, x1, y1, x2, y2, o = {}) => Object.assign({ k, t: "line", x1, y1, x2, y2, arrow: true }, o);
M.frame = (k, x, y, w, h, label = "", o = {}) => Object.assign({ k, t: "frame", x, y, w, h, label }, o);
M.path = (k, d, o = {}) => Object.assign({ k, t: "path", d }, o);
M.cam = (cx, cy, zoom) => ({ t: "cam", cx, cy, zoom });

// A grid of small circles as one path (thousands of GPU threads).
M.dotsD = (cols, rows, gap, r, skip) => {
  let d = "";
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    if (skip && skip(i, j)) continue;
    const x = i * gap, y = j * gap;
    d += `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
  }
  return d;
};

// A code listing: a dark panel, one text per line, and a highlight bar.
// Returns shapes; M.codeY(geom, line) gives the bar's y for a line.
M.code = (k, x, y, lines, hl = -1, o = {}) => {
  const lh = o.lh || 19, fs = o.fs || 12.5, w = o.w || 380;
  const out = [M.frame(k + "_bg", x - 12, y - 18, w, lines.length * lh + 16, o.title || "", { cls: "codebg", la: 1, rx: 8 })];
  out.push(M.box(k + "_hl", x - 12 + w / 2, y - fs * 0.35 + Math.max(0, hl) * lh, "", "codehl", { w: w - 8, h: lh, rx: 3, o: hl >= 0 ? 1 : 0 }));
  lines.forEach((s, i) => out.push(M.txt(`${k}_${i}`, x, y + i * lh, s, { fs, anchor: "start", cls: "code" })));
  return out;
};
M.codeTo = (a, k, x, y, line, o = {}) => {
  const lh = o.lh || 19, fs = o.fs || 12.5;
  return a.to(k + "_hl", { y: y - fs * 0.35 + line * lh, o: line < 0 ? 0 : 1 }, 220);
};

// ---------------------------------------------------------------- scripts

function api(my) {
  const ok = v => {
    if (my !== epoch) throw CANCEL;
    return v;
  };
  const a = {
    live: () => my === epoch,
    show: (l, ms) => show(l, ms).then(ok),
    add: (sp, ms = 400) => upsert(sp, ms).then(ok),
    put: sp => { upsert(sp, 0); },
    to(k, p, ms = 450, lin = false) {
      const n = nodes.get(k);
      if (!n) return Promise.resolve().then(ok);
      let changed = false;
      for (const s of STATIC) if (s in p) { n.st[s] = p[s]; changed = true; }
      if (changed) applyStatic(n);
      return tween(n, numeric(p), ms, lin).then(ok);
    },
    del(k, ms = 300) {
      const n = nodes.get(k);
      return (n ? remove(n, ms) : Promise.resolve()).then(ok);
    },
    wait: ms => new Promise(r => setTimeout(r, T(ms))).then(ok),
    all: ps => Promise.all(ps).then(ok),
    pulse(k, s = 1.18, ms = 170) {
      return a.to(k, { s }, ms).then(() => a.to(k, { s: 1 }, ms));
    },
    get: k => (nodes.get(k) || {}).cur,
    read(html) { readEl.innerHTML = html; },
    // Move a copy of shape `src` to (x, y); the copy keeps key `k`.
    async fly(k, src, x, y, ms = 600, o = {}) {
      const s = nodes.get(src);
      if (!s) return;
      upsert(Object.assign({}, s.st, { k, from: src, x, y, sub: "" }, o), 0);
      const n = nodes.get(k);
      Object.assign(n.cur, { x: s.cur.x, y: s.cur.y, o: 1 });
      draw(n);
      return tween(n, numeric(Object.assign({ x, y }, o)), ms).then(ok);
    }
  };
  return a;
}

// ---------------------------------------------------------------- scenes

const scenes = {};
M._scene = id => scenes[id];
M.scene = (id, def) => {
  scenes[id] = Object.assign({ st: Object.assign({}, def.init || {}) }, def);
};

let active = null;   // the step element
let current = null;  // its scene

function enter(sc, ms = 750) {
  const my = ++epoch;
  const a = api(my);
  readEl.innerHTML = "";
  const built = sc.build ? sc.build(sc.st) : [];
  M.done = false;
  const p = show(built, ms).then(() => {
    if (my !== epoch) return;
    if (sc.after) sc.after(a, sc.st);
    if (sc.play) return sc.play(a, sc.st);
  }).then(() => { if (my === epoch) M.done = true; });
  p.catch(e => { if (e !== CANCEL) console.error(e); });
}

function renderControls(sc) {
  ctlEl.innerHTML = "";
  const st = sc.st;
  if (sc.play) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "replay";
    b.textContent = "↻ Replay";
    b.addEventListener("click", () => enter(sc, 350));
    ctlEl.appendChild(b);
  }
  for (const c of sc.controls || []) {
    const wrap = document.createElement("div");
    wrap.className = "ctl-group";
    if (c.label) {
      const l = document.createElement("span");
      l.className = "ctl-label";
      l.textContent = c.label;
      wrap.appendChild(l);
    }
    if (c.type === "range") {
      const r = document.createElement("input");
      r.type = "range";
      r.min = c.min; r.max = c.max; r.step = c.step || 1;
      r.value = st[c.key];
      r.setAttribute("aria-label", c.label || c.key);
      const out = document.createElement("output");
      const fmt = c.fmt || (v => v);
      out.textContent = fmt(+r.value, st);
      r.addEventListener("input", () => {
        st[c.key] = +r.value;
        out.textContent = fmt(+r.value, st);
        if (c.live) c.live(api(epoch), st);
        else enter(sc, 300);
      });
      wrap.appendChild(r);
      wrap.appendChild(out);
    } else {
      const seg = document.createElement("div");
      seg.className = "seg";
      seg.setAttribute("role", "group");
      for (const [v, label] of c.opts) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.setAttribute("aria-pressed", String(st[c.key] === v));
        b.addEventListener("click", () => {
          if (c.action) c.action(v, st);
          else st[c.key] = v;
          for (const o of seg.children) o.setAttribute("aria-pressed", String(o === b));
          enter(sc, 450);
        });
        seg.appendChild(b);
      }
      wrap.appendChild(seg);
    }
    ctlEl.appendChild(wrap);
  }
  for (const btn of sc.buttons || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "act-btn";
    b.textContent = btn.label;
    b.addEventListener("click", () => {
      const my = ++epoch;
      const a = api(my);
      Promise.resolve(btn.run(a, st)).catch(e => { if (e !== CANCEL) console.error(e); });
    });
    ctlEl.appendChild(b);
  }
}

// ---------------------------------------------------------------- the page

M.start = () => {
  const stage = document.querySelector("[data-stage]");
  svg = stage.querySelector("svg");
  world = mk("g", svg, { class: "world" });
  layers = [0, 1, 2, 3, 4].map(() => mk("g", world));
  draw(cam);
  ctlEl = stage.querySelector("[data-ctl]");
  readEl = stage.querySelector("[data-readout]");
  const hud = stage.querySelectorAll("[data-hud] dd");
  const actEl = stage.querySelector("[data-act]");
  const bar = stage.querySelector("[data-progress]");
  const steps = [...document.querySelectorAll("[data-scene]")];

  let act = "";
  for (const el of document.querySelectorAll(".act, [data-scene]")) {
    if (el.classList.contains("act")) act = el.textContent.trim();
    else el.dataset.act = act;
  }

  function activate(step) {
    if (step === active) return;
    if (active) active.classList.remove("is-active");
    active = step;
    step.classList.add("is-active");
    const sc = scenes[step.dataset.scene];
    if (!sc) { console.warn("no scene", step.dataset.scene); return; }
    current = sc;
    const h = (step.dataset.hud || "").split("|");
    hud.forEach((dd, i) => { dd.textContent = (h[i] || "—").trim(); });
    actEl.textContent = step.dataset.act || "";
    const i = steps.indexOf(step);
    bar.style.width = ((i + 1) / steps.length) * 100 + "%";
    renderControls(sc);
    enter(sc);
  }

  const narrow = matchMedia("(max-width: 900px)");
  function pick() {
    const line = innerHeight * (narrow.matches ? 0.78 : 0.5);
    let cur = steps[0];
    for (const s of steps) if (s.getBoundingClientRect().top < line) cur = s;
    activate(cur);
  }
  let queued = false;
  addEventListener("scroll", () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; pick(); });
  }, { passive: true });
  addEventListener("resize", pick);

  function go(d) {
    const i = Math.max(0, Math.min(steps.length - 1, steps.indexOf(active) + d));
    steps[i].scrollIntoView({ behavior: reduced.matches ? "auto" : "smooth", block: narrow.matches ? "start" : "center" });
    activate(steps[i]);
  }
  stage.querySelector("[data-prev]").addEventListener("click", () => go(-1));
  stage.querySelector("[data-next]").addEventListener("click", () => go(1));
  addEventListener("keydown", e => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowRight" || e.key === "j") { go(1); e.preventDefault(); }
    if (e.key === "ArrowLeft" || e.key === "k") { go(-1); e.preventDefault(); }
  });
  for (const b of stage.querySelectorAll("[data-speed] button")) {
    b.addEventListener("click", () => {
      speed = +b.dataset.v;
      for (const o of b.parentNode.children) o.setAttribute("aria-pressed", String(o === b));
    });
  }
  pick();
};
})();
