// Shared behaviour for every page: theme, navigation built from pages.js,
// glossary hover cards from glossary.js, the "I can explain this" checkbox,
// and the interactive widgets. Each widget init returns early if its
// element is not on the current page.

const SITE_ROOT = new URL("../", document.currentScript.src);

const store = {
  get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private window */ } },
  remove(key) { try { localStorage.removeItem(key); } catch (e) { /* private window */ } }
};

(function () {
  initTheme();
  const nav = initNav();
  initUnderstood(nav);
  initGlossaryPage();
  initTerms();
  initStride();
  initCeiling();
  initMedian();
  initRoof();
  initRoofPick();
  initMatvec();
  initGenLoop();
  initRoadmap();
  initAlignSplit();
  initAlignCheck();
  initClockCost();
  initLatency();
  initFit();
  initLayout();
  initCpuMask();
  initSampler();
  initUnits();
  initAxis();
})();

// ---------------------------------------------------------------- helpers

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children || []) node.append(c);
  return node;
}

function pageKey(url) {
  const u = new URL(url, location.href);
  let p = u.pathname;
  if (p.endsWith("/")) p += "index.html";
  return p;
}

function fmtBytes(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(b >= 1e8 ? 0 : 1) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(0) + " KB";
  return b + " B";
}

// ---------------------------------------------------------------- theme

function initTheme() {
  const root = document.documentElement;
  const key = "aether-notes-theme";
  const stored = store.get(key);
  if (stored === "light" || stored === "dark") root.dataset.theme = stored;
  const btn = document.querySelector("[data-theme-toggle]");
  function label() {
    if (!btn) return;
    const mode = root.dataset.theme || "auto";
    btn.textContent = mode === "dark" ? "Dark" : mode === "light" ? "Light" : "Auto";
  }
  label();
  btn && btn.addEventListener("click", () => {
    const cur = root.dataset.theme;
    const next = cur === "dark" ? "light" : cur === "light" ? "" : "dark";
    if (next) { root.dataset.theme = next; store.set(key, next); }
    else { delete root.dataset.theme; store.remove(key); }
    label();
  });
}

// ---------------------------------------------------------------- navigation

function initNav() {
  const sections = window.AETHER_PAGES || [];
  const here = pageKey(location.href);
  const flat = [];
  for (const s of sections) for (const p of s.pages) {
    flat.push({ ...p, section: s.section, crumb: s.crumb, url: new URL(p.href, SITE_ROOT).href });
  }
  const idx = flat.findIndex((p) => pageKey(p.url) === here);
  const cur = flat[idx];

  const side = document.querySelector("[data-nav]");
  if (side) {
    for (const s of sections) {
      side.append(el("p", { class: "nav-label", text: s.section }));
      for (const p of s.pages) {
        const url = new URL(p.href, SITE_ROOT).href;
        const a = el("a", { href: url, "data-key": pageKey(url) });
        if (p.num) a.append(el("span", { text: p.num }));
        a.append(document.createTextNode(p.title));
        if (pageKey(url) === here) a.setAttribute("aria-current", "page");
        side.append(a);
      }
    }
    // Phone layout: the nav becomes a drawer opened from the header. It starts
    // with this page's own sections (copied from its .toc), then every page.
    const top = document.querySelector(".top");
    const theme = top && top.querySelector("[data-theme-toggle]");
    const pageToc = document.querySelector("main .toc");
    if (pageToc) {
      const links = [...pageToc.querySelectorAll("a[href^='#']")];
      if (links.length) {
        const first = side.firstChild;
        const group = [el("p", { class: "nav-label onpage", text: "On this page" })];
        for (const l of links) group.push(el("a", { class: "onpage", href: l.getAttribute("href"), text: l.textContent }));
        group.push(el("p", { class: "nav-label onpage", text: "All pages" }));
        for (const n of group) side.insertBefore(n, first);
      }
    }
    if (top && theme) {
      const btn = el("button", { class: "navtoggle", type: "button", "aria-expanded": "false", "aria-controls": "site-nav", "aria-label": "Menu" });
      btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d="M3 5.5h14M3 10h14M3 14.5h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
      side.id = "site-nav";
      const scrim = el("div", { class: "scrim" });
      document.body.append(scrim);
      top.prepend(btn);
      const setOpen = (open) => {
        side.classList.toggle("open", open);
        scrim.classList.toggle("open", open);
        document.body.classList.toggle("nav-locked", open);
        btn.setAttribute("aria-expanded", String(open));
      };
      // Wide screens: the same button hides or shows the menu column, remembered per browser.
      const wide = window.matchMedia("(min-width: 861px)");
      const collapsedKey = "aether-nav-collapsed";
      const setCollapsed = (c) => {
        document.documentElement.classList.toggle("nav-collapsed", c);
        if (c) store.set(collapsedKey, "1"); else store.remove(collapsedKey);
        if (wide.matches) btn.setAttribute("aria-expanded", String(!c));
      };
      if (store.get(collapsedKey)) setCollapsed(true);
      btn.addEventListener("click", () => {
        if (wide.matches) setCollapsed(!document.documentElement.classList.contains("nav-collapsed"));
        else setOpen(!side.classList.contains("open"));
      });
      scrim.addEventListener("click", () => setOpen(false));
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") setOpen(false); });
      side.addEventListener("click", (e) => {
        const a = e.target.closest("a");
        if (!a) return;
        // Unlock scrolling before jumping to a section, or the jump is ignored.
        const hash = a.getAttribute("href");
        const target = hash && hash.startsWith("#") && document.getElementById(hash.slice(1));
        if (target) {
          e.preventDefault();
          setOpen(false);
          history.replaceState(null, "", hash);
          target.scrollIntoView();
        } else setOpen(false);
      });
      window.matchMedia("(min-width: 861px)").addEventListener("change", (e) => { if (e.matches) setOpen(false); });
    }
  }

  const crumb = document.querySelector("[data-crumb]");
  if (crumb && cur) {
    crumb.innerHTML = "";
    if (cur.href === "index.html") crumb.textContent = "map";
    else {
      crumb.append(el("a", { href: new URL("index.html", SITE_ROOT).href, text: "notes" }), " / ");
      const first = sections.find((s) => s.crumb === cur.crumb).pages[0];
      if (first.href !== cur.href && first.href !== "index.html") {
        crumb.append(el("a", { href: new URL(first.href, SITE_ROOT).href, text: cur.crumb }));
      } else crumb.append(cur.crumb);
    }
  }

  const pager = document.querySelector("[data-pager]");
  if (pager && idx >= 0) {
    const prev = flat[idx - 1], next = flat[idx + 1];
    if (prev) pager.append(el("a", { href: prev.url }, [el("small", { text: prev.section === cur.section ? "Previous" : prev.crumb }), prev.title]));
    if (next) pager.append(el("a", { class: "next", href: next.url }, [el("small", { text: next.section === cur.section ? "Next" : next.crumb }), next.title]));
  }

  for (const box of document.querySelectorAll("[data-cards]")) {
    const s = sections.find((x) => x.section === box.dataset.cards);
    if (!s) continue;
    for (const p of s.pages) {
      if (!p.desc) continue;
      const url = new URL(p.href, SITE_ROOT).href;
      box.append(el("a", { class: "card", href: url, "data-key": pageKey(url) }, [
        el("span", { class: "num", text: p.num || "" }),
        el("h3", { text: p.title }),
        el("p", { text: p.desc })
      ]));
    }
  }
  return { cur };
}

// ---------------------------------------------------------------- progress

function initUnderstood(nav) {
  const key = "aether-understood";
  let done = {};
  try { done = JSON.parse(store.get(key) || "{}") || {}; } catch (e) { done = {}; }
  function paint() {
    for (const a of document.querySelectorAll("[data-key]")) a.classList.toggle("done", !!done[a.dataset.key]);
  }
  const pager = document.querySelector("[data-pager]");
  const cur = nav.cur;
  if (pager && cur && cur.href !== "index.html" && cur.href !== "glossary.html") {
    const k = pageKey(cur.url);
    const box = el("input", { type: "checkbox" });
    box.checked = !!done[k];
    box.addEventListener("change", () => {
      if (box.checked) done[k] = true; else delete done[k];
      store.set(key, JSON.stringify(done));
      paint();
    });
    const tick = el("label", { class: "understood" }, [box, el("span", { text: "I can explain this page in my own words. (Ticks show in the menu. Saved in this browser only.)" })]);
    const slot = document.querySelector("[data-understood]");
    if (slot) slot.append(tick); else pager.before(tick);
  }
  paint();
}

// ---------------------------------------------------------------- glossary

function glossaryEntries() {
  return (window.AETHER_GLOSSARY || []).map((g, i) => ({ ...g, id: "g-" + g.term.toLowerCase().replace(/[^a-z0-9]+/g, "-"), i }));
}

function initGlossaryPage() {
  const list = document.querySelector("[data-glossary-list]");
  if (!list) return;
  const entries = glossaryEntries().sort((a, b) => a.term.localeCompare(b.term, "en", { sensitivity: "base" }));
  const filter = document.querySelector("[data-glossary-filter]");
  const count = document.querySelector("[data-glossary-count]");
  function render() {
    const q = (filter && filter.value.trim().toLowerCase()) || "";
    list.innerHTML = "";
    let n = 0, letter = "";
    for (const g of entries) {
      const hay = (g.term + " " + g.match.join(" ") + " " + g.def).toLowerCase();
      if (q && !hay.includes(q)) continue;
      n++;
      const L = g.term[0].toUpperCase();
      if (L !== letter && !q) { letter = L; list.append(el("h2", { class: "gl-letter", text: L })); }
      const dd = el("dd", { text: g.def + " " });
      if (g.see) dd.append(el("a", { href: new URL(g.see, SITE_ROOT).href, text: "Read more →" }));
      list.append(el("dt", { id: g.id, text: g.term }), dd);
    }
    if (count) count.textContent = n + " of " + entries.length + " terms";
  }
  filter && filter.addEventListener("input", render);
  render();
}

function initTerms() {
  const main = document.querySelector("main");
  if (!main || main.hasAttribute("data-no-glossary")) return;
  const entries = glossaryEntries();
  if (!entries.length) return;
  const skip = new Set((main.dataset.noTerms || "").split(",").map((s) => s.trim()).filter(Boolean));
  const byVariant = new Map();
  for (const g of entries) {
    if (skip.has(g.term)) continue;
    for (const v of g.match) {
      byVariant.set(v, g);
      const cap = v[0].toUpperCase() + v.slice(1);
      if (!byVariant.has(cap)) byVariant.set(cap, g);
    }
  }
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const variants = [...byVariant.keys()].sort((a, b) => b.length - a.length);
  const re = new RegExp("(?<![\\w-])(" + variants.map(esc).join("|") + ")(?![\\w])", "g");
  const excluded = "code, pre, a, h1, h2, h3, svg, .widget, .kicker, .toc, .pager, .tag, .lbl, .stat, th, .term, .formula, .understood, button, label, summary, .pull";
  const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement.closest(excluded) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT)
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  const used = new Set();
  for (let node of nodes) {
    re.lastIndex = 0;
    let m;
    while (node && (m = re.exec(node.nodeValue))) {
      const g = byVariant.get(m[1]);
      if (used.has(g.i)) continue;
      used.add(g.i);
      const mid = node.splitText(m.index);
      const after = mid.splitText(m[1].length);
      const span = el("span", { class: "term", tabindex: "0", "data-g": String(g.i), text: mid.nodeValue });
      mid.replaceWith(span);
      node = after;
      re.lastIndex = 0;
    }
  }
  initTip(entries);
}

function initTip(entries) {
  const tip = el("div", { class: "tip", role: "tooltip", hidden: "" });
  document.body.append(tip);
  let owner = null, hideTimer = 0;
  function show(term) {
    clearTimeout(hideTimer);
    const g = entries[Number(term.dataset.g)];
    tip.innerHTML = "";
    tip.append(el("b", { text: g.term }), el("span", { text: " " + g.def + " " }));
    const href = g.see ? new URL(g.see, SITE_ROOT).href : new URL("glossary.html#" + g.id, SITE_ROOT).href;
    tip.append(el("a", { href, text: g.see ? "Read more →" : "Glossary →" }));
    tip.hidden = false;
    const r = term.getBoundingClientRect();
    const w = Math.min(340, window.innerWidth - 32);
    tip.style.width = w + "px";
    let left = r.left + window.scrollX;
    left = Math.max(16 + window.scrollX, Math.min(left, window.scrollX + window.innerWidth - w - 16));
    tip.style.left = left + "px";
    tip.style.top = r.bottom + window.scrollY + 6 + "px";
    owner = term;
  }
  function hideSoon() { clearTimeout(hideTimer); hideTimer = setTimeout(() => { tip.hidden = true; owner = null; }, 160); }
  document.addEventListener("mouseover", (e) => { const t = e.target.closest(".term"); if (t) show(t); });
  document.addEventListener("mouseout", (e) => { if (e.target.closest(".term")) hideSoon(); });
  document.addEventListener("focusin", (e) => { const t = e.target.closest(".term"); if (t) show(t); });
  document.addEventListener("focusout", (e) => { if (e.target.closest(".term")) hideSoon(); });
  tip.addEventListener("mouseover", () => clearTimeout(hideTimer));
  tip.addEventListener("mouseout", hideSoon);
  document.addEventListener("click", (e) => {
    const t = e.target.closest(".term");
    if (t) { if (owner === t && !tip.hidden) { tip.hidden = true; owner = null; } else show(t); return; }
    if (!e.target.closest(".tip")) { tip.hidden = true; owner = null; }
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { tip.hidden = true; owner = null; } });
}

// ---------------------------------------------------------------- tensors: stride lab

function initStride() {
  const root = document.querySelector("[data-stride]");
  if (!root) return;
  const out = root.querySelector("[data-out]");
  const grid = root.querySelector("[data-grid]");
  const line = root.querySelector("[data-line]");
  const tr = root.querySelector("[data-transpose]");
  const read = (sel) => [...root.querySelectorAll(sel)].map((i) => Number(i.value));
  function render() {
    const shape = read("[data-shape]");
    const pos = read("[data-pos]");
    grid.innerHTML = "";
    line.innerHTML = "";
    if (shape.some((n) => !Number.isInteger(n) || n < 1 || n > 12)) {
      out.textContent = "Rows and cols need to be whole numbers from 1 to 12.";
      return;
    }
    const strides = [shape[1], 1];
    const t = tr && tr.checked;
    const vshape = t ? [shape[1], shape[0]] : shape;
    const vstrides = t ? [strides[1], strides[0]] : strides;
    const numel = shape[0] * shape[1];
    const k = pos.findIndex((p, i) => !Number.isInteger(p) || p < 0 || p >= vshape[i]);
    const flat = pos[0] * vstrides[0] + pos[1] * vstrides[1];
    const head = (t ? "transposed view: shape [" + vshape.join(", ") + "], strides [" + vstrides.join(", ") + "], same buffer, nothing copied"
                    : "shape [" + shape.join(", ") + "], strides [" + strides.join(", ") + "], numel " + numel);
    out.innerHTML = head + "<br>" + (k >= 0
      ? "index " + pos[k] + " out of range for dim " + k + " (size " + vshape[k] + "). The Rust code panics here."
      : "flat = " + pos[0] + "×" + vstrides[0] + " + " + pos[1] + "×" + vstrides[1] + " = " + flat);
    grid.style.gridTemplateColumns = "repeat(" + vshape[1] + ", minmax(2.2rem, 1fr))";
    for (let r = 0; r < vshape[0]; r++) {
      for (let c = 0; c < vshape[1]; c++) {
        const idx = r * vstrides[0] + c * vstrides[1];
        const b = el("button", { type: "button", text: String(idx), title: "(" + r + ", " + c + ") → buffer slot " + idx });
        if (k < 0 && r === pos[0] && c === pos[1]) b.className = "on";
        b.addEventListener("click", () => {
          const inputs = root.querySelectorAll("[data-pos]");
          inputs[0].value = r; inputs[1].value = c; render();
        });
        grid.append(b);
      }
    }
    line.style.gridTemplateColumns = "repeat(" + numel + ", minmax(1.5rem, 1fr))";
    for (let i = 0; i < numel; i++) line.append(el("span", { class: k < 0 && i === flat ? "on" : "", text: String(i) }));
  }
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- decode ceiling lab

function initCeiling() {
  const root = document.querySelector("[data-ceiling]");
  if (!root) return;
  const bw = root.querySelector("[data-bw]");
  const label = root.querySelector("[data-bw-label]");
  const model = root.querySelector("[data-model]");
  const table = root.querySelector("[data-table]");
  const formats = [
    ["f32", 4, "4 bytes"],
    ["bf16", 2, "2 bytes"],
    ["Q8_0", 34 / 32, "8.5 bits"],
    ["Q4_0", 18 / 32, "4.5 bits"],
    ["4-bit ideal", 0.5, "4 bits"]
  ];
  function render() {
    const gbs = Number(bw.value);
    const params = Number(model.value);
    label.textContent = gbs.toFixed(0) + " GB/s";
    const rows = formats.map(([name, bpp, note]) => {
      const bytes = params * bpp;
      return { name, note, bytes, tps: (gbs * 1e9) / bytes };
    });
    const max = Math.max(...rows.map((r) => r.tps));
    table.innerHTML = "";
    for (const r of rows) {
      const bar = el("div", { class: "minibar" }, [el("i", { style: "width:" + (100 * r.tps / max).toFixed(1) + "%" })]);
      table.append(el("tr", {}, [
        el("td", { text: r.name }), el("td", { text: r.note }), el("td", { text: fmtBytes(r.bytes) }),
        el("td", { class: "num", text: r.tps >= 100 ? r.tps.toFixed(0) : r.tps.toFixed(1) }),
        el("td", { class: "barcell" }, [bar])
      ]));
    }
  }
  bw.addEventListener("input", render);
  model.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- median widget

function initMedian() {
  const root = document.querySelector("[data-median]");
  if (!root) return;
  const slider = root.querySelector("[data-outlier]");
  const out = root.querySelector("[data-out]");
  const svg = root.querySelector("svg");
  function render() {
    const outlier = Number(slider.value);
    const xs = [10, 10, 10, 10, outlier].sort((a, b) => a - b);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    out.innerHTML = "samples [10, 10, 10, 10, " + outlier + "] ms<br>median " + xs[2].toFixed(1) + " ms · mean " + mean.toFixed(1) + " ms";
    if (!svg) return;
    const W = 640, L = 30, R = 20, x = (v) => L + (v - 0) / 110 * (W - L - R);
    let s = '<line x1="' + L + '" y1="40" x2="' + (W - R) + '" y2="40" stroke="var(--line)" stroke-width="2"/>';
    for (const t of [0, 25, 50, 75, 100]) s += '<text x="' + x(t) + '" y="96" text-anchor="middle" class="muted" font-size="11">' + t + " ms</text>";
    [10, 10, 10, 10].forEach((v, i) => { s += '<circle cx="' + x(v) + '" cy="' + (40 - 8 * i + 12) + '" r="5" fill="var(--blue)"/>'; });
    s += '<circle cx="' + x(outlier) + '" cy="40" r="6" fill="var(--copper)"/>';
    s += '<line x1="' + x(xs[2]) + '" y1="14" x2="' + x(xs[2]) + '" y2="66" stroke="var(--accent)" stroke-width="2.5"/><text x="' + x(xs[2]) + '" y="10" text-anchor="middle" font-size="11" fill="var(--accent)">median</text>';
    s += '<line x1="' + x(mean) + '" y1="20" x2="' + x(mean) + '" y2="66" stroke="var(--copper)" stroke-width="2" stroke-dasharray="4 3"/><text x="' + x(mean) + '" y="78" text-anchor="middle" font-size="11" fill="var(--copper)">mean</text>';
    svg.innerHTML = s;
  }
  slider.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- roofline

function initRoof() {
  const root = document.querySelector("[data-roof]");
  if (!root) return;
  const bw = root.querySelector("[data-bw]");
  const peak = root.querySelector("[data-peak]");
  const bwLabel = root.querySelector("[data-bw-label]");
  const peakLabel = root.querySelector("[data-peak-label]");
  const out = root.querySelector("[data-out]");
  const svg = root.querySelector("svg");
  const points = [
    { name: "Phase 0 sum (measured)", ai: 0.25, y: 0.82, measured: true },
    { name: "f32 GEMV (decode)", ai: 0.5 },
    { name: "Q4_0 GEMV, ideal", ai: 2 / (18 / 32) },
    { name: "prefill, 32 tokens", ai: 16 }
  ];
  function render() {
    const B = Number(bw.value), P = Number(peak.value);
    bwLabel.textContent = B + " GB/s";
    peakLabel.textContent = P + " GFLOP/s";
    const ridge = P / B;
    const W = 640, H = 300, L = 54, R = 16, T = 14, Bm = 40;
    const x0 = 0.05, x1 = 64, y0 = 0.1, y1 = 500;
    const lx = (v) => L + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - L - R);
    const ly = (v) => T + (1 - (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (H - T - Bm);
    const roof = (ai) => Math.min(P, ai * B);
    let s = "";
    for (const t of [0.1, 1, 10]) s += '<line x1="' + lx(t) + '" y1="' + T + '" x2="' + lx(t) + '" y2="' + (H - Bm) + '" stroke="var(--line)" stroke-dasharray="2 4"/><text x="' + lx(t) + '" y="' + (H - Bm + 16) + '" text-anchor="middle" class="muted" font-size="11">' + t + "</text>";
    for (const t of [0.1, 1, 10, 100]) s += '<line x1="' + L + '" y1="' + ly(t) + '" x2="' + (W - R) + '" y2="' + ly(t) + '" stroke="var(--line)" stroke-dasharray="2 4"/><text x="' + (L - 6) + '" y="' + (ly(t) + 4) + '" text-anchor="end" class="muted" font-size="11">' + t + "</text>";
    s += '<text x="' + ((L + W - R) / 2) + '" y="' + (H - 6) + '" text-anchor="middle" class="muted" font-size="12">arithmetic intensity, FLOP per byte (log scale)</text>';
    s += '<text x="12" y="' + ((T + H - Bm) / 2) + '" transform="rotate(-90 12 ' + ((T + H - Bm) / 2) + ')" text-anchor="middle" class="muted" font-size="12">GFLOP/s (log)</text>';
    const ra = Math.min(Math.max(ridge, x0), x1);
    s += '<polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="' + lx(x0) + "," + ly(Math.max(roof(x0), y0)) + " " + lx(ra) + "," + ly(roof(ra)) + " " + lx(x1) + "," + ly(roof(x1)) + '"/>';
    s += '<line x1="' + lx(ra) + '" y1="' + ly(roof(ra)) + '" x2="' + lx(ra) + '" y2="' + (H - Bm) + '" stroke="var(--accent)" stroke-dasharray="4 4"/>';
    const nearRight = lx(ra) > W - R - 90;
    s += '<text x="' + (lx(ra) + (nearRight ? -6 : 6)) + '" y="' + (H - Bm - 8) + '" text-anchor="' + (nearRight ? "end" : "start") + '" font-size="11" fill="var(--accent)">ridge ' + ridge.toFixed(2) + "</text>";
    // Roof labels sit above their lines; the slope label is rotated to follow the slope.
    const sx = lx(0.07), sy = ly(Math.max(roof(0.07), y0));
    const ang = Math.atan2(ly(roof(Math.min(ra, 0.2))) - ly(roof(0.07)), lx(Math.min(ra, 0.2)) - lx(0.07)) * 180 / Math.PI;
    if (ridge > 0.12) s += '<text x="' + sx + '" y="' + (sy - 7) + '" transform="rotate(' + ang.toFixed(1) + " " + sx + " " + sy + ')" font-size="11" fill="var(--accent)">bandwidth roof</text>';
    s += '<text x="' + (W - R - 4) + '" y="' + (ly(P) - 8) + '" text-anchor="end" font-size="11" fill="var(--accent)">compute roof</text>';
    points.forEach((p) => {
      const y = p.measured ? p.y : roof(p.ai);
      const c = p.measured ? "var(--copper)" : "var(--blue)";
      // Labels go below the dot, so they never sit on the roof line itself.
      s += '<circle cx="' + lx(p.ai) + '" cy="' + ly(y) + '" r="5.5" fill="' + c + '"/>';
      s += '<text x="' + (lx(p.ai) + 2) + '" y="' + (ly(y) + 19) + '" text-anchor="middle" font-size="11" fill="' + c + '">' + p.name + "</text>";
    });
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.innerHTML = s;
    out.innerHTML = points.map((p) => {
      const r = roof(p.ai);
      const bound = p.ai < ridge ? "bandwidth-bound" : "compute-bound";
      return p.name + ": " + p.ai.toFixed(2) + " FLOP/byte → roof " + r.toFixed(1) + " GFLOP/s, " + bound + (p.measured ? ". Actually reached 0.82, under the roof" : "");
    }).join("<br>");
  }
  bw.addEventListener("input", render);
  peak.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- roofline: which limit wins

function initRoofPick() {
  const root = document.querySelector("[data-roofpick]");
  if (!root) return;
  // Made-up roofs, the same as the worked example on the page.
  const B = 30, P = 28;
  const slider = root.querySelector("[data-ai]"), label = root.querySelector("[data-ai-label]");
  const bars = root.querySelector("[data-pickbars]"), note = root.querySelector("[data-picknote]");
  const presets = root.querySelector("[data-ai-presets]");
  presets.style.cssText = "display:flex;flex-wrap:wrap;gap:.35rem";
  for (const [name, ai] of [["Phase 0 sum", 0.25], ["f32 GEMV", 0.5], ["Q4_0 GEMV", 2 / (18 / 32)], ["prefill, 32 tokens", 16]]) {
    const b = el("button", { type: "button", class: "btn ghost", text: name });
    b.addEventListener("click", () => { slider.value = Math.log2(ai); render(); });
    presets.append(b);
  }
  const fmt = (v) => (v < 10 ? v.toFixed(v < 1 ? 2 : 1) : v.toFixed(0));
  function render() {
    const ai = Math.pow(2, Number(slider.value));
    label.textContent = fmt(ai) + " FLOP/byte";
    const feed = ai * B, top = Math.max(feed, P);
    const memBound = feed < P;
    const width = (v) => (100 * v / top).toFixed(1) + "%";
    bars.innerHTML = "";
    bars.append(el("div", { class: "bar", style: memBound ? "" : "opacity:.4" }, [el("span", { text: "memory can feed" }), el("i", { style: "--w:" + width(feed) }), el("b", { text: fmt(feed) + " GFLOP/s" })]));
    bars.append(el("div", { class: "bar alt", style: memBound ? "opacity:.4" : "" }, [el("span", { text: "core can do" }), el("i", { style: "--w:" + width(P) }), el("b", { text: P + " GFLOP/s" })]));
    note.textContent = memBound
      ? fmt(ai) + " × " + B + " = " + fmt(feed) + " GFLOP/s is the lower limit. Memory-bound: the core is idle " + Math.round(100 * (1 - feed / P)) + "% of the time, waiting for bytes. A faster core would not help; fewer bytes would."
      : fmt(ai) + " × " + B + " = " + fmt(feed) + " GFLOP/s is more than the core can use, so its " + P + " GFLOP/s is the limit. Compute-bound: the bytes wait for the arithmetic.";
  }
  slider.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- matrix × vector

function initMatvec() {
  const root = document.querySelector("[data-matvec]");
  if (!root) return;
  const R = 4, C = 8;
  const W = [[2, -1, 0, 3, 1, -2, 1, 0], [1, 1, -1, 0, 2, 1, -3, 1], [0, 2, 1, -1, -1, 0, 2, 3], [-2, 0, 1, 1, 0, 3, -1, 2]];
  const X = [[1, 0, 2, -1, 1, 1, 0, 2], [0, 1, 1, 2, -1, 0, 1, 1], [2, -1, 0, 1, 1, 2, -1, 0], [1, 1, 1, 0, 0, -1, 2, 1]];
  const wBox = root.querySelector("[data-w]"), xBox = root.querySelector("[data-x]"), yBox = root.querySelector("[data-y]");
  const stats = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const stepBtn = root.querySelector("[data-step]"), playBtn = root.querySelector("[data-play]"), resetBtn = root.querySelector("[data-reset]");
  let row = 0, timer = 0;
  const tokens = () => (root.querySelector("[data-mode]:checked").value === "prefill" ? 4 : 1);
  function draw() {
    const k = tokens();
    wBox.innerHTML = ""; xBox.innerHTML = ""; yBox.innerHTML = "";
    wBox.style.gridTemplateColumns = xBox.style.gridTemplateColumns = "repeat(" + C + ", 1fr)";
    yBox.style.gridTemplateColumns = "repeat(" + k + ", 1fr)";
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      wBox.append(el("span", { class: r < row - 1 ? "read" : r === row - 1 ? "now" : "", text: String(W[r][c]) }));
    }
    for (let t = 0; t < k; t++) for (let c = 0; c < C; c++) xBox.append(el("span", { class: "x" + t, text: String(X[t][c]) }));
    for (let r = 0; r < R; r++) for (let t = 0; t < k; t++) {
      const v = W[r].reduce((a, w, c) => a + w * X[t][c], 0);
      yBox.append(el("span", { class: (r < row ? "done x" + t : "") + (r === row - 1 ? " now" : ""), text: r < row ? String(v) : "·" }));
    }
    const read = row * C, flops = 2 * read * k, bytes = 4 * read;
    stats.innerHTML = "";
    for (const [v, l] of [[read + " / " + R * C, "weights read"], [bytes + " B", "bytes moved (f32)"], [flops, "FLOPs done"], [k, "uses of each weight"], [(2 * k / 4).toFixed(1), "FLOP per byte"]]) {
      stats.append(el("div", { class: "stat" }, [el("b", { text: String(v) }), el("span", { text: l })]));
    }
    if (row === 0) note.textContent = k === 1 ? "Decode: one token's vector. Press Step to compute one output number per row." : "Prefill: four tokens' vectors at once. Each weight, once loaded, is used four times.";
    else if (row < R) note.textContent = "Row " + (row - 1) + ": each of its " + C + " weights is multiplied by " + (k === 1 ? "one input number" : "four input numbers, one per token") + " and summed. " + (k === 1 ? "Then the weight is never needed again for this token." : "Same bytes, four times the work.");
    else note.textContent = k === 1 ? "Done. Every weight was read exactly once and used exactly once. For the next token, all of them must be read again. That is decode." : "Done. Same 128 bytes read, four times the FLOPs. That reuse is why prefill can be compute-bound.";
    stepBtn.disabled = row >= R;
  }
  function stop() { clearInterval(timer); timer = 0; playBtn.textContent = "Play"; }
  stepBtn.addEventListener("click", () => { if (row < R) { row++; draw(); } });
  playBtn.addEventListener("click", () => {
    if (timer) return stop();
    if (row >= R) row = 0;
    playBtn.textContent = "Pause";
    timer = setInterval(() => { row++; draw(); if (row >= R) stop(); }, 900);
    row++; draw();
  });
  resetBtn.addEventListener("click", () => { stop(); row = 0; draw(); });
  root.addEventListener("change", (e) => { if (e.target.matches("[data-mode]")) { stop(); row = 0; draw(); } });
  draw();
}

// ---------------------------------------------------------------- generation loop

function initGenLoop() {
  const root = document.querySelector("[data-genloop]");
  if (!root) return;
  const prompt = ["Once", " upon", " a", " time"];
  const script = [
    [[" there", .62], [" lived", .21], [",", .09]],
    [[" was", .71], [" lived", .18], [" were", .05]],
    [[" a", .80], [" an", .08], [" one", .05]],
    [[" little", .44], [" big", .19], [" small", .12]],
    [[" cat", .23], [" girl", .21], [" dog", .17]],
    [[" named", .52], [" who", .30], [".", .08]],
    [[" Tom", .31], [" Lily", .12], [" Max", .10]],
    [[".", .66], [",", .22], [" who", .06]],
    [[" He", .48], [" Tom", .37], [" One", .07]],
    [[" loved", .41], [" liked", .33], [" was", .11]],
    [[" to", .58], [" his", .14], [" the", .11]],
    [[" play", .62], [" chase", .15], [" nap", .09]]
  ];
  const chips = root.querySelector("[data-chips]"), cands = root.querySelector("[data-cands]");
  const kv = root.querySelector("[data-kv]"), stats = root.querySelector("[data-stats]"), note = root.querySelector("[data-note]");
  const go = root.querySelector("[data-go]"), reset = root.querySelector("[data-reset]"), model = root.querySelector("[data-model]");
  let step = 0;
  function draw() {
    const size = Number(model.value);
    chips.innerHTML = "";
    prompt.forEach((t) => chips.append(el("span", { class: "chip p", text: t.trim() || "␣" })));
    for (let i = 0; i < step; i++) chips.append(el("span", { class: "chip g" + (i === step - 1 ? " new" : ""), text: script[i][0][0].trim() || "␣" }));
    cands.innerHTML = "";
    if (step > 0) {
      for (const [t, p] of script[step - 1]) {
        cands.append(el("div", { class: "cand" + (t === script[step - 1][0][0] ? " pick" : "") }, [
          el("span", { text: "“" + t + "”" }), el("i", { style: "width:" + (p * 100).toFixed(0) + "%" }), el("b", { text: (p * 100).toFixed(0) + "%" })
        ]));
      }
    }
    kv.innerHTML = "";
    const remembered = step === 0 ? 0 : prompt.length + step - 1;
    for (let i = 0; i < remembered; i++) kv.append(el("span"));
    kv.setAttribute("aria-label", remembered + " tokens in the KV cache");
    const passes = step;
    stats.innerHTML = "";
    for (const [v, l] of [[passes, "forward passes"], [step === 0 ? 0 : 1, "of them prefill"], [Math.max(0, step - 1), "of them decode"], [fmtBytes(passes * size), "weights read in total"], [(size / 30e9 * 1000).toFixed(1) + " ms", "floor per token at 30 GB/s (assumed)"]]) {
      stats.append(el("div", { class: "stat" }, [el("b", { text: String(v) }), el("span", { text: l })]));
    }
    if (step === 0) { note.textContent = "Nothing has run yet. The prompt is four tokens."; go.textContent = "Run prefill"; }
    else if (step === 1) { note.textContent = "Prefill: all four prompt tokens went through the model together, reading every weight once. Out came scores for every token in the vocabulary. The top one was picked (greedy)."; go.textContent = "Generate next token"; }
    else if (step < script.length) { note.textContent = "Decode: one new token in, one full read of the weights, one token out. The KV cache remembers earlier tokens, so they are not reprocessed, but the weights are still read again."; }
    else { note.textContent = "End of the demo. " + (step - 1) + " decode steps meant " + (step - 1) + " more full reads of the weights. That repetition is the whole reason memory bandwidth sets decode speed."; }
    go.disabled = step >= script.length;
  }
  go.addEventListener("click", () => { if (step < script.length) { step++; draw(); } });
  reset.addEventListener("click", () => { step = 0; draw(); });
  model.addEventListener("change", draw);
  draw();
}

// ---------------------------------------------------------------- roadmap

function initRoadmap() {
  const root = document.querySelector("[data-roadmap]");
  if (!root) return;
  const btns = [...root.querySelectorAll("[data-phase-btn]")];
  const boxes = [...root.querySelectorAll("[data-phases]")];
  const caption = root.querySelector("[data-caption]");
  function select(n) {
    btns.forEach((b) => b.classList.toggle("on", b.dataset.phaseBtn === n));
    boxes.forEach((b) => b.classList.toggle("lit", b.dataset.phases.split(" ").includes(n)));
    const b = btns.find((x) => x.dataset.phaseBtn === n);
    caption.textContent = b ? b.dataset.says : "";
  }
  btns.forEach((b) => b.addEventListener("click", () => select(b.dataset.phaseBtn)));
  select(root.dataset.roadmap || "0");
}

// ---------------------------------------------------------------- alignment: split loads

function initAlignSplit() {
  const root = document.querySelector("[data-split]");
  if (!root) return;
  const slider = root.querySelector("[data-offset]");
  const label = root.querySelector("[data-offset-label]");
  const svg = root.querySelector("svg");
  const out = root.querySelector("[data-out]");
  function render() {
    const off = Number(slider.value);
    label.textContent = off + " bytes";
    const px = 3.3, L = 12, W = L * 2 + 192 * px;
    let s = "";
    for (let line = 0; line < 3; line++) {
      s += '<rect x="' + (L + line * 64 * px) + '" y="20" width="' + (64 * px) + '" height="34" fill="' + (line % 2 ? "var(--bg-2)" : "var(--surface)") + '" stroke="var(--muted)"/>';
      s += '<text x="' + (L + line * 64 * px + 32 * px) + '" y="14" text-anchor="middle" class="muted" font-size="11">cache line ' + line + " (64 B)</text>";
    }
    let splits = 0;
    for (let i = 0; i < 8; i++) {
      const start = off + i * 16, end = start + 16;
      const split = Math.floor(start / 64) !== Math.floor((end - 1) / 64);
      if (split) splits++;
      const c = split ? "var(--copper)" : "var(--accent)";
      s += '<rect x="' + (L + start * px + 1) + '" y="26" width="' + (16 * px - 2) + '" height="22" rx="3" fill="' + (split ? "var(--copper-soft)" : "var(--accent-soft)") + '" stroke="' + c + '"/>';
      s += '<text x="' + (L + (start + 8) * px) + '" y="41" text-anchor="middle" font-size="10" fill="' + c + '">' + i + "</text>";
    }
    s += '<text x="' + L + '" y="74" class="muted" font-size="11">each box is one 16-byte NEON load of four f32s; copper ones straddle two cache lines</text>';
    svg.setAttribute("viewBox", "0 0 " + W + " 84");
    svg.innerHTML = s;
    out.textContent = splits === 0
      ? "Start offset " + off + ": no load crosses a line. Every load touches exactly one cache line."
      : "Start offset " + off + ": " + splits + " of 8 loads touch two cache lines. Each of those does extra work, and on a big stream that adds up.";
  }
  slider.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- alignment: is this address aligned?

function initAlignCheck() {
  const root = document.querySelector("[data-aligncheck]");
  if (!root) return;
  const input = root.querySelector("[data-addr]"), chips = root.querySelector("[data-chips]"), note = root.querySelector("[data-alignnote]");
  const why = { 4: "one f32", 16: "one NEON load", 64: "one cache line" };
  function render() {
    const a = Math.max(0, Math.floor(Number(input.value) || 0));
    chips.innerHTML = "";
    for (const n of [4, 16, 64]) {
      const r = a % n;
      chips.append(el("div", { class: "stat" }, [
        el("b", { text: (r === 0 ? "✓ " : "✗ ") + n + "-aligned", style: "color:" + (r === 0 ? "var(--accent-ink)" : "var(--copper)") }),
        el("span", { text: a + " % " + n + " = " + r + " · " + why[n] })
      ]));
    }
    const lineStart = a - (a % 64);
    if (a % 64 === 0) note.textContent = a + " is a cache-line edge. An f32 array starting here never splits a 16-byte load.";
    else if (a % 16 === 0) note.textContent = "Not a line edge (the line starts at " + lineStart + "), but 16-byte loads from here still never cross a line edge.";
    else if (a % 4 === 0) note.textContent = "A legal f32 address, but " + (a % 16) + " bytes past a 16-byte boundary. One NEON load in four will split across two lines.";
    else note.textContent = "Not even a multiple of 4. An f32 array would never start here.";
  }
  input.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- measuring time: clock cost

function initClockCost() {
  const root = document.querySelector("[data-clock]");
  if (!root) return;
  const slider = root.querySelector("[data-work]");
  const out = root.querySelector("[data-out]");
  const bars = root.querySelector("[data-bars]");
  const clock = 45;
  function render() {
    const ns = Math.round(Math.pow(10, Number(slider.value)));
    const label = ns >= 1e6 ? (ns / 1e6).toFixed(1) + " ms" : ns >= 1e3 ? (ns / 1e3).toFixed(1) + " µs" : ns + " ns";
    const err = 100 * clock / ns;
    bars.innerHTML = "";
    const total = ns + clock;
    bars.append(el("div", { class: "split" }, [
      el("i", { class: "work", style: "width:" + (100 * ns / total).toFixed(2) + "%" }),
      el("i", { class: "clk", style: "width:" + (100 * clock / total).toFixed(2) + "%" })
    ]));
    const verdict = err > 10 ? "Mostly clock. Time a batch of calls and divide instead." : err > 1 ? "Noticeable error. Batch it if the difference matters." : "Fine. The clock is lost in the noise.";
    out.innerHTML = "work " + label + " + one clock read ≈ 45 ns<br>clock is " + (err >= 1 ? err.toFixed(0) : err.toFixed(2)) + "% of the work. " + verdict;
  }
  slider.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- memory: latency ladder

function initLatency() {
  const root = document.querySelector("[data-latency]");
  if (!root) return;
  const levels = [["register", 0.3], ["L1 cache", 1], ["L2 cache", 5], ["SLC", 20], ["DRAM", 120]];
  const bars = root.querySelector("[data-bars]");
  function render() {
    const human = root.querySelector("[data-scale]:checked").value === "human";
    bars.innerHTML = "";
    for (const [name, ns] of levels) {
      const w = 6 + 94 * (Math.log10(ns / 0.3) / Math.log10(120 / 0.3));
      let txt;
      if (human) { const s = ns; txt = s < 60 ? s.toFixed(s < 1 ? 1 : 0) + " seconds" : (s / 60).toFixed(0) + " minutes"; }
      else txt = "~" + ns + " ns";
      bars.append(el("div", { class: "bar" + (name === "DRAM" ? " alt" : "") }, [el("span", { text: name }), el("i", { style: "--w:" + w.toFixed(1) + "%" }), el("b", { text: txt })]));
    }
  }
  root.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- memory: will it fit?

function initFit() {
  const root = document.querySelector("[data-fit]");
  if (!root) return;
  const KB = 1024, MB = KB * KB, GB = MB * KB;
  // Cache sizes are orders of magnitude and launch claims, not measurements.
  const caches = [["L1 (~100 KB, typical)", 100 * KB], ["L2 (12 MB, claimed)", 12 * MB]];
  const presets = [["a 4 KB row", 4 * KB], ["stories15M f32", 60 * MB], ["0.5B Q4_0", 281 * MB], ["0.5B f32", 2 * GB]];
  const slider = root.querySelector("[data-size]"), bars = root.querySelector("[data-fitbars]"), note = root.querySelector("[data-fitnote]");
  const lo = Math.log2(KB), hi = Math.log2(4 * GB);
  const width = b => (4 + 96 * (Math.log2(b) - lo) / (hi - lo)).toFixed(1) + "%";
  const fmt = b => b >= GB ? (b / GB).toFixed(1) + " GB" : b >= MB ? (b / MB).toFixed(b < 10 * MB ? 1 : 0) + " MB" : (b / KB).toFixed(0) + " KB";
  const seg = root.querySelector("[data-presets]");
  for (const [name, bytes] of presets) {
    const b = el("button", { type: "button", class: "btn ghost", text: name });
    b.addEventListener("click", () => { slider.value = Math.log2(bytes); render(); });
    seg.append(b);
  }
  seg.className = "";
  seg.style.cssText = "display:flex;flex-wrap:wrap;gap:.35rem";
  function render() {
    const bytes = Math.pow(2, Number(slider.value));
    bars.innerHTML = "";
    for (const [name, size] of caches) {
      bars.append(el("div", { class: "bar" }, [el("span", { text: name }), el("i", { style: "--w:" + width(size) }), el("b", { text: fmt(size) })]));
    }
    bars.append(el("div", { class: "bar alt" }, [el("span", { text: "your working set" }), el("i", { style: "--w:" + width(bytes) }), el("b", { text: fmt(bytes) })]));
    const home = caches.find(([, size]) => bytes <= size);
    if (home) note.textContent = "Fits in " + home[0].split(" ")[0] + ". Read it again and again, and every pass after the first is served from that cache.";
    else note.textContent = "Bigger than every cache. Repeated reads keep pushing out the start before you get back to it, so each pass comes from DRAM. Only about " + Math.max(0.1, 100 * 12 * MB / bytes).toFixed(1) + "% could stay in L2.";
  }
  slider.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- rust memory: Layout builder

function initLayout() {
  const root = document.querySelector("[data-layout]");
  if (!root) return;
  const nIn = root.querySelector("[data-n]"), alignIn = root.querySelector("[data-align]");
  const out = root.querySelector("[data-out]"), note = root.querySelector("[data-note]");
  // Start from the defaults in the markup, not from values a browser restored into the fields.
  nIn.value = "1000";
  alignIn.value = "64";
  function render() {
    const n = Math.max(0, Math.floor(Number(nIn.value) || 0));
    const want = Number(alignIn.value);
    const size = 4 * n;
    const row = (left, right) => left.padEnd(27) + "→ " + right;
    const lines = [row("Layout::array::<f32>(" + n + ")", "size " + size + " B, align 4")];
    if (want & (want - 1)) {
      lines.push(row("    .align_to(" + want + ")", "Err(LayoutError)"));
      note.textContent = "48 is not a power of two. Alignments are always 1, 2, 4, 8, 16, 32, 64 …, because an address is aligned when its low bits are zero.";
      out.textContent = lines.join("\n");
      return;
    }
    const align = Math.max(4, want);
    lines.push(row("    .align_to(" + want + ")", "size " + size + " B, align " + align));
    if (n === 0) {
      lines.push(row("alloc_zeroed(layout)", "not allowed: size is 0"));
      note.textContent = "Calling the allocator with size 0 is undefined behaviour. An empty buffer uses a dangling pointer and length 0, and skips dealloc.";
    } else {
      lines.push(row("alloc_zeroed(layout)", "address % " + align + " == 0"));
      const aligned = Math.floor((size - 1) / 64) + 1;
      const worst = Math.floor((64 - align + size - 1) / 64) + 1;
      if (align >= 64) note.textContent = "Starts on a cache-line boundary, so the buffer covers " + aligned.toLocaleString() + " cache lines, the minimum.";
      else note.textContent = "Only " + align + "-byte aligned, so it may start mid-line: " + aligned.toLocaleString() + " cache lines at best, " + worst.toLocaleString() + " if it starts " + align + " bytes before a boundary.";
    }
    out.textContent = lines.join("\n");
  }
  nIn.addEventListener("input", render);
  alignIn.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- rust pinning: CPU set bitmask

function initCpuMask() {
  const root = document.querySelector("[data-cpumask]");
  if (!root) return;
  const box = root.querySelector("[data-cores]"), out = root.querySelector("[data-out]"), note = root.querySelector("[data-note]");
  const on = new Set([6]);
  const btns = [];
  for (let c = 0; c < 8; c++) {
    const prime = c >= 6;
    const b = el("button", { type: "button", class: "core" + (prime ? " prime" : ""), "aria-pressed": "false" },
      [el("b", { text: "cpu" + c }), el("small", { text: prime ? "prime" : "perf" })]);
    b.addEventListener("click", () => { if (on.has(c)) on.delete(c); else on.add(c); render(); });
    box.append(b);
    btns.push(b);
  }
  function render() {
    let mask = 0;
    btns.forEach((b, c) => { b.setAttribute("aria-pressed", String(on.has(c))); if (on.has(c)) mask |= 1 << c; });
    const list = [...on].sort((a, b) => a - b);
    const calls = list.length ? list.map((c) => "CPU_SET(" + c + ", &mut set);").join("\n") : "// no CPU_SET calls";
    out.textContent = "cpu7 … cpu0:  " + mask.toString(2).padStart(8, "0") + "\nas hex:       0x" + mask.toString(16).padStart(2, "0") + "\n\nCPU_ZERO(&mut set);\n" + calls;
    const primes = list.filter((c) => c >= 6).length, perfs = list.length - primes;
    if (!list.length) note.textContent = "Empty set: sched_setaffinity returns −1 with EINVAL. There is nowhere the thread may run.";
    else if (list.length === 1) note.textContent = "A set of one is a pin. The thread stays on cpu" + list[0] + ".";
    else if (primes && perfs) note.textContent = "Mixes prime and performance cores. A number from this one thread describes neither class; the scheduler picks the mix.";
    else note.textContent = "One thread with this set can still move between these " + list.length + " cores. For the parallel sweep, give each worker a set of one.";
  }
  render();
}

// ---------------------------------------------------------------- rust threads: sample count

function initSampler() {
  const root = document.querySelector("[data-sampler]");
  if (!root) return;
  const len = root.querySelector("[data-len]"), every = root.querySelector("[data-every]");
  const lenLabel = root.querySelector("[data-len-label]"), svg = root.querySelector("[data-svg]"), out = root.querySelector("[data-out]");
  function fmt(ms) { return ms >= 60000 ? (ms / 60000).toFixed(1) + " min" : ms >= 1000 ? (ms / 1000).toFixed(ms >= 10000 ? 0 : 1) + " s" : Math.round(ms) + " ms"; }
  function render() {
    const run = Math.pow(10, Number(len.value)), dt = Number(every.value);
    lenLabel.textContent = "(" + fmt(run) + ")";
    // Draw a window 1.25× the run so ticks outside it are visible too.
    const span = run * 1.25, L = 12, W = 616, x = (t) => L + W * t / span;
    const offset = dt * 0.37;
    let s = '<rect x="' + x(span * 0.1) + '" y="18" width="' + (W * run / span) + '" height="22" rx="3" fill="var(--accent-soft)" stroke="var(--accent)"/>';
    const total = Math.floor((span - offset) / dt) + 1;
    let inside = 0;
    if (total <= 160) {
      for (let k = 0; k < total; k++) {
        const t = offset + k * dt, inRun = t >= span * 0.1 && t <= span * 0.1 + run;
        if (inRun) inside++;
        s += '<line x1="' + x(t) + '" x2="' + x(t) + '" y1="12" y2="46" stroke="' + (inRun ? "var(--copper)" : "var(--faint)") + '" stroke-width="1.5"/>';
      }
    } else {
      inside = Math.floor(run / dt);
      s += '<rect x="' + x(span * 0.1) + '" y="12" width="' + (W * run / span) + '" height="34" fill="var(--copper)" opacity=".35"/>';
    }
    s += '<text class="muted" x="' + L + '" y="64" font-size="11">copper ticks fall inside the timed run; grey ones are before or after it</text>';
    svg.innerHTML = s;
    const verdict = inside < 3 ? "Too few to average. Lengthen the run." : inside < 20 ? "Usable, but one odd reading still moves the average." : "Plenty, as long as the sensor itself updates this often.";
    out.textContent = fmt(run) + " run, one sample every " + fmt(dt) + " → " + inside.toLocaleString() + " sample" + (inside === 1 ? "" : "s") + " inside. " + verdict;
  }
  len.addEventListener("input", render);
  every.addEventListener("change", render);
  render();
}

// ---------------------------------------------------------------- rust sysfs: unit converter

function initUnits() {
  const root = document.querySelector("[data-units]");
  if (!root) return;
  const v = (k) => Number(root.querySelector('[data-u="' + k + '"]').value) || 0;
  const out = root.querySelector("[data-out]"), note = root.querySelector("[data-note]");
  function render() {
    const ghz = v("freq") / 1e6, c = v("temp") / 1000, amps = v("cur") / 1e6, volts = v("volt") / 1e6;
    const watts = Math.abs(amps) * volts;
    out.textContent =
      "clock    " + v("freq") + " kHz  ÷ 1e6  = " + ghz.toFixed(3) + " GHz\n" +
      "temp     " + v("temp") + " m°C  ÷ 1000 = " + c.toFixed(1) + " °C\n" +
      "current  " + v("cur") + " µA   ÷ 1e6  = " + amps.toFixed(3) + " A\n" +
      "voltage  " + v("volt") + " µV   ÷ 1e6  = " + volts.toFixed(3) + " V\n" +
      "power    |I| × V          = " + watts.toFixed(2) + " W";
    note.textContent = amps < 0 ? "Current is negative. On many phones that means discharging, on some charging. An idle, unplugged reading settles which." :
      amps > 0 ? "Current is positive. Check with an idle, unplugged reading which sign means discharge on this phone." : "Zero current: the gauge may be idle, or the phone is on the charger.";
  }
  root.addEventListener("input", render);
  render();
}

// ---------------------------------------------------------------- plotting: linear vs log size axis

function initAxis() {
  const root = document.querySelector("[data-axis]");
  if (!root) return;
  const svg = root.querySelector("[data-svg]"), out = root.querySelector("[data-out]");
  // Illustrative shape only: L1, L2, SLC, DRAM plateaus.
  const pts = [];
  for (let p = 12; p <= 28; p++) {
    const g = p <= 16 ? 90 : p <= 23 ? 45 : p === 24 ? 26 : 12;
    pts.push([Math.pow(2, p), g]);
  }
  const L = 46, R = 596, T = 26, B = 186, ymax = 100;
  const y = (g) => B - (B - T) * g / ymax;
  function render() {
    const log = root.querySelector("[data-axisscale]:checked").value === "log";
    const x = (b) => log ? L + (R - L) * (Math.log2(b) - 12) / 16 : L + (R - L) * b / Math.pow(2, 28);
    let s = '<line x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + B + '" stroke="var(--line)"/>' +
      '<line x1="' + L + '" y1="' + B + '" x2="' + R + '" y2="' + B + '" stroke="var(--line)"/>';
    for (const g of [0, 50, 100]) s += '<text class="muted" x="' + (L - 6) + '" y="' + (y(g) + 4) + '" text-anchor="end" font-size="11">' + g + "</text>";
    s += '<text class="muted" x="' + (L - 6) + '" y="12" text-anchor="end" font-size="10">GB/s</text>';
    const ticks = log ? [[2 ** 12, "4 KiB"], [2 ** 16, "64 KiB"], [2 ** 20, "1 MiB"], [2 ** 24, "16 MiB"], [2 ** 28, "256 MiB"]]
      : [[0, "0"], [2 ** 26, "64 MiB"], [2 ** 27, "128 MiB"], [3 * 2 ** 26, "192 MiB"], [2 ** 28, "256 MiB"]];
    for (const [b, t] of ticks) {
      s += '<line x1="' + x(b) + '" x2="' + x(b) + '" y1="' + B + '" y2="' + (B + 4) + '" stroke="var(--muted)"/>';
      s += '<text class="muted" x="' + x(b) + '" y="' + (B + 18) + '" text-anchor="middle" font-size="11">' + t + "</text>";
    }
    s += '<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="' + pts.map(([b, g]) => x(b).toFixed(1) + "," + y(g)).join(" ") + '"/>';
    for (const [b, g] of pts) s += '<circle cx="' + x(b).toFixed(1) + '" cy="' + y(g) + '" r="3.5" fill="var(--accent)"/>';
    s += '<text class="muted" x="' + ((L + R) / 2) + '" y="' + (B + 40) + '" text-anchor="middle" font-size="11">working set</text>';
    svg.innerHTML = s;
    const crowded = pts.filter(([b]) => b <= 8 * 2 ** 20).length;
    out.textContent = log
      ? "Each doubling gets the same width. The L1 plateau, the L2 plateau and both cliffs are all readable."
      : crowded + " of the " + pts.length + " points (4 KiB to 8 MiB) sit in the leftmost 3% of the axis. The L1 and L2 cliffs are invisible.";
  }
  root.addEventListener("change", render);
  render();
}
