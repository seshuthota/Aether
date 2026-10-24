// Learning components, loaded after site.js (uses el(), pageKey() and store).
//
//   [data-quiz="predict"]  a guess before a section: open, no confidence asked
//   [data-quiz="check"]    a check after a section: folded until opened
//   [data-quiz="probe"]    a question in the folded "Already know some of
//                          this?" skim test inside the how-to box
//   [data-probe]           the skim test; its result marks sections to read or skim
//   .toc[data-stepmap]     the page's steps: a folded list in the page, and the
//                          "On this page" sidebar from 1400px
//   [data-record]          this page's answers, copyable as Markdown
//
// A quiz is plain markup. Without JavaScript (no "js" class on <html>, which
// site.js adds) the CSS shows each question with its right answer and note:
//   <div class="quiz" data-quiz="check" data-for="layout" id="q-layout">
//     <p>Question?</p>
//     <ul>
//       <li data-right>Answer <div class="why">Why it is right.</div></li>
//       <li>Wrong answer <div class="why">The idea behind picking it, and what is true.</div></li>
//     </ul>
//   </div>
// Answers are saved in this browser only, one record per page.

const CONFIDENCE = [["guess", "Guessing"], ["fair", "Fairly sure"], ["sure", "Certain"]];

(function () {
  // The <head> script drops "js" if this file is slow to run; it is running now.
  document.documentElement.classList.add("js");
  const key = "aether-answers:" + pageKey(location.href);
  let saved = {};
  try { saved = JSON.parse(store.get(key) || "{}") || {}; } catch (e) { saved = {}; }
  const listeners = [];
  const changed = () => { store.set(key, JSON.stringify(saved)); listeners.forEach(f => f()); };
  const ctx = { saved, changed, onChange: f => listeners.push(f) };

  // Each quiz is built on its own: one with bad markup is restored to its
  // original HTML and shown the scripts-off way (question, answer, note),
  // and the rest of the page still works.
  const quizzes = [];
  for (const root of document.querySelectorAll("[data-quiz]")) {
    const original = root.innerHTML;
    try {
      quizzes.push(initQuiz(root, ctx));
      root.classList.add("is-ready");
    } catch (e) {
      console.error("learn.js: quiz #" + (root.id || "(no id)") + " could not be built:", e);
      root.innerHTML = original;
      root.classList.add("is-fallback");
    }
  }
  for (const [name, init] of [["skim test", initProbe], ["step map", initStepMap], ["answer record", initRecord]]) {
    try { init(quizzes, ctx); } catch (e) { console.error("learn.js: " + name + " could not be built:", e); }
  }
})();

function initQuiz(root, ctx) {
  const kind = root.dataset.quiz, id = root.id;
  // Check the markup before changing anything, so a failure leaves it intact.
  const qp = root.querySelector(":scope > p"), ul = root.querySelector(":scope > ul");
  if (!id) throw new Error("quiz has no id");
  if (!["predict", "check", "probe"].includes(kind)) throw new Error("unknown data-quiz " + JSON.stringify(kind));
  if (!qp || !ul) throw new Error("quiz needs a <p> question and a <ul> of options");
  if (ul.querySelectorAll(":scope > li[data-right]").length !== 1) throw new Error("quiz needs exactly one data-right option");
  const items = [...root.querySelectorAll(":scope > ul > li")].map(li => {
    const why = li.querySelector(".why");
    const label = li.cloneNode(true);
    const w = label.querySelector(".why");
    if (w) w.remove();
    return { label: label.innerHTML.trim(), why: why ? why.innerHTML : "", right: li.hasAttribute("data-right") };
  });
  const question = qp.innerHTML;
  const text = qp.textContent.trim();
  ul.remove();
  qp.remove();

  const body = el("div", { class: "quiz-body" });
  const tag = { predict: "Predict first", check: "Check your understanding", probe: "" }[kind];
  if (kind === "check") {
    const det = el("details", { class: "quiz-fold" }, [el("summary", {}, [el("span", { text: tag }), el("span", { class: "quiz-state" })]), body]);
    root.append(det);
  } else {
    if (tag) root.append(el("span", { class: "lbl", text: tag }));
    root.append(body);
  }
  const state = root.querySelector(".quiz-state");

  const q = { root, id, kind, text, items, for: root.dataset.for || "", result: () => ctx.saved[id] };

  function render() {
    body.innerHTML = "";
    body.append(el("p", { class: "quiz-q", html: question }));
    const rec = ctx.saved[id];
    let pick = null, conf = null;
    const opts = el("div", { class: "quiz-opts", role: "radiogroup" });
    items.forEach((it, i) => {
      const input = el("input", { type: "radio", name: id });
      const row = el("label", { class: "opt" }, [input, el("span", { html: it.label })]);
      if (rec) {
        input.disabled = true;
        input.checked = rec.pick === i;
        if (it.right) row.classList.add("right");
        if (rec.pick === i && !it.right) row.classList.add("wrong");
        if (it.why && (rec.pick === i || it.right)) row.append(el("div", { class: "why", html: it.why }));
      }
      input.addEventListener("change", () => { pick = i; ready(); });
      opts.append(row);
    });
    body.append(opts);
    if (rec) { body.append(feedback(rec)); paintState(); return; }

    const confRow = el("div", { class: "seg conf" });
    if (kind !== "predict") {
      for (const [v, l] of CONFIDENCE) {
        const input = el("input", { type: "radio", name: id + "-conf", value: v });
        input.addEventListener("change", () => { conf = v; ready(); });
        confRow.append(el("label", {}, [input, document.createTextNode(l)]));
      }
      body.append(el("div", { class: "quiz-row" }, [el("span", { class: "quiz-hint", text: "How sure are you?" }), confRow]));
    }
    const think = el("textarea", { rows: "2", placeholder: kind === "predict" ? "Why do you think so?" : "Your reasoning, in your own words" });
    think.hidden = true;
    const reason = el("button", { type: "button", class: "linkbtn", text: "+ add a reason" });
    reason.addEventListener("click", () => { think.hidden = false; reason.hidden = true; think.focus(); });
    const submit = el("button", { type: "button", class: "btn", text: kind === "predict" ? "Lock in my guess" : "Answer" });
    submit.disabled = true;
    const dunno = el("button", { type: "button", class: "btn ghost", text: "I don’t know" });
    body.append(el("div", { class: "quiz-row" }, [submit, dunno, reason]), think);
    function ready() { submit.disabled = pick === null || (kind !== "predict" && conf === null); }
    function done(r) {
      ctx.saved[id] = Object.assign(r, { thinking: think.value.trim(), at: new Date().toISOString() });
      ctx.changed();
      render();
    }
    submit.addEventListener("click", () => done({ pick, conf, right: items[pick].right }));
    dunno.addEventListener("click", () => done({ pick: null, conf: null, right: false, dontKnow: true }));
  }

  function feedback(rec) {
    const box = el("div", { class: "quiz-fb" });
    const right = items.find(it => it.right);
    let verdict, cls;
    if (kind === "predict") {
      cls = "ok";
      verdict = rec.dontKnow ? "No guess is fine. The section below answers it."
        : rec.right ? "Your guess is right. The section below shows why it has to be."
        : "Not this time. Keep your guess in mind; the section below shows where it goes wrong.";
    } else if (rec.dontKnow) {
      cls = "gap"; verdict = "A gap, not a wrong idea. " + (right && right.why ? "The answer is marked above." : "");
    } else if (rec.right) {
      cls = "ok";
      verdict = rec.conf === "guess" ? "Right, but you were guessing. Read the explanation under the answer once more: that is what makes it stick."
        : "Right. Move on.";
    } else {
      cls = rec.conf === "sure" ? "flag" : "gap";
      verdict = rec.conf === "sure"
        ? "This is the most useful answer to get wrong: you were certain, so you hold a definite idea that isn’t true. The note under your pick names it. Reread the section before going on."
        : "Not quite. The note under your pick says what that answer assumes, and the marked answer says what is true.";
    }
    box.classList.add(cls);
    if (kind !== "probe") box.append(el("p", { text: verdict }));
    if (rec.thinking) box.append(el("p", { class: "quiz-you" }, [el("b", { text: "You wrote: " }), document.createTextNode(rec.thinking)]));
    const again = el("button", { type: "button", class: "btn ghost", text: "Answer again" });
    again.addEventListener("click", () => { delete ctx.saved[id]; ctx.changed(); render(); });
    box.append(again);
    return box;
  }

  function paintState() {
    if (!state) return;
    const rec = ctx.saved[id];
    state.textContent = !rec ? "" : rec.dontKnow ? "· didn’t know" : rec.right ? "· right" : "· wrong";
    state.className = "quiz-state" + (!rec ? "" : rec.right ? " ok" : rec.conf === "sure" ? " flag" : " gap");
  }

  render();
  paintState();
  return q;
}

// A section is "skim" when every probe question about it was right and at
// least fairly sure; any miss, guess or "don't know" makes it "read".
function sectionVerdicts(quizzes, ctx) {
  const v = {};
  for (const q of quizzes.filter(q => q.kind === "probe")) {
    const r = ctx.saved[q.id];
    if (!r) continue;
    const ok = r.right && r.conf !== "guess";
    for (const s of q.for.split(/\s+/).filter(Boolean)) v[s] = (v[s] === undefined ? true : v[s]) && ok;
  }
  return v;
}

function initProbe(quizzes, ctx) {
  const root = document.querySelector("[data-probe]");
  if (!root) return;
  const probes = quizzes.filter(q => q.kind === "probe");
  const out = root.querySelector("[data-probe-out]");
  function paint() {
    const answered = probes.filter(q => ctx.saved[q.id]).length;
    out.innerHTML = "";
    if (answered < probes.length) {
      out.append(el("p", { class: "quiz-hint", text: answered + " of " + probes.length + " answered. The result appears when all are done." }));
      return;
    }
    const v = sectionVerdicts(quizzes, ctx);
    const title = id => { const h = document.getElementById(id); return h ? h.textContent : id; };
    const skim = Object.keys(v).filter(s => v[s]), read = Object.keys(v).filter(s => !v[s]);
    for (const q of probes) {
      const r = ctx.saved[q.id];
      if (q.root.dataset.prereq && !(r.right && r.conf !== "guess")) {
        out.append(el("p", { class: "quiz-fb gap" }, [document.createTextNode("Read "), el("a", { href: q.root.dataset.prereq, text: q.root.dataset.prereqTitle }), document.createTextNode(" first. This page assumes it.")]));
      }
    }
    const list = (label, ids) => ids.length ? el("p", {}, [el("b", { text: label }), ...ids.flatMap((s, i) => [document.createTextNode(i ? ", " : " "), el("a", { href: "#" + s, text: title(s) })])]) : null;
    const parts = [list("Read closely:", read), list("You can skim:", skim)].filter(Boolean);
    if (!read.length) parts.unshift(el("p", { text: "You already hold the main ideas. Skim the page and go straight to the checks at the end of each section." }));
    out.append(...parts);
  }
  ctx.onChange(paint);
  paint();
}

function initStepMap(quizzes, ctx) {
  const toc = document.querySelector(".toc[data-stepmap]");
  if (!toc) return;
  const steps = [...document.querySelectorAll("main h2[data-step]")];
  const now = document.querySelector("[data-stepnow]") || (() => {
    const span = el("span", { class: "stepnow", "data-stepnow": "" });
    const crumb = document.querySelector(".top .crumb");
    if (crumb) crumb.after(span);
    return span;
  })();
  // Two copies of the map: inline under the lede (narrow screens) and a
  // sticky rail right of the text (wide screens). CSS shows one of them.
  const rows = [];
  function build(compact) {
    const list = el("ol", { class: "stepmap" + (compact ? " compact" : "") });
    steps.forEach((h, i) => {
      const marks = el("span", { class: "marks" });
      const link = el("a", { href: "#" + h.id }, [el("b", { text: compact && h.dataset.short ? h.dataset.short : h.textContent })]);
      if (!compact) link.append(el("span", { class: "d", text: h.dataset.step }));
      const li = el("li", {}, [link, marks]);
      list.append(li);
      rows.push({ h, li, marks, n: i + 1 });
    });
    return list;
  }
  toc.hidden = true;
  // Inline copy, shown below 1400px where there is no sidebar. It starts
  // folded so the first section stays near the top of the screen.
  toc.after(el("details", { class: "stepfold" }, [el("summary", { text: "Steps on this page (" + steps.length + ")" }), build(false)]));
  const layout = document.querySelector(".layout");
  if (layout) {
    layout.classList.add("has-rail");
    layout.append(el("aside", { class: "pagenav", "aria-label": "On this page" }, [el("p", { class: "nav-label", text: "On this page" }), build(true)]));
  }

  function paint() {
    const v = sectionVerdicts(quizzes, ctx);
    for (const r of rows) {
      r.marks.innerHTML = "";
      const id = r.h.id;
      if (v[id] === true) r.marks.append(el("span", { class: "mk skim", text: "skim" }));
      if (v[id] === false) r.marks.append(el("span", { class: "mk read", text: "read" }));
      for (const q of quizzes.filter(q => q.for === id && q.kind !== "probe")) {
        const rec = ctx.saved[q.id];
        if (!rec) continue;
        if (q.kind === "predict") r.marks.append(el("span", { class: "mk", text: "guessed" }));
        else r.marks.append(el("span", { class: "mk " + (rec.right ? (rec.conf === "guess" ? "gap" : "ok") : rec.conf === "sure" ? "flag" : "gap"), text: rec.right ? (rec.conf === "guess" ? "✓ but guessed" : "✓ checked") : rec.dontKnow ? "? gap" : rec.conf === "sure" ? "✗ recheck" : "✗ missed" }));
      }
    }
  }
  ctx.onChange(paint);
  paint();

  // "You are here": the last step heading above the top fifth of the
  // screen, or the last step once the page is scrolled to the bottom.
  function here() {
    const y = innerHeight * 0.2;
    let cur = 0;
    steps.forEach((h, i) => { if (h.getBoundingClientRect().top < y) cur = i + 1; });
    if (cur && innerHeight + scrollY >= document.documentElement.scrollHeight - 4) cur = steps.length;
    rows.forEach(r => r.li.classList.toggle("here", r.n === cur));
    now.textContent = cur ? "Step " + cur + " of " + steps.length : "";
  }
  addEventListener("scroll", here, { passive: true });
  here();
}

function initRecord(quizzes, ctx) {
  const root = document.querySelector("[data-record]");
  if (!root) return;
  const sbox = root.querySelector("[data-stats]"), btn = root.querySelector("[data-copy]"), note = root.querySelector("[data-note]");
  const confWord = c => (CONFIDENCE.find(x => x[0] === c) || [, ""])[1].toLowerCase();
  function paint() {
    const checks = quizzes.filter(q => q.kind === "check");
    const done = checks.filter(q => ctx.saved[q.id]);
    const right = done.filter(q => ctx.saved[q.id].right);
    const flags = done.filter(q => !ctx.saved[q.id].right && ctx.saved[q.id].conf === "sure");
    sbox.innerHTML = "";
    for (const [v, l] of [[done.length + " / " + checks.length, "checks answered"], [right.length, "right"], [flags.length, "wrong but certain"]]) {
      sbox.append(el("div", { class: "stat" }, [el("b", { text: String(v) }), el("span", { text: l })]));
    }
  }
  function markdown() {
    const title = document.querySelector("main h1").textContent;
    const lines = ["## " + title + ": my answers (" + new Date().toISOString().slice(0, 10) + ")", ""];
    for (const q of quizzes) {
      const r = ctx.saved[q.id];
      if (!r) continue;
      const picked = r.dontKnow ? "I don't know" : tmpText(q.items[r.pick].label);
      const how = q.kind === "predict" ? "" : " (" + (r.dontKnow ? "gap" : (r.right ? "right" : "wrong") + ", " + confWord(r.conf)) + ")";
      lines.push("- **" + q.kind + "**: " + q.text);
      lines.push("  - picked: " + picked + how);
      if (r.thinking) lines.push("  - my thinking: " + r.thinking.replace(/\s*\n\s*/g, " "));
    }
    return lines.join("\n") + "\n";
  }
  function tmpText(html) { const d = el("div", { html }); return d.textContent.trim(); }
  btn.addEventListener("click", async () => {
    const md = markdown();
    try { await navigator.clipboard.writeText(md); note.textContent = "Copied. Paste it into your notes for this phase."; }
    catch (e) { note.textContent = ""; note.append(el("pre", { text: md })); }
  });
  const clear = root.querySelector("[data-clear]");
  if (clear) {
    const label = clear.textContent;
    let armed = null;
    const disarm = () => { clearTimeout(armed); armed = null; clear.textContent = label; };
    clear.addEventListener("click", () => {
      const n = Object.keys(ctx.saved).length;
      if (!n) { note.textContent = "There are no answers to clear."; return; }
      if (!armed) {
        clear.textContent = "Click again to delete all " + n + " answers";
        armed = setTimeout(disarm, 5000);
        return;
      }
      disarm();
      for (const k of Object.keys(ctx.saved)) delete ctx.saved[k];
      ctx.changed();
      location.reload();
    });
  }
  ctx.onChange(paint);
  paint();
}
