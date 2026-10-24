#!/usr/bin/env node
// Runtime test for guided pages: one quiz with broken markup must not blank
// the page. Serves site/ locally, loads a copy of a guided page with one
// quiz's id removed (a typical copy-paste mistake), and checks in headless
// Chrome that:
//   - the broken quiz is marked is-fallback and shows its question, its right
//     answer and that answer's note, with the wrong options hidden;
//   - every other quiz is built (is-ready);
//   - the step map and the "On this page" sidebar are built;
//   - after the 5 s <head> timeout the page is still in scripts-on mode.
// It also loads the unmodified page and checks that every quiz is built.
//
//   scripts/test-guided-fallback.mjs [site/phase-1/rust-memory.html]
//
// Needs Node 22 or newer (for its built-in WebSocket), google-chrome (or
// CHROME=/path/to/chrome) and python3 on the PATH.
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, rmSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

if (typeof WebSocket === "undefined") {
  console.error("This test needs Node 22 or newer (built-in WebSocket); this is Node " + process.versions.node + ".");
  process.exit(2);
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const site = join(root, "site");
const page = resolve(process.argv[2] || join(site, "phase-1/rust-memory.html"));
const copy = join(dirname(page), "zz-fallback-test.html");
const PORT = 8790 + Math.floor(Math.random() * 100), DEBUG = PORT + 200;
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Break the first check quiz by dropping its id.
const html = readFileSync(page, "utf8");
const m = html.match(/<div class="quiz" data-quiz="check"([^>]*) id="([^"]+)">/);
if (!m) { console.error("no check quiz with an id found in " + page); process.exit(2); }
const brokenFor = (m[1].match(/data-for="([^"]+)"/) || [])[1];
writeFileSync(copy, html.replace(m[0], `<div class="quiz" data-quiz="check"${m[1]}>`));

const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1", "--directory", site], { stdio: "ignore" });
const profile = mkdtempSync(join(tmpdir(), "guided-test-"));
const chrome = spawn(process.env.CHROME || "google-chrome", ["--headless=new", `--remote-debugging-port=${DEBUG}`, `--user-data-dir=${profile}`, "--window-size=1500,1000", "about:blank"], { stdio: "ignore" });
const cleanup = async () => {
  rmSync(copy, { force: true });
  server.kill();
  // Chrome keeps writing its profile until it has exited.
  const exited = new Promise(r => chrome.once("exit", r));
  chrome.kill();
  await Promise.race([exited, sleep(3000)]);
  try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch { /* temp dir; the OS cleans it */ }
};

let failures = 0;
const ok = (cond, msg) => { console.log((cond ? "  ok   " : "  FAIL ") + msg); if (!cond) failures++; };

try {
  let target;
  for (let i = 0; i < 60 && !target; i++) {
    try { target = (await (await fetch(`http://127.0.0.1:${DEBUG}/json`)).json()).find(t => t.type === "page"); } catch { /* not up yet */ }
    if (!target) await sleep(250);
  }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(r => ws.onopen = r);
  let id = 0; const pending = {};
  ws.onmessage = e => { const msg = JSON.parse(e.data); if (msg.id && pending[msg.id]) { pending[msg.id](msg); delete pending[msg.id]; } };
  const send = (method, params = {}) => new Promise(r => { const i = ++id; pending[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async expr => (await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true })).result.result.value;
  const load = async file => {
    const url = `http://127.0.0.1:${PORT}/${relative(site, file)}`;
    await send("Page.navigate", { url });
    await sleep(6000);  // past the 5 s <head> timeout
  };
  const state = `(() => {
    const qs = [...document.querySelectorAll(".quiz")];
    const vis = e => !!e && e.checkVisibility({ visibilityProperty: true });
    const fb = document.querySelector(".quiz.is-fallback");
    return {
      js: document.documentElement.classList.contains("js"),
      total: qs.length,
      ready: qs.filter(q => q.classList.contains("is-ready")).length,
      fallback: qs.filter(q => q.classList.contains("is-fallback")).length,
      fallbackFor: fb && fb.dataset.for,
      question: vis(fb && fb.querySelector(":scope > p")),
      answer: vis(fb && fb.querySelector("li[data-right]")),
      note: vis(fb && fb.querySelector("li[data-right] .why")),
      wrongShown: fb ? [...fb.querySelectorAll("li:not([data-right])")].filter(vis).length : -1,
      steps: document.querySelectorAll("h2[data-step]").length,
      inline: document.querySelectorAll(".stepfold .stepmap li").length,
      rail: document.querySelectorAll(".pagenav .stepmap li").length
    };
  })()`;

  await send("Page.enable");
  console.log("broken quiz (" + relative(root, page) + ", quiz #" + m[2] + " without its id):");
  await load(copy);
  const b = await evaluate(state);
  ok(b.fallback === 1 && b.fallbackFor === brokenFor, `the broken quiz is marked is-fallback (${b.fallback} fallback)`);
  ok(b.question && b.answer && b.note, "it shows its question, right answer and note");
  ok(b.wrongShown === 0, "its wrong options are hidden");
  ok(b.ready === b.total - 1, `every other quiz is built (${b.ready} of ${b.total - 1})`);
  ok(b.inline === b.steps && b.rail === b.steps, `step map and sidebar are built (${b.inline} and ${b.rail} of ${b.steps} steps)`);
  ok(b.js, "after the 5 s timeout the page is still in scripts-on mode");

  console.log("unmodified page:");
  await load(page);
  const n = await evaluate(state);
  ok(n.ready === n.total && n.fallback === 0, `every quiz is built (${n.ready} of ${n.total})`);
  ok(n.js, "the page is in scripts-on mode");
} finally {
  await cleanup();
}
console.log(failures ? failures + " failed" : "all passed");
process.exit(failures ? 1 : 0);
