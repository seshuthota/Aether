#!/usr/bin/env python3
"""Check notes-site pages against site/GUIDED-PAGES.md.

  scripts/check-guided-pages.py                 # every page in site/
  scripts/check-guided-pages.py site/phase-1/rust-memory.html

A page is "guided" when it has <nav class="toc" data-stepmap>. Guided pages
get every check below; other pages only get the learn.js rule. Exits 1 if
any guided page fails.
"""
import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
SPECIAL = {"goal", "check", "takeaway"}  # sections without a guess and a check
MAX_CODE = 72


class Page(HTMLParser):
    """Collects, in document order, the elements the rules talk about."""

    def __init__(self):
        super().__init__()
        self.events = []      # (kind, attrs) in order: h2, quiz, takeaway
        self.toc = []         # hrefs inside the step-map toc
        self.in_toc = False
        self.stepmap = False
        self.learn_js = False
        self.learn_js_fallback = False
        self.quiz_stack = []  # open quiz divs, to collect their options
        self.depth = 0
        self.strong = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "nav" and "toc" in a.get("class", "").split():
            self.in_toc = True
            self.stepmap = "data-stepmap" in a
        if tag == "a" and self.in_toc:
            self.toc.append(a.get("href", ""))
        if tag == "script" and "learn.js" in a.get("src", ""):
            self.learn_js = True
            self.learn_js_fallback = "classList.remove" in a.get("onerror", "")
        if tag == "h2":
            self.events.append(("h2", a))
        if tag == "div":
            self.depth += 1
            if "data-quiz" in a:
                q = dict(a, options=[], depth=self.depth)
                self.events.append(("quiz", q))
                self.quiz_stack.append(q)
            elif "why" in a.get("class", "").split() and self.quiz_stack:
                opts = self.quiz_stack[-1]["options"]
                if opts:
                    opts[-1]["why"] = True
        if tag == "li" and self.quiz_stack:
            self.quiz_stack[-1]["options"].append({"right": "data-right" in a, "why": False})
        if tag == "strong":
            self.strong = True

    def handle_endtag(self, tag):
        if tag == "nav":
            self.in_toc = False
        if tag == "div":
            if self.quiz_stack and self.quiz_stack[-1]["depth"] == self.depth:
                self.quiz_stack.pop()
            self.depth -= 1
        if tag == "strong":
            self.strong = False

    def handle_data(self, data):
        if self.strong and data.strip().startswith("Takeaway:"):
            self.events.append(("takeaway", {}))


def code_lines(text):
    for m in re.finditer(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", text, re.S):
        for line in html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).split("\n"):
            if len(line) > MAX_CODE:
                yield line


def check(path):
    text = path.read_text()
    p = Page()
    p.feed(text)
    errors = []
    quizzes = [a for k, a in p.events if k == "quiz"]
    if bool(quizzes) != p.learn_js:
        errors.append("load js/learn.js if and only if the page has quizzes")
    if not p.stepmap:
        return errors, False

    # The inline <head> script, with its comments removed, so a comment that
    # merely mentions the code can't stand in for it.
    head = text[:text.find("</head>")]
    m = re.search(r"<script>(.*?)</script>", head, re.S)
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", m.group(1), flags=re.S) if m else ""
    if 'classList.add("js")' not in code or m.start() > head.find("css/site.css"):
        errors.append('add the classList.add("js") script in <head>, before site.css (copy it from _template.html)')
    if "setTimeout" not in code or ".quiz:not(.is-ready):not(.is-fallback)" not in code or 'classList.remove("js")' not in code:
        errors.append("the <head> script needs the 5 s timeout from _template.html")
    if p.learn_js and not p.learn_js_fallback:
        errors.append("the learn.js <script> needs onerror=\"document.documentElement.classList.remove('js')\"")

    h2s = [a for k, a in p.events if k == "h2"]
    ids = [a.get("id") for a in h2s]
    if [h.lstrip("#") for h in p.toc] != ids:
        errors.append("step-map links must match the h2 ids in order")
    for a in h2s:
        for attr in ("data-step", "data-short"):
            if not a.get(attr):
                errors.append(f"h2 #{a.get('id')}: missing {attr}")
    seen = set()
    for q in quizzes:
        qid = q.get("id")
        if not qid or qid in seen:
            errors.append(f"quiz {qid!r}: needs a unique id")
        seen.add(qid)
        if sum(o["right"] for o in q["options"]) != 1:
            errors.append(f"quiz #{qid}: exactly one option needs data-right")
        if not all(o["why"] for o in q["options"]):
            errors.append(f"quiz #{qid}: every option needs a <div class=\"why\">")

    # Walk the sections: within each, the order must be guess, takeaway, check.
    section, order = None, {}
    for kind, a in p.events:
        if kind == "h2":
            section = a.get("id")
            order[section] = []
        elif section is not None:
            if kind == "quiz" and a.get("data-quiz") != "probe":
                if a.get("data-for") != section:
                    errors.append(f"quiz #{a.get('id')}: data-for should be {section!r}")
                order[section].append(a.get("data-quiz"))
            elif kind == "takeaway":
                order[section].append("takeaway")
    for sid, seq in order.items():
        if sid in SPECIAL:
            if any(x in ("predict", "check") for x in seq):
                errors.append(f"#{sid}: no guess or check in this section")
            continue
        if seq != ["predict", "takeaway", "check"]:
            errors.append(f"#{sid}: expected guess, then 'Takeaway:', then check; found {seq}")

    for line in code_lines(text):
        errors.append(f"code line over {MAX_CODE} chars: {line.strip()[:60]}…")
    return errors, True


def main():
    files = [Path(f) for f in sys.argv[1:]] or sorted(p for p in SITE.rglob("*.html") if p.name != "_template.html")
    failed = 0
    guided = 0
    for f in files:
        errors, is_guided = check(f)
        guided += is_guided
        if errors:
            failed += is_guided
            print(f"{f}:")
            for e in errors:
                print(f"  - {e}")
    print(f"{len(files)} pages, {guided} guided, {failed} guided with problems")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
