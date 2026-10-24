# Writing a guided page

Every new or rewritten page is a guided page: each section sets up a problem,
asks the reader to guess, explains, and ends with a check. Start from
`_template.html`; `phase-1/rust-memory.html` is the worked example. The writing
style rules (question titles, problem first, define terms, bullets after code,
`SAFETY:` comments) still apply on top of this list.

Run the checker from the repository root before committing. It lives in the
repository, not on the published site: [`scripts/check-guided-pages.py`](https://github.com/seshuthota/Aether/blob/main/scripts/check-guided-pages.py).

```
scripts/check-guided-pages.py site/phase-N/topic.html
```

It enforces the items marked ✔︎. The rest need a human read.

## Structure

- [ ] **Explanation before any question.** The first section, `#goal`, says what
      the topic is, why the engine needs it and what the reader can do by the
      end. It has no guess and no check. ✔︎
- [ ] **One box under the lede:** "How to use this page", with the optional skim
      test folded inside it (`<details class="probe">`). Delete the skim test if
      the page has none.
- [ ] **Step map:** one `<a>` per `<h2>` in `<nav class="toc" data-stepmap>`, same
      order. ✔︎
- [ ] **Every `<h2>`** is a question, with a one-line summary in `data-step` and a
      label of three words or fewer in `data-short` (shown in the "On this page"
      sidebar). ✔︎
- [ ] **Every content section** has, in this order: setup, one guess
      (`data-quiz="predict"`), the explanation, a `Takeaway:` line in bold, one
      check (`data-quiz="check"`). ✔︎ for the order.
- [ ] **The guess comes after its setup:** every term it uses is defined above
      it. It never opens a page or a section cold.
- [ ] **Ends with** `#check` (the answer record widget, with the
      `data-understood` slot for the "I can explain this" tickbox) and
      `#takeaway`. Copy both from the template.

## Questions

- [ ] Each quiz has a unique `id`, exactly one `data-right` option, and a
      `<div class="why">` under every option. ✔︎
- [ ] `data-for` names the section the quiz belongs to. ✔︎
- [ ] **Each wrong option stands for one wrong idea.** Its note says why it is
      tempting and what is true. No joke options.
- [ ] **A check applies the idea** to a new case and is answerable from that
      section alone. It does not ask the reader to repeat a sentence.
- [ ] **Notes add no new claims.** Anything technical in a note is either
      already on the page or flagged for review.

## Code and length

- [ ] Code lines are 72 characters or fewer; long comments go on their own
      line. ✔︎
- [ ] Quizzes add length: keep each note to one or two sentences, and keep the
      explanation text itself within the usual 1.5× of the old page.

## Wiring

- [ ] Load `js/learn.js` only on pages with quizzes, after `site.js`. ✔︎
- [ ] Anything that only makes sense with scripts running (instructions to
      guess or answer, the skim test) gets `class="js-only"`. With scripts off,
      each quiz shows as its question, the right answer and that answer's note.
- [ ] Fonts are self-hosted from `fonts/` through `site.css`; don't add Google
      Fonts links to pages.
- [ ] Bump the `?v=` on every asset when `site.css` or a script changes.
- [ ] Add the page to `js/pages.js`; check links and anchors.
