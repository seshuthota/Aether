# Aether notes site

Static pages that explain the concepts behind the engine. No build step.

```
python3 -m http.server 8765 --directory site     # from the repo root
```

Then open http://localhost:8765/ (or open `site/index.html` directly).

## Adding a page

1. Copy `_template.html` to `phase-N/topic.html` and write the content,
   following the checklist in `GUIDED-PAGES.md`.
2. Add one line for it to `js/pages.js`. A new phase gets a new section there.
3. From the repository root, run `scripts/check-guided-pages.py site/phase-N/topic.html`
   ([source](https://github.com/seshuthota/Aether/blob/main/scripts/check-guided-pages.py); it is not part of the published site).

The side menu, breadcrumb, previous/next links and the home-page cards are
all built from `js/pages.js`. Nothing else needs editing.

## Adding a glossary term

Add an entry to `js/glossary.js`. Its first use on every page gets a dotted
underline and a hover/tap definition, and it appears on `glossary.html`.
If a term matches a word that means something else on one page, opt that page
out with `<main data-no-terms="term">`.

## Files

- `css/site.css` — all styles, light and dark.
- `js/pages.js` — the page list.
- `js/glossary.js` — the terms.
- `js/site.js` — navigation, glossary cards, progress ticks, and the
  interactive widgets (each one starts from a `data-*` attribute in a page).
- `js/learn.js` — guided pages only: guesses, checks, the skim test, the step
  map and "On this page" sidebar, and the answer record.

Progress ticks ("I can explain this page") are saved in the browser's
localStorage, so they are per browser and are not committed anywhere.
