# learning-ai.cafe

A textbook anyone can extend. One canonical page per idea, figures drawn for the
argument, numbers checked by code, contributors credited on the page they improved.

## Run it

```bash
pnpm install
pnpm dev        # http://localhost:4321
pnpm build      # static site in dist/, plus dist/manifest.json for the app
pnpm dev:samples  # same, but with the three sample tutorials in samples/ instead of tutorials/
```

The scripts clear Astro's content cache first (`node_modules/.astro`); without that,
deleted pages keep being built.

## Where things live

```
tutorials/<slug>/tutorial.yaml   the tutorial: title, tagline, hue, languages, prerequisites, page order
tutorials/<slug>/pages/<lang>/<page>.md   the page body, one folder per language (en, pt-BR)
tutorials/<slug>/figures/*.svg   built figures (sources live in the private diagrams repo)
ledger.json                      who did what, appended by CI from merged pull requests
samples/                         three fake tutorials and a fake ledger, for looking at the design (SAMPLES=1)
src/                             the platform: layouts, routes, styles, the client scripts, the Markdown plugins
src/scripts/                     site.js is the entry; prefs, book, contributions, modals, dom are the pieces
```

A page that is listed in `tutorial.yaml` but has no `.md` file is shown as planned.
A page missing from a language folder falls back to the default language with a notice.

## Writing a page

Plain Markdown, plus four conventions:

```markdown
## A heading opens a section chips, anchors and the on-page rail hang off it

==the one highlighter== use it once or twice per page
$x$ inline, and display maths in a fence:

$$
q \cdot k = \sum_i q_i k_i
$$

:::figure{#figure_name .narrow} .narrow is optional
![alt text that says what to see](../../figures/figure_name.svg)

The caption. It states the takeaway, not the contents.
:::

:::aside[Title of the note]
A collapsible note for something that would break the flow.
:::
```

Front matter: `title`, `subtitle`, and optionally `verified`, a sentence naming the
script that recomputes the page's numbers.

## Licences

Text and figures CC BY-NC-SA 4.0. Platform code MIT.
