# learning-ai.cafe

A collaborative textbook of AI tutorials. One tutorial is one folder of Markdown
chapters; the platform (Astro, static, no client framework) turns it into a site
with a book-reading mode and per-section contributor credit.

## Commands

```bash
pnpm dev            # site with the real tutorials in tutorials/
pnpm dev:samples    # site with the three fake tutorials in samples/ (design work)
pnpm build          # static site in dist/; the build is also the schema check
pnpm check:tutorial <slug>   # lint one tutorial (or all): figures, headings, links, blocks
pnpm format         # Prettier over the repo (a pre-commit hook does this for staged files)
```

Every script clears Astro's content cache first. If pages you deleted keep
appearing, `rm -rf node_modules/.astro`.

## Where things live

```
tutorials/<slug>/tutorial.yaml    manifest: id, title, track, tagline, hue, cover, languages, pages
tutorials/<slug>/pages/<lang>/<page>.md   one chapter per file, one folder per language
tutorials/<slug>/figures/*.svg    built figures (sources live in ../diagrams, outside git); pages link them as ../../figures/<name>.svg
samples/                          fake tutorials + fake ledger, only for looking at the design
ledger.json                       who did what; written by CI on merge, never by hand
src/                              the platform; see src/README notes in each file header
src/covers/*.svg                  the crests a tutorial can pick as its cover
src/i18n/<locale>.ts              every interface string, one file per language
.claude/skills/tutorial-*         how to scaffold, write, illustrate, translate and check a tutorial
```

## Writing a tutorial: the short version

Use the skills. `tutorial-scaffold` creates the folder, `tutorial-page` is the
house style for a chapter, `tutorial-figure` specifies figures, `tutorial-check`
lints before a pull request. They contain the full conventions; the essentials:

- A chapter is plain Markdown plus four things: `## ` headings open sections,
  `==text==` highlights the one sentence that matters (once or twice per page),
  `$...$` / `$$` for maths, and three `:::` blocks: `figure`, `aside`, `note`.
- Heading text becomes the section id that credit hangs on. Changing a heading
  after merge orphans the credit for that section. Prefer adding a sentence over
  renaming a heading.
- Every figure has alt text that says what to see and a caption that states the
  takeaway, not the contents.
- Numbers in prose must agree with the code or table next to them.
- Tone: explain, from first intuition to the final architecture. No marketing,
  no "in this section we will".

## Rules for agents working here

- Do not edit `ledger.json`. CI appends to it from merged pull requests.
- Do not commit `dist/`, `.astro/`, `node_modules/.astro`.
- Do not reformat files by hand; Prettier runs on commit. Keep the config.
- New interface text goes in `src/i18n/en.ts` and `src/i18n/pt-BR.ts`, both.
- New crests go in `src/covers/` with their source and licence in `CREDITS.md`.
- `samples/` is design scaffolding. Real content goes in `tutorials/`.
- The site must stay fast: no client-side frameworks, no third-party scripts,
  fonts self-hosted in `public/fonts/`.
