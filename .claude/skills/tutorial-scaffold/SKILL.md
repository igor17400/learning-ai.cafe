---
name: tutorial-scaffold
description: Create the folder and manifest for a new learning-ai.cafe tutorial (tutorial.yaml, chapter stubs, figures dir, cover crest). Use when the user wants to start a new tutorial or add a chapter to the plan of an existing one.
---

# Scaffold a tutorial

A tutorial is a folder under `tutorials/`. Its manifest, `tutorial.yaml`, is
validated by the build (`src/content.config.ts` holds the schema). Everything the
home shelf, the rail, the contents modal and the app manifest show comes from it.

## 1. Settle the plan with the user

Ask, in one message, for whatever is missing:

- **Topic and title**, e.g. "Queries, Keys, and Values". Title case, short.
- **Slug**: lowercase, underscores, e.g. `queries_keys_and_values`. It is the
  folder name and the URL, so it never changes later.
- **Track**: the shelf groups by it. Existing tracks are in other manifests;
  reuse one before inventing one.
- **Tagline**: one sentence on the cover. What the reader will be able to do.
- **Chapter outline**: 6 to 12 chapters, each with slug, title and a one-line
  subtitle. Order is reading order. Chapters can start as `status: planned`.
- **Prerequisites** (`requires`): ids of other tutorials the reader needs first.
- **Cover crest**: one of the names in `src/covers/` (`lion`, `unicorn`,
  `dragon`, `griffin`, `phoenix`, `stag`, `eagle`). Pick one the topic can carry
  as a family crest; suggest one and let the user overrule.
- **Hue**: 0 to 360, the cover and accent colour. Look at existing tutorials and
  pick one that is not adjacent to a neighbour on the shelf.
- **Languages**: `[en]` or `[en, pt-BR]`. The first is the default.

## 2. Write the manifest

```yaml
schema: 1
id: cafe/<slug> # namespace/slug; the ledger keys pages by this
title: <Title>
track: <Track>
tagline: <one sentence>
cover: <crest>
hue: <0-360>
languages: [en, pt-BR]
requires: [] # other tutorial ids
related: [] # soft "see also"
pages:
  - { slug: intro, title: "Why ...?", subtitle: "..." }
  - { slug: <next>, title: "...", subtitle: "...", status: planned }
```

Rules the schema enforces: `id` is `[a-z0-9-]+/[a-z0-9_]+`, page slugs are
`[a-z0-9_]+`, `status` is `written` (default), `draft` or `planned`, `cover` is
one of the crest names.

## 3. Create the files

```
tutorials/<slug>/tutorial.yaml
tutorials/<slug>/pages/en/        # one .md per chapter you write now
tutorials/<slug>/pages/pt-BR/     # translations, same file names
tutorials/<slug>/figures/         # empty until figures arrive; git needs a .gitkeep
```

Write the first chapter with the `tutorial-page` skill. Chapters listed in the
manifest without a `.md` file render as planned in the rail, which is fine: a
tutorial can ship one chapter at a time.

## 4. Verify

```bash
pnpm check:tutorial <slug>
pnpm build
pnpm dev     # open /<slug>/<first-chapter>/ and press the book icon
```

The build fails on any schema error. The check script reports missing figures,
duplicate headings, unbalanced blocks and links to chapters that do not exist.
