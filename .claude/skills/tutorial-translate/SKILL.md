---
name: tutorial-translate
description: Translate a learning-ai.cafe chapter into another language (pages/<locale>/<page>.md), keeping structure, maths, figures and section ids aligned with the source. Use for pt-BR or any locale listed in tutorial.yaml.
---

# Translate a chapter

A translation is the same file name in the language's folder: `pages/en/intro.md`
becomes `pages/pt-BR/intro.md`. Same figures, same URL under `/pt-BR/`.
Chapters without a translation fall back to the default language with a notice,
so translating chapter by chapter is fine.

## Rules

- The locale must be listed in `tutorial.yaml` `languages`, else the route is
  not built.
- Front matter `title` and `subtitle` are translated; they replace the manifest
  titles on the page.
- Keep the section structure one to one: same number of `##` headings, same
  order, same figures, same asides. The ledger credits sections by id, and the
  contents modal lists them; a translation that merges or splits sections is a
  rewrite, not a translation.
- Translate heading text naturally. Ids are computed per language, so the
  Portuguese heading gets its own id; nothing to keep in sync there.
- Maths, code, figure file names and the figure id stay exactly as in the source.
  Translate alt text and captions.
- The running example stays in the source language when it is data ("the cat
  chased the mouse" are tokens the figures show). Explain it once in the
  translation the first time it appears.
- Notation words follow the field's usage in the target language (pt-BR:
  "produto escalar", "softmax", "atenção", "cabeças"), and stay consistent
  across the tutorial. Keep a glossary at the top of the first translated chapter
  as a `:::note[Glossário]` if terms could surprise.
- Numbers use the source's decimal point in maths and code, and the locale's in
  prose only if the whole tutorial does so consistently. Simplest: keep the point.

## Credit

A translation pull request is recorded as `type: translation` on the page it
translates, with `lang` set, and appears on the leaderboard like any other
contribution.

## Check

```bash
pnpm check:tutorial <slug>
pnpm dev   # open /pt-BR/<slug>/<page>/ and switch language in the settings
```
