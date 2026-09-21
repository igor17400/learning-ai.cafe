---
name: tutorial-page
description: House style for writing or revising a learning-ai.cafe chapter (a page under tutorials/<slug>/pages). Use whenever writing tutorial prose, maths, figures blocks, asides, notes, or reviewing a chapter draft.
---

# Write a chapter

A chapter is one Markdown file, `tutorials/<slug>/pages/<lang>/<page>.md`, read either as
a web page or as two facing pages in book mode. It is also the unit of credit:
every `##` section is tracked in the ledger, so structure is not cosmetic.

## Shape

```markdown
---
title: "Why Attention?"
subtitle: "The fixed-vector bottleneck, and the idea of letting a model look things up."
verified: "codes/self_attention.py recomputes every number on this page" # optional
---

## A heading that is a claim or a question

Two to five paragraphs. The first paragraph says what problem this section
solves, in the reader's terms, before any notation appears.

:::figure{#figure_name}
![Alt text that says what to see.](../../figures/figure_name.svg)

The caption states the takeaway, not the contents.
:::

## The next section

...
```

- **Length**: 1,200 to 2,500 words. Longer means two chapters.
- **Sections**: 3 to 6 `##` headings. No `#` (the title comes from front matter),
  `###` only for sub-steps inside a derivation.
- **Heading text** becomes the section id (`github-slugger` of the text with
  inline maths reduced to its letters). Credit hangs on it. Write headings you
  will not need to rename: a claim ("Attention cannot see order on its own") or
  a question ("Why divide by the square root?"), never "Introduction" or "Part 2".
- **Front matter** `title` and `subtitle` override the manifest on the page. Keep
  them in sync with `tutorial.yaml` unless the chapter deliberately reframes.

## Voice

- From first intuition to the final architecture, in that order. Start with the
  problem a reader without the idea would hit, then the idea, then the formula,
  then what it buys and what it hides.
- One running example per tutorial, reused in every chapter (the sample uses
  "the cat chased the mouse"). Numbers from the example recur across chapters,
  so keep a note of them and make later chapters quote the same values.
- Address the reader directly, present tense, short sentences. No "in this
  section we will", no "as we saw above" without a link, no exclamation marks.
- Every formula is preceded by a sentence saying what it computes and followed
  by one saying what changed. A formula is never the first line of a section.
- Define a symbol the first time it appears in the tutorial, not in every chapter.
  Consistent notation across chapters: $Q, K, V$ matrices, $q_i, k_j, v_j$ rows,
  $d_k$ dimensions, $n$ sequence length, $h$ heads.

## The four conventions

**Highlight** `==the one sentence==`: once or twice per page, on the sentence a
reader should be able to recite afterwards. It may contain inline maths. Never
on a heading, never on a whole paragraph.

**Maths**: `$x$` inline, display maths in its own `$$` fence with blank lines
around it. Display maths is rendered at build time, so anything MathJax accepts
works, but keep it to what a reader can check by hand. Use `\cdot` for dot
products, `\top` for transpose, `\sqrt{d_k}`, `\operatorname{softmax}`.

**Figure block**:

```markdown
:::figure{#five_scores .narrow}
![One query compared against five keys, each producing one score; the largest score is the key that matches.](../../figures/five_scores.svg)

Five keys, five scores, one winner. The scores are not yet weights: they can be negative and do not sum to one.
:::
```

- The id after `#` is the figure name and must equal the file name.
- `.narrow` for figures under 32rem wide (a single vector, a small matrix).
- Alt text: what to see, one sentence, for a reader who cannot see it.
- Caption: the takeaway. If the caption could be true of any figure, rewrite it.
- Place the figure after the paragraph that needs it, never before the idea.
- Book mode caps figures at 38% of the screen height; wide, short figures work
  better than tall ones. See the `tutorial-figure` skill for making one.

**Aside and note**:

```markdown
:::aside[Why the square root]
Collapsed by default. For a derivation or a history note that would break the
flow. Two to eight sentences; if it needs a figure it is a section.
:::

:::note[Notation]
Open, with a small title. For something the reader must not miss but that is
not part of the argument: notation, a caveat, a pointer to another chapter.
:::
```

## Code and tables

- Code blocks with a language: ` ```python `. Show the minimum that
  reproduces the numbers in the text; imports and helper plumbing go in
  `codes/` if the tutorial has it, not on the page.
- If the page quotes numbers a script computes, add `verified:` to the front
  matter naming the script. The page then carries a "numbers verified" badge.
- Tables for small numeric examples (five words, five scores). Right-align numbers
  with `---:`. No tables wider than five columns; book mode is two narrow columns.

## Links

- Other chapters of the same tutorial: `[the softmax chapter](../softmax_weights/)`.
  Only link chapters that exist; the check script flags links to planned ones.
- External links only to primary sources (papers, docs). No link farms.

## Ending

The last section hands off: one paragraph on what the chapter leaves open and
which chapter picks it up. No summary bullet list.

## Before proposing the chapter

1. `pnpm check:tutorial <slug>` is clean.
2. Every `==highlight==` is a sentence the reader should remember.
3. Every caption states a takeaway.
4. Every number in prose appears in a table, a code block, or a figure nearby.
5. Read it once in book mode (`pnpm dev`, book icon): no heading stranded at the
   bottom of a page, no figure taller than a page.
