---
name: tutorial-figure
description: Specify, produce and reference a figure for a learning-ai.cafe chapter (SVG in tutorials/<slug>/figures, sources in the diagrams repo). Use when a chapter needs an illustration, when reviewing figure quality, or when exporting SVG.
---

# Figures

Figures are drawn for the argument, not decoration. One figure per idea that words
struggle with: a shape, a flow, a comparison of magnitudes. If a sentence explains
it as well, no figure.

## Two homes

- **Source** lives in the private diagrams repo, `../diagrams/<slug>/<name>.*`
  (TikZ, Python, Figma export, whatever produced it). Never in `code/`.
- **Built SVG** lives in `tutorials/<slug>/figures/<name>.svg` and is committed.
  The page references it as `../../figures/<name>.svg` (pages sit one folder deeper, per language).

Name: lowercase, underscores, says what the figure shows (`causal_mask`,
`softmax_bars`), not where it sits (`fig3`).

## Specification, before drawing

Write these four lines and agree them with the user:

1. **Claim**: the one sentence the figure proves. It becomes the caption.
2. **Elements**: what is on the canvas, in reading order (left to right, top to
   bottom). Five or fewer distinct elements.
3. **Encoding**: what carries the message: width of an arrow, height of a bar,
   colour of a cell. One encoding per figure.
4. **Numbers**: any values shown must be the tutorial's running example, and must
   match the text and the code.

## SVG requirements

- `viewBox` set; no fixed `width`/`height` on the root, or the page cannot
  scale it. Aspect ratio between 3:2 and 2:1 for full-width figures; the book
  mode caps height at 38% of the screen, so tall figures shrink badly.
- Text converted to outlines, or limited to system fonts (`Inter`, `sans-serif`).
  No web font references, no `@import`.
- Transparent background. The page draws a paper tint behind it in dark mode.
- Colours: ink `#221a14`, muted `#5a4d42`, rule `#e6dccf`, and the tutorial's
  accent hue at 45% saturation, 36% lightness (`hsl(<hue> 45% 36%)`). One accent
  plus greys is enough; two accents only when the figure compares two things.
- Stroke widths 1.5 to 2.5px at the figure's natural size. Labels 12 to 14px.
- Under 60 KB. Run it through `svgo` if the tool exported metadata.
- No embedded raster images.

## Referencing it

```markdown
:::figure{#<name>}
![<alt: what to see, one sentence>](../../figures/<name>.svg)

<caption: the takeaway>
:::
```

`.narrow` after the id for figures that should not exceed 32rem.

## Credit

A new figure in a pull request is recorded in the ledger under `figures` for the
page that references it, with the author's handle. The page shows an (i) next to
the caption that names the author. Redrawing a figure counts as a `figure`
contribution; touching its caption counts as a `fix` on the section.

## Placeholder while the real figure is pending

Commit a placeholder SVG with the same name so the page builds and the layout
can be judged, and say so in the pull request. The sample tutorials show the
placeholder style: dashed border, two boxes, an arrow, a label.
