---
name: tutorial-check
description: Lint a learning-ai.cafe tutorial before a pull request (figures exist, headings unique, blocks balanced, links resolve, translations aligned, ledger sections still match). Use after writing or editing any file under tutorials/.
---

# Check a tutorial

```bash
pnpm check:tutorial            # every tutorial
pnpm check:tutorial <slug>     # one
SAMPLES=1 pnpm check:tutorial  # the design samples instead
pnpm build                     # the schema check; run both
```

The script is `scripts/check-tutorial.mjs`. It reports, per page:

| check                                                                     | error or warning |
| ------------------------------------------------------------------------- | ---------------- |
| front matter has `title` and `subtitle`                                   | error            |
| at least two `##` sections, ids unique                                    | error            |
| `:::` blocks open and close                                               | error            |
| `==` highlight markers balanced                                           | error            |
| every `../figures/<name>.svg` exists, alt text non-empty, caption present | error            |
| figure block id equals the file name                                      | error            |
| links to `../<chapter>/` point at written chapters                        | error            |
| `$$` fences balanced                                                      | error            |
| translation has the same number of sections as the source                 | warning          |
| SVG in `figures/` referenced by no page                                   | warning          |
| SVG larger than 60 KB or with fixed width/height on the root              | warning          |
| more than two highlights, or none                                         | warning          |
| under 800 or over 3,000 words                                             | warning          |
| ledger entries name a section id that no longer exists                    | error            |

Fix errors before opening the pull request. Warnings are judgment calls; say in
the PR why you kept one.

When the build fails but the check passes, the problem is in `tutorial.yaml`
(schema) and the error names the field.
