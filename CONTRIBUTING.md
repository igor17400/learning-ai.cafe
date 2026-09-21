# Contributing

Every merged pull request is recorded in `ledger.json` and credited on the page it
touched. The credit is typed and scored, and the scores decide how reader support is
shared among contributors.

## Writing

One tutorial is one folder under `tutorials/`. `README.md` explains the layout and the
four Markdown conventions. Run `pnpm dev` and check the page before opening the PR.

## What happens to a pull request

1. CI builds the site. A page that breaks the schema, a missing figure, or bad front
   matter fails here.
2. A review agent reads the diff and posts a comment: does the change follow the
   conventions, do the numbers add up, is anything unclear. It never merges.
3. A code owner reviews and approves. Before merging they label the PR:
   - `type:page`, `type:fix`, `type:figure`, `type:worked-example`, `type:review`,
     `type:translation` (guessed from the diff when missing)
   - `score:N` when the default score is wrong
4. On merge, CI appends an entry to `ledger.json` for each page touched: who, when,
   what, which sections, which figures. The site rebuilds and the credit appears.

## Default scores

| type               | score                  |
| ------------------ | ---------------------- |
| page (new chapter) | 8                      |
| translation        | 5                      |
| figure             | 3                      |
| worked-example     | 3                      |
| review             | 2                      |
| fix                | 1 to 5, by lines added |

## Licences

Text and figures CC BY-NC-SA 4.0. Platform code MIT. By contributing you agree to
publish your contribution under these licences.
