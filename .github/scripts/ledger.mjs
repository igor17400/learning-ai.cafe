// Append the merged pull request to ledger.json.
//
// Reads the pull request from GITHUB_EVENT_PATH and its files from the GitHub API,
// works out which pages and sections were touched, and writes one ledger entry per
// page. Type and score come from labels (`type:fix`, `score:3`) or from defaults
// that CONTRIBUTING.md documents. Run with no GITHUB_EVENT_PATH to self-test.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import GithubSlugger from "github-slugger";
import { prFiles } from "./pr-diff.mjs";

const DEFAULT_SCORE = {
  page: 8,
  translation: 5,
  figure: 3,
  "worked-example": 3,
  review: 2,
};

// The id of a heading the way Astro computes it: inline math and formatting stripped.
export function headingId(text, slugger = new GithubSlugger()) {
  const plain = text
    .replace(/\$([^$]+)\$/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_]+/g, "");
  return slugger.slug(plain);
}

// Line numbers (in the new file) of the added lines in a unified diff patch.
export function addedLines(patch = "") {
  const lines = [];
  let n = 0;
  for (const l of patch.split("\n")) {
    const hunk = /^@@ -\d+(?:,\d+)? \+(\d+)/.exec(l);
    if (hunk) n = Number(hunk[1]);
    else if (l.startsWith("+")) lines.push(n++);
    else if (!l.startsWith("-")) n++;
  }
  return lines;
}

// Which sections (## headings) the given new-file lines fall under.
export function sectionsTouched(markdown, lines) {
  const heads = [];
  const slugger = new GithubSlugger();
  markdown.split("\n").forEach((l, i) => {
    const m = /^## (.+)$/.exec(l);
    if (m) heads.push({ line: i + 1, id: headingId(m[1], slugger) });
  });
  const hit = new Set();
  for (const n of lines) {
    const h = heads.filter((h) => h.line <= n).pop();
    if (h) hit.add(h.id);
  }
  return [...hit];
}

// One ledger entry per page touched by the pull request.
//   files: [{ filename, status, additions, patch }]
//   pr:    { number, login, date, title, labels: [] }
//   read:  (path) => file contents after the merge, or null
export function entriesFor(files, pr, read) {
  const label = (prefix) =>
    pr.labels.find((l) => l.startsWith(prefix))?.slice(prefix.length);
  const defaultLang = (slug) =>
    (
      /^languages:\s*\[\s*"?([^",\]]+)/m.exec(
        read(`tutorials/${slug}/tutorial.yaml`) || "",
      )?.[1] || "en"
    ).trim();
  const tutorialId = (slug) =>
    /^id:\s*"?([^"\n]+?)"?\s*$/m.exec(
      read(`tutorials/${slug}/tutorial.yaml`) || "",
    )?.[1];
  const byTutorial = {};
  for (const f of files) {
    // tutorials/<slug>/figures/<name>.svg or tutorials/<slug>/pages/<lang>/<page>.md
    const m =
      /^tutorials\/([^/]+)\/(?:(figures)\/([^/]+\.svg)|pages\/([^/]+)\/([^/]+\.md))$/.exec(
        f.filename,
      );
    if (!m) continue;
    const t = (byTutorial[m[1]] ??= { pages: [], figures: [] });
    if (m[2] === "figures" && f.status === "added")
      t.figures.push(m[3].replace(/\.svg$/, ""));
    if (m[4] && f.status !== "removed")
      t.pages.push({ ...f, lang: m[4], name: m[5] });
  }
  const out = [];
  for (const [slug, t] of Object.entries(byTutorial)) {
    const id = tutorialId(slug);
    if (!id) continue;
    // figures with no page change still deserve an entry: hang them on the page that uses them
    if (!t.pages.length && t.figures.length) {
      for (const fig of t.figures) {
        const page = files.find((f) =>
          f.filename.startsWith(`tutorials/${slug}/pages/`),
        )?.filename;
        out.push({
          page: `${id}/${page ? page.split("/").pop().replace(/\.md$/, "") : "?"}`,
          date: pr.date,
          by: `@${pr.login}`,
          type: label("type:") || "figure",
          score: Number(label("score:")) || DEFAULT_SCORE.figure,
          what: pr.title,
          pr: pr.number,
          figures: [fig],
        });
      }
      continue;
    }
    for (const p of t.pages) {
      const pageSlug = p.name.replace(/\.md$/, "");
      const lang = p.lang === defaultLang(slug) ? null : p.lang;
      const type =
        label("type:") ||
        (lang
          ? "translation"
          : p.status === "added"
            ? "page"
            : t.figures.length && p.additions < 10
              ? "figure"
              : "fix");
      const score =
        Number(label("score:")) ||
        DEFAULT_SCORE[type] ||
        Math.min(5, Math.max(1, Math.ceil(p.additions / 40)));
      const entry = {
        page: `${id}/${pageSlug}`,
        date: pr.date,
        by: `@${pr.login}`,
        type,
        score,
        what: pr.title,
        pr: pr.number,
      };
      if (lang) entry.lang = lang;
      const body = read(p.filename);
      const sections =
        body && p.status !== "added"
          ? sectionsTouched(body, addedLines(p.patch))
          : [];
      if (sections.length) entry.sections = sections;
      if (t.figures.length) entry.figures = t.figures;
      out.push(entry);
    }
  }
  return out;
}

function hue(s) {
  let h = 0;
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}

function main() {
  const event = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH, "utf8"),
  ).pull_request;
  const gh = (args) =>
    JSON.parse(execFileSync("gh", ["api", ...args], { encoding: "utf8" }));
  const files = prFiles(process.env.REPO, event.number);
  const pr = {
    number: event.number,
    login: event.user.login,
    date: event.merged_at.slice(0, 10),
    title: event.title,
    labels: event.labels.map((l) => l.name),
  };
  const read = (p) => (existsSync(p) ? readFileSync(p, "utf8") : null);
  const entries = entriesFor(files, pr, read);
  if (!entries.length)
    return console.log("no tutorial pages touched, nothing to record");
  const ledger = JSON.parse(readFileSync("ledger.json", "utf8"));
  const handle = `@${pr.login}`;
  if (!ledger.users[handle]) {
    const u = gh([`users/${pr.login}`]);
    ledger.users[handle] = {
      name: u.name || pr.login,
      hue: hue(pr.login),
      github: pr.login,
    };
  }
  ledger.contributions.push(...entries);
  writeFileSync("ledger.json", JSON.stringify(ledger, null, 2) + "\n");
  console.log(
    `recorded ${entries.length} entr${entries.length === 1 ? "y" : "ies"} for ${handle}:`,
    entries.map((e) => e.page).join(", "),
  );
}

function selfTest() {
  const md =
    "---\ntitle: x\n---\n\n## Where $Q$, $K$ and $V$ come from\n\nold\nold\n\n## Layer normalisation (LN)\n\nold\nnew line\n";
  const patch =
    "@@ -8,3 +8,4 @@\n old\n \n ## Layer normalisation (LN)\n+new line\n";
  const files = [
    {
      filename: "tutorials/attn/pages/en/self_attention.md",
      status: "modified",
      additions: 1,
      patch,
    },
    {
      filename: "tutorials/attn/figures/new_fig.svg",
      status: "added",
      additions: 30,
    },
  ];
  const read = (p) =>
    p.endsWith("tutorial.yaml")
      ? 'id: "cafe/attn"\nlanguages: [en, pt-BR]\n'
      : md;
  const [e] = entriesFor(
    files,
    {
      number: 7,
      login: "ana",
      date: "2026-09-21",
      title: "Fix LN",
      labels: ["type:fix"],
    },
    read,
  );
  console.assert(
    e.page === "cafe/attn/self_attention" && e.type === "fix" && e.score === 1,
    e,
  );
  console.assert(e.sections.join() === "layer-normalisation-ln", e.sections);
  console.assert(e.figures.join() === "new_fig", e.figures);
  console.assert(
    headingId("Where $Q$, $K$ and $V$ come from") ===
      "where-q-k-and-v-come-from",
  );
  const [tr] = entriesFor(
    [
      {
        filename: "tutorials/attn/pages/pt-BR/self_attention.md",
        status: "added",
        additions: 90,
      },
    ],
    {
      number: 8,
      login: "ana",
      date: "2026-09-22",
      title: "Traduzir",
      labels: [],
    },
    read,
  );
  console.assert(
    tr.type === "translation" && tr.lang === "pt-BR" && tr.score === 5,
    tr,
  );
  console.log("self-test ok");
}

process.env.GITHUB_EVENT_PATH ? main() : selfTest();
