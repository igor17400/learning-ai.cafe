// Lint tutorials before a pull request. The build checks the manifest schema;
// this checks what the schema cannot: figures, headings, blocks, links, the
// ledger. Usage: node scripts/check-tutorial.mjs [slug]   (SAMPLES=1 for samples/)
import { readdirSync, readFileSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import GithubSlugger from "github-slugger";

const root = process.env.SAMPLES ? "samples" : "tutorials";
const ledger = JSON.parse(
  readFileSync(
    process.env.SAMPLES ? "samples/ledger.json" : "ledger.json",
    "utf8",
  ),
);
const only = process.argv[2];
let errors = 0,
  warnings = 0;
const err = (where, msg) => {
  errors++;
  console.log(`  ✗ ${where}: ${msg}`);
};
const warn = (where, msg) => {
  warnings++;
  console.log(`  · ${where}: ${msg}`);
};

export const headingId = (text, slugger = new GithubSlugger()) =>
  slugger.slug(
    text
      .replace(/\$([^$]+)\$/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/[*_]+/g, ""),
  );

function sections(md) {
  const slugger = new GithubSlugger();
  return [...md.matchAll(/^## (.+)$/gm)].map((m) => headingId(m[1], slugger));
}

function checkPage(dir, file, manifest, writtenSlugs) {
  const where = `${file}`;
  const md = readFileSync(join(dir, "pages", file), "utf8");
  const fm = /^---\n([\s\S]*?)\n---/.exec(md);
  if (!fm) err(where, "no front matter");
  else
    for (const k of ["title", "subtitle"])
      if (!new RegExp(`^${k}:`, "m").test(fm[1]))
        err(where, `front matter lacks ${k}`);
  const body = fm ? md.slice(fm[0].length) : md;

  const ids = sections(body);
  if (ids.length < 2)
    err(
      where,
      `${ids.length} section(s); a chapter needs at least two ## headings`,
    );
  const dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
  if (dupes.length) err(where, `duplicate section ids: ${dupes.join(", ")}`);
  if (/^# /m.test(body))
    err(where, "a # heading; the title comes from front matter, use ##");

  const opens = (body.match(/^:::\w/gm) || []).length,
    closes = (body.match(/^:::\s*$/gm) || []).length;
  if (opens !== closes)
    err(where, `${opens} ::: blocks opened, ${closes} closed`);
  const hl = (body.match(/==/g) || []).length;
  if (hl % 2) err(where, "unbalanced == highlight markers");
  if (hl / 2 > 2)
    warn(where, `${hl / 2} highlights; the convention is one or two`);
  if (hl === 0) warn(where, "no ==highlight==");
  if ((body.match(/^\$\$\s*$/gm) || []).length % 2)
    err(where, "unbalanced $$ fences");
  const html = [
    ...body.matchAll(/<(?!\/|u>|\/u>|br|sup|sub)[a-z][a-z0-9]*[\s>]/g),
  ].map((m) => m[0].trim());
  if (html.length) warn(where, `raw HTML: ${[...new Set(html)].join(" ")}`);

  const used = new Set();
  for (const m of body.matchAll(
    /^:::figure\{#([a-z0-9_]+)([^}]*)\}\n([\s\S]*?)^:::/gm,
  )) {
    const [, id, , inner] = m;
    const img =
      /^!\[([^\]]*)\]\(\.\.\/\.\.\/figures\/([a-z0-9_]+)\.svg\)/m.exec(inner);
    if (!img) {
      err(where, `figure ${id}: no ![alt](../../figures/<name>.svg) line`);
      continue;
    }
    if (img[2] !== id)
      err(
        where,
        `figure ${id}: image file is ${img[2]}.svg, id and file must match`,
      );
    if (!img[1].trim()) err(where, `figure ${id}: empty alt text`);
    if (!inner.replace(img[0], "").trim())
      err(where, `figure ${id}: no caption`);
    if (!existsSync(join(dir, "figures", `${img[2]}.svg`)))
      err(where, `figure ${id}: figures/${img[2]}.svg does not exist`);
    used.add(img[2]);
  }
  for (const m of body.matchAll(/\]\(\.\.\/([a-z0-9_]+)\/(?:#[^)]*)?\)/g))
    if (!writtenSlugs.has(m[1]))
      err(where, `link to ../${m[1]}/ but that chapter is not written`);
  const words = body.split(/\s+/).filter(Boolean).length;
  if (words < 800) warn(where, `${words} words, short for a chapter`);
  if (words > 3000) warn(where, `${words} words, consider splitting`);
  return { ids, used, words };
}

function checkTutorial(slug) {
  const dir = join(root, slug);
  console.log(`\n${slug}`);
  const yaml = readFileSync(join(dir, "tutorial.yaml"), "utf8");
  const id = /^id:\s*"?([^"\n]+?)"?\s*$/m.exec(yaml)?.[1];
  const languages = (/^languages:\s*\[([^\]]*)\]/m.exec(yaml)?.[1] || "en")
    .split(",")
    .map((s) => s.trim().replace(/"/g, ""));
  const pages = [
    ...yaml.matchAll(
      /"?slug"?:\s*"?([a-z0-9_]+)"?[^}\n]*?(?:"?status"?:\s*"?(\w+)"?)?/g,
    ),
  ].map((m) => ({ slug: m[1], status: m[2] || "written" }));
  // page files as "<lang>/<page>.md"; the first language is the default
  const pageFiles = languages.flatMap((l) =>
    existsSync(join(dir, "pages", l))
      ? readdirSync(join(dir, "pages", l))
          .filter((f) => f.endsWith(".md"))
          .map((f) => `${l}/${f}`)
      : [],
  );
  for (const l of existsSync(join(dir, "pages"))
    ? readdirSync(join(dir, "pages"))
    : [])
    if (!languages.includes(l))
      err(
        `pages/${l}`,
        "language folder not listed in tutorial.yaml languages",
      );
  const defaultLang = languages[0];
  const written = new Set(
    pages
      .filter(
        (p) =>
          p.status === "written" &&
          pageFiles.includes(`${defaultLang}/${p.slug}.md`),
      )
      .map((p) => p.slug),
  );
  for (const p of pages)
    if (
      p.status === "written" &&
      !pageFiles.includes(`${defaultLang}/${p.slug}.md`)
    )
      warn(
        `tutorial.yaml`,
        `${p.slug} is "written" but pages/${defaultLang}/${p.slug}.md is missing (renders as planned)`,
      );
  for (const f of pageFiles) {
    const s = f.split("/")[1].replace(/\.md$/, "");
    if (!pages.some((p) => p.slug === s))
      err(f, "page file not listed in tutorial.yaml");
  }

  const used = new Set(),
    idsByPage = {};
  for (const f of pageFiles) {
    const [lang, slug] = f.replace(/\.md$/, "").split("/");
    const r = checkPage(dir, f, yaml, written);
    r.used.forEach((u) => used.add(u));
    idsByPage[f] = r.ids;
    const src = idsByPage[`${defaultLang}/${slug}.md`];
    if (lang !== defaultLang && src && src.length !== r.ids.length)
      warn(f, `${r.ids.length} sections, source has ${src.length}`);
  }
  if (existsSync(join(dir, "figures")))
    for (const f of readdirSync(join(dir, "figures")).filter((f) =>
      f.endsWith(".svg"),
    )) {
      const name = f.replace(/\.svg$/, "");
      if (!used.has(name)) warn(`figures/${f}`, "referenced by no page");
      const size = statSync(join(dir, "figures", f)).size;
      if (size > 60_000)
        warn(
          `figures/${f}`,
          `${Math.round(size / 1024)} KB, over the 60 KB guideline`,
        );
      const head =
        /<svg[^>]*>/.exec(readFileSync(join(dir, "figures", f), "utf8"))?.[0] ||
        "";
      if (!/viewBox=/.test(head)) warn(`figures/${f}`, "no viewBox");
      else if (/\s(width|height)=/.test(head))
        warn(
          `figures/${f}`,
          "fixed width/height on the root; the page cannot scale it",
        );
    }
  for (const c of ledger.contributions.filter((c) =>
    c.page.startsWith(`${id}/`),
  )) {
    const page = c.page.split("/").pop();
    const ids = idsByPage[`${page}${c.lang ? "." + c.lang : ""}.md`];
    if (!ids) continue;
    for (const s of c.sections || [])
      if (!ids.includes(s))
        err(
          `ledger ${c.page}`,
          `section "${s}" no longer exists (a heading was renamed?)`,
        );
  }
}

const slugs = readdirSync(root).filter((d) =>
  existsSync(join(root, d, "tutorial.yaml")),
);
for (const s of slugs) if (!only || s === only) checkTutorial(s);
if (only && !slugs.includes(only)) {
  console.log(`no tutorial "${only}" in ${root}/`);
  process.exit(2);
}
console.log(`\n${errors} error(s), ${warnings} warning(s)`);
process.exit(errors ? 1 : 0);
