// The files of a pull request, from its unified diff. The files API leaves
// `patch` and the line counts empty on pull requests whose diff is large (this
// repo's SVG figures get there quickly), so the diff is parsed here instead.
// Returns [{ filename, status, additions, patch }]; `patch` is kept only for
// the extensions in `keepPatch`, so figure bodies are dropped early.
import { execFileSync } from "node:child_process";

export function parseDiff(text, keepPatch = /\.(md|yaml)$/) {
  const files = [];
  let f = null;
  for (const line of text.split("\n")) {
    const head = /^diff --git a\/(.+?) b\/(.+)$/.exec(line);
    if (head) {
      f = {
        filename: head[2],
        status: "modified",
        additions: 0,
        patch: "",
        hunk: false,
      };
      files.push(f);
      continue;
    }
    if (!f) continue;
    if (line.startsWith("new file mode")) f.status = "added";
    else if (line.startsWith("deleted file mode")) f.status = "removed";
    else if (line.startsWith("@@") || f.hunk) {
      f.hunk = true;
      if (keepPatch.test(f.filename)) f.patch += line + "\n";
      if (line.startsWith("+") && !line.startsWith("+++")) f.additions++;
    }
  }
  for (const x of files) {
    delete x.hunk;
    if (!x.patch) x.patch = null;
  }
  return files;
}

export function prFiles(repo, pr) {
  const diff = execFileSync("gh", ["pr", "diff", String(pr), "--repo", repo], {
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  });
  return parseDiff(diff);
}

// self-test: node .github/scripts/pr-diff.mjs
if (process.argv[1]?.endsWith("pr-diff.mjs")) {
  const sample = [
    "diff --git a/tutorials/t/pages/en/a.md b/tutorials/t/pages/en/a.md",
    "new file mode 100644",
    "--- /dev/null",
    "+++ b/tutorials/t/pages/en/a.md",
    "@@ -0,0 +1,2 @@",
    "+## Heading",
    "+text",
    "diff --git a/tutorials/t/figures/x.svg b/tutorials/t/figures/x.svg",
    "new file mode 100644",
    "--- /dev/null",
    "+++ b/tutorials/t/figures/x.svg",
    "@@ -0,0 +1 @@",
    "+<svg/>",
    "diff --git a/tutorials/t/pages/en/b.md b/tutorials/t/pages/en/b.md",
    "--- a/tutorials/t/pages/en/b.md",
    "+++ b/tutorials/t/pages/en/b.md",
    "@@ -3,2 +3,3 @@",
    " old",
    "+new",
    " old",
    "",
  ].join("\n");
  const [a, svg, b] = parseDiff(sample);
  console.assert(
    a.status === "added" && a.additions === 2 && a.patch.startsWith("@@"),
    a,
  );
  console.assert(
    svg.status === "added" && svg.patch === null && svg.additions === 1,
    svg,
  );
  console.assert(
    b.status === "modified" && b.additions === 1 && b.patch.includes("+new"),
    b,
  );
  console.log("self-test ok");
}
