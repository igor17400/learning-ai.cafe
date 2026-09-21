// The review agent. Reads the tutorial changes in a pull request, asks a model
// whether they follow the house conventions and hold up, and posts the answer as
// a comment. It never approves or merges; a code owner does that.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import OpenAI from "openai";

const { PR, REPO } = process.env;
const MODEL = process.env.REVIEW_MODEL || "gpt-5-mini";
const gh = (args, input) =>
  execFileSync("gh", args, { encoding: "utf8", input });

const files = JSON.parse(
  gh(["api", "--paginate", `repos/${REPO}/pulls/${PR}/files`]),
).filter((f) => /^tutorials\/.*\.(md|yaml)$/.test(f.filename) && f.patch);
if (!files.length) {
  console.log("no tutorial text changed, nothing to review");
  process.exit(0);
}
if (!process.env.OPENAI_API_KEY) {
  console.log("OPENAI_API_KEY not set, skipping the review agent");
  process.exit(0);
}

const conventions = readFileSync("README.md", "utf8");
const diff = files
  .map(
    (f) =>
      `### ${f.filename} (${f.status}, +${f.additions} -${f.deletions})\n\`\`\`diff\n${f.patch}\n\`\`\``,
  )
  .join("\n\n");

const client = new OpenAI();
const response = await client.responses.create({
  model: MODEL,
  reasoning: { effort: "medium" },
  instructions: `You review pull requests for learning-ai.cafe, a collaborative textbook of AI tutorials written in Markdown. The site's README, which defines the page conventions, follows.

${conventions}

Judge only the tutorial text in the diff. Check: the four Markdown conventions are used correctly; front matter is present; headings are sentence-shaped sections, not fragments; maths is correct and notation is consistent within the page; numbers in prose agree with any code or table nearby; figure references point at files that exist in the diff or are unchanged; claims are accurate; the tone is explanatory, not marketing. Do not comment on style preferences. Be concrete: quote the line, say what is wrong, say what would fix it.

Answer in Markdown for a GitHub comment. First line exactly one of: VERDICT: ready, VERDICT: changes needed. Then a two-sentence summary, then a list of findings ordered by importance, each with the file and the quoted line. If there are no findings say so in one line.`,
  input: `Pull request #${PR}.\n\n${diff}`,
});

const text = response.output_text.trim();
const ready = /^VERDICT: ready/m.test(text);
const body = `${ready ? "🟢" : "🟠"} **Review agent** (${MODEL})\n\n${text}\n\n<sub>Automated first pass. A code owner still reviews and approves.</sub>`;
gh(["pr", "comment", PR, "--repo", REPO, "--body-file", "-"], body);
try {
  gh([
    "pr",
    "edit",
    PR,
    "--repo",
    REPO,
    "--add-label",
    ready ? "agent:ready" : "agent:changes",
  ]);
  gh([
    "pr",
    "edit",
    PR,
    "--repo",
    REPO,
    "--remove-label",
    ready ? "agent:changes" : "agent:ready",
  ]);
} catch {} // labels are cosmetic; the comment is the record
console.log(text.split("\n")[0]);
