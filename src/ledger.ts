// The ledger is appended by CI from merged pull requests. This module turns it
// into what one page needs: contributors with typed counts, who touched which
// section, who drew and reviewed which figure, and the history.
import live from "../ledger.json";
import samples from "../samples/ledger.json";

const raw = process.env.SAMPLES ? samples : live;

type Contribution = {
  page: string; date: string; by: string; type: string; score: number; what: string; pr: number;
  sections?: string[]; figures?: string[]; figures_reviewed?: string[];
};

export const REPO = "https://github.com/igor17400/learning-ai.cafe";

export function pageMeta(pageId: string) {
  const cs = (raw.contributions as Contribution[]).filter((c) => c.page === pageId);
  const users = raw.users as Record<string, { name: string; hue: number; github: string }>;
  const people: Record<string, any> = {};
  for (const c of cs) {
    const p = (people[c.by] ??= { handle: c.by, ...(users[c.by] || { hue: 200 }), score: 0, types: {}, entries: [] });
    p.score += c.score;
    p.types[c.type] = (p.types[c.type] || 0) + 1;
    p.entries.push({ date: c.date, type: c.type, score: c.score, what: c.what, pr: c.pr, sections: c.sections || [] });
  }
  const sections: Record<string, string[]> = {};
  const figures: Record<string, any> = {};
  for (const c of cs) {
    for (const s of c.sections || []) (sections[s] ??= []).includes(c.by) || sections[s].push(c.by);
    for (const f of c.figures || []) Object.assign((figures[f] ??= {}), { by: c.by, pr: c.pr });
    for (const f of c.figures_reviewed || []) Object.assign((figures[f] ??= {}), { reviewed: c.by, reviewed_pr: c.pr });
  }
  return {
    page: pageId, repo: REPO,
    contributors: Object.values(people).sort((a, b) => b.score - a.score),
    sections, figures,
    history: cs.map(({ date, by, type, score, what, pr }) => ({ date, by, type, score, what, pr })).sort((a, b) => (a.date < b.date ? 1 : -1)),
  };
}

// everyone who touched any page under the prefix, for the book's credits page
export function tutorialContributors(prefix: string) {
  const users = raw.users as Record<string, { name: string; hue: number; github: string }>;
  const people: Record<string, { handle: string; name: string; hue: number; score: number; types: Record<string, number> }> = {};
  for (const c of (raw.contributions as Contribution[]).filter((c) => c.page.startsWith(prefix))) {
    const p = (people[c.by] ??= { handle: c.by, name: users[c.by]?.name || "", hue: users[c.by]?.hue ?? 200, score: 0, types: {} });
    p.score += c.score;
    p.types[c.type] = (p.types[c.type] || 0) + 1;
  }
  return Object.values(people).sort((a, b) => b.score - a.score);
}

export function contributorCount(prefix: string) {
  return new Set((raw.contributions as Contribution[]).filter((c) => c.page.startsWith(prefix)).map((c) => c.by)).size;
}
