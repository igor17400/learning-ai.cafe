// Contribution chips next to headings, the figure info buttons, the credits
// line, and the two contribution modals. Data comes from #page-meta.
import { $, $$, T } from "./dom.js";
const metaEl = $("#page-meta");
if (metaEl) {
  const meta = JSON.parse(metaEl.textContent);
  const people = Object.fromEntries(
    meta.contributors.map((c) => [c.handle, c]),
  );
  const initials = (h) => {
    const s = h
      .replace(/^@/, "")
      .replace(/[^a-z0-9]/gi, " ")
      .trim()
      .split(/\s+/);
    return (s.length > 1 ? s[0][0] + s[1][0] : s[0].slice(0, 2)).toUpperCase();
  };
  const prLink = (n) => (n ? `<a href="${meta.repo}/pull/${n}">#${n}</a>` : "");
  const esc = (s) =>
    String(s ?? "").replace(
      /[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
    );
  const chip = (handle, small) => {
    const c = people[handle] || { handle, hue: 200 };
    const b = document.createElement("button");
    b.className = "chip" + (small ? " small" : "");
    b.title = handle + (c.name ? `, ${c.name}` : "");
    b.textContent = initials(handle);
    b.dataset.person = handle;
    return b;
  };
  const detail = $("[data-detail]"),
    detailDlg = $('dialog[data-modal="detail"]');
  const showDetail = (html) => {
    detail.innerHTML = html;
    detailDlg.showModal();
  };
  const showPerson = (handle, section) => {
    const c = people[handle];
    if (!c) return;
    const entries = section
      ? c.entries.filter((e) => (e.sections || []).includes(section))
      : c.entries;
    const title = section
      ? `${T.contributions_section} · ${esc($(`#${CSS.escape(section)} > h2`)?.firstChild?.textContent?.trim() || section)}`
      : T.contributions_of;
    showDetail(`<div class="detail"><h3>${chip(handle).outerHTML} ${esc(handle)}</h3>
      <p class="sub">${esc(c.name || "")} · ${c.score} pts · <a href="https://github.com/${esc(c.github || handle.slice(1))}">GitHub</a></p>
      <h4>${title}</h4>
      <ol class="history">${entries.map((e) => `<li><time>${e.date}</time><span>${esc(e.what)} <span class="muted">${esc(e.type)}, ${e.score} pts</span>${prLink(e.pr)}</span></li>`).join("")}</ol></div>`);
  };
  const showFigure = (name) => {
    const f = meta.figures[name];
    if (!f) return;
    showDetail(`<div class="detail"><h3><code>${esc(name)}</code></h3><ol class="history">
      ${f.by ? `<li><time></time><span>${esc(T.figureby)} <b>${esc(f.by)}</b>${prLink(f.pr)}</span></li>` : ""}
      ${f.reviewed ? `<li><time></time><span>${esc(T.reviewedby)} <b>${esc(f.reviewed)}</b>${prLink(f.reviewed_pr)}</span></li>` : ""}
      </ol></div>`);
  };
  document.addEventListener("click", (e) => {
    const p = e.target.closest("[data-person]");
    if (p) {
      e.preventDefault();
      showPerson(p.dataset.person, p.dataset.section);
    }
    const f = e.target.closest("[data-figure-info]");
    if (f) {
      e.preventDefault();
      showFigure(f.dataset.figureInfo);
    }
  });
  $$("section[data-section]").forEach((sec) => {
    const who = meta.sections[sec.dataset.section] || [];
    if (!who.length) return;
    const box = document.createElement("span");
    box.className = "marks";
    box.title = `${T.section_by} ${who.join(", ")}`;
    who.forEach((h) => {
      const c = chip(h, true);
      c.dataset.section = sec.dataset.section;
      box.appendChild(c);
    });
    ($("h2", sec) || sec).appendChild(box);
  });
  $$("figure[data-figure]").forEach((fig) => {
    const f = meta.figures[fig.dataset.figure],
      cap = $("figcaption", fig);
    if (!f || !cap) return;
    const i = document.createElement("button");
    i.className = "info";
    i.textContent = "i";
    i.dataset.figureInfo = fig.dataset.figure;
    i.title = `${T.figureby} ${f.by || ""}`;
    i.setAttribute("aria-label", i.title);
    cap.appendChild(i);
  });
  const by = $("[data-byline]");
  if (by && meta.contributors.length) {
    const chips = document.createElement("span");
    chips.className = "chips";
    meta.contributors.forEach((c) => chips.appendChild(chip(c.handle)));
    by.appendChild(chips);
  }
  const cr = $("[data-credits]");
  if (cr && meta.contributors.length) {
    const chips = document.createElement("span");
    chips.className = "chips";
    meta.contributors.forEach((c) => chips.appendChild(chip(c.handle)));
    cr.appendChild(document.createTextNode(T.builtby + " "));
    cr.appendChild(chips);
    cr.appendChild(
      document.createTextNode(` · ${meta.history.length} ${T.revisions} · `),
    );
    const b = document.createElement("button");
    b.className = "link";
    b.dataset.open = "contrib";
    b.textContent = T.who;
    cr.appendChild(b);
  }
  const rows = $("[data-contrib-rows]");
  if (rows) {
    const max = Math.max(...meta.contributors.map((c) => c.score));
    meta.contributors.forEach((c) => {
      const row = document.createElement("div");
      row.className = "contrib-row";
      row.appendChild(chip(c.handle));
      const name = document.createElement("div");
      name.className = "name";
      const nb = document.createElement("button");
      nb.dataset.person = c.handle;
      nb.textContent = c.handle;
      name.appendChild(nb);
      row.appendChild(name);
      const types = document.createElement("div");
      types.className = "types";
      Object.entries(c.types).forEach(([k, v]) => {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = `${v} ${k}`;
        types.appendChild(t);
      });
      const bar = document.createElement("div");
      bar.className = "bar";
      const i = document.createElement("i");
      i.style.width = `${Math.round((c.score / max) * 100)}%`;
      bar.appendChild(i);
      types.appendChild(bar);
      row.appendChild(types);
      const sc = document.createElement("div");
      sc.className = "score";
      sc.textContent = c.score;
      row.appendChild(sc);
      rows.appendChild(row);
    });
  }
  const hist = $("[data-history]");
  if (hist)
    meta.history.forEach((h) => {
      const li = document.createElement("li");
      li.innerHTML = `<time>${h.date}</time><span>${esc(h.what)} <span class="muted">${esc(h.by)}, ${esc(h.type)}, ${h.score} pts</span>${prLink(h.pr)}</span>`;
      hist.appendChild(li);
    });
}
