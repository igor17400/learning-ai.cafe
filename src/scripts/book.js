// Book mode: two pages per screen, arrows, taps, page numbers, progress bar.
import { root, store, $, $$ } from "./dom.js";
const meta = JSON.parse($("#page-meta")?.textContent || "{}");
const chapters = meta.chapters || [];
const chapter = meta.chapter || 0;
const article = $("article.page");
const book = $(".bookwrap");
const count = $("[data-pagecount]");
const bar = $(".progress");
let spreads = 1,
  spread = 0;

function stride() {
  // columns advance by column width plus gap; two columns per spread, so the
  // next spread starts one content-width plus one gap to the right
  const cs = getComputedStyle(article);
  const inner =
    article.clientWidth -
    parseFloat(cs.paddingLeft) -
    parseFloat(cs.paddingRight);
  const gap = parseFloat(cs.columnGap);
  return inner + (Number.isFinite(gap) ? gap : 0);
}
function perSpread() {
  return parseInt(getComputedStyle(article).columnCount || "1", 10) || 1;
}
// pages before this chapter and in the whole book, estimated from word counts
// calibrated on this chapter's real page count, so numbering runs across chapters
let offset = 0,
  total = 0;
function measure() {
  const s = stride();
  spreads =
    s > 0
      ? Math.max(
          1,
          Math.ceil((article.scrollWidth - article.clientWidth + s) / s),
        )
      : 1;
  const local = spreads * perSpread();
  const wpp = Math.max(1, (chapters[chapter]?.words || 1) / local);
  const est = (c, j) => (j === chapter ? local : c.words ? Math.max(1, Math.round(c.words / wpp)) : 0);
  offset = chapters.slice(0, chapter).reduce((n, c, j) => n + est(c, j), 0);
  total = chapters.reduce((n, c, j) => n + est(c, j), 0) || local;
}
function closeBook() {
  root.dataset.book = "closed";
  if (count) count.textContent = "";
  if (bar) bar.style.width = "0";
}
function openBook() {
  delete root.dataset.book;
  measure();
  show(0);
}
// past the last spread the next chapter opens; before the first, the previous
// chapter opens on its last spread
function show(i) {
  if (!Number.isFinite(i)) i = 0;
  if (i >= spreads && spreads > 0) {
    const n = $("a[rel=next]");
    if (n) return (location.href = n.href);
  }
  if (i < 0) {
    const p = $("a[rel=prev]");
    if (p) {
      store.set("page." + new URL(p.href).pathname, "end");
      return (location.href = p.href);
    }
    if ($("[data-open-book]")) return closeBook();
  }
  spread = Math.min(Math.max(0, i), spreads - 1);
  article.scrollTo({ left: spread * stride(), behavior: "auto" });
  const n = perSpread();
  const first = offset + spread * n + 1;
  const last = Math.min(first + n - 1, offset + spreads * n);
  if (book) {
    book.dataset.lp = String(first);
    book.dataset.rp = n > 1 && last > first ? String(last) : "";
  }
  if (count) count.textContent = n > 1 ? `${first}–${last} / ${total}` : `${first} / ${total}`;
  if (bar) bar.style.width = ((last / total) * 100).toFixed(1) + "%";
  store.set("page." + location.pathname, String(spread));
}
function progressScroll() {
  if (!bar || root.dataset.reader === "on") return;
  bar.style.width =
    (
      (100 * root.scrollTop) /
      Math.max(1, root.scrollHeight - root.clientHeight)
    ).toFixed(1) + "%";
}
// Chrome will not split a <details> across columns, so in book mode each aside
// is swapped for a plain block holding a copy of its content, and swapped back after.
function bookAsides(on) {
  if (on)
    $$("details").forEach((d) => {
      const w = document.createElement("div");
      w.className = "aside-open";
      const sum = $("summary", d),
        title = document.createElement("p");
      title.className = "title";
      title.innerHTML = sum ? sum.innerHTML : "";
      w.appendChild(title);
      Array.from(d.children)
        .filter((c) => c !== sum)
        .forEach((c) => w.appendChild(c.cloneNode(true)));
      w._details = d;
      d.replaceWith(w);
    });
  else $$(".aside-open").forEach((w) => w.replaceWith(w._details));
}
export function setReader(on) {
  if (!article) return;
  root.dataset.reader = on ? "on" : "off";
  store.set("reader", root.dataset.reader);
  bookAsides(on);
  if (on) {
    const raw = store.get("page." + location.pathname);
    const saved = raw === "end" ? Infinity : Number(raw) || 0;
    // a first visit to chapter one starts on the closed book
    if (raw === null && !location.hash && $("[data-open-book]")) closeBook();
    requestAnimationFrame(() => {
      measure();
      if (root.dataset.book !== "closed")
        show(location.hash ? spreadOf($(decodeURIComponent(location.hash))) : saved);
    });
    const settle = () => {
      if (root.dataset.reader !== "on" || root.dataset.book === "closed") return;
      measure();
      show(spread);
    };
    setTimeout(settle, 600); // images and fonts settle
    document.fonts?.ready.then(settle);
  } else {
    delete root.dataset.book;
    article.scrollTo({ left: 0 });
    if (count) count.textContent = "";
    progressScroll();
  }
}
// which spread holds an element: columns advance left by one stride per spread
function spreadOf(el) {
  if (!el) return spread;
  const s = stride();
  return s > 0 ? Math.floor((el.getBoundingClientRect().left - article.getBoundingClientRect().left + article.scrollLeft) / s) : 0;
}
if (article) {
  if (root.dataset.reader === "on") setReader(true);
  // in-page anchors (contents modal, links in the text) turn to the right spread
  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a || root.dataset.reader !== "on") return;
    const el = $(decodeURIComponent(a.getAttribute("href")));
    if (!el) return;
    e.preventDefault();
    history.replaceState(null, "", a.getAttribute("href"));
    show(spreadOf(el));
  });
  // swipe on touch screens
  let x0 = null;
  article.addEventListener("touchstart", (e) => (x0 = e.touches[0].clientX), { passive: true });
  article.addEventListener("touchend", (e) => {
    if (x0 === null || root.dataset.reader !== "on") return;
    const dx = e.changedTouches[0].clientX - x0;
    x0 = null;
    if (Math.abs(dx) > 40) show(spread + (dx < 0 ? 1 : -1));
  }, { passive: true });
  document.addEventListener("click", (e) => {
    if (root.dataset.reader !== "on") return;
    if (root.dataset.book === "closed") {
      if (e.target.closest("[data-open-book], .tapzone.right")) openBook();
      return;
    }
    if (e.target.closest(".tapzone.left")) show(spread - 1);
    if (e.target.closest(".tapzone.right")) show(spread + 1);
  });
  document.addEventListener("keydown", (e) => {
    if (root.dataset.reader !== "on" || $("dialog[open]")) return;
    if (root.dataset.book === "closed") {
      if (["ArrowRight", " ", "PageDown", "Enter"].includes(e.key)) {
        e.preventDefault();
        openBook();
      }
      if (e.key === "Escape") setReader(false);
      return;
    }
    if (["ArrowRight", " ", "PageDown"].includes(e.key)) {
      e.preventDefault();
      show(spread + 1);
    }
    if (["ArrowLeft", "PageUp"].includes(e.key)) {
      e.preventDefault();
      show(spread - 1);
    }
    if (e.key === "Home") show(0);
    if (e.key === "End") show(spreads - 1);
    if (e.key === "Escape") setReader(false);
  });
  article.addEventListener(
    "wheel",
    (e) => {
      if (root.dataset.reader !== "on") return;
      e.preventDefault();
      const d = Math.abs(e.deltaY) >= Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
      if (Math.abs(d) > 8) show(spread + (d > 0 ? 1 : -1));
    },
    { passive: false },
  );
  window.addEventListener("resize", () => {
    if (root.dataset.reader === "on" && root.dataset.book !== "closed") {
      measure();
      show(spread);
    } else progressScroll();
  });
  window.addEventListener("scroll", progressScroll, { passive: true });
  progressScroll();
}
