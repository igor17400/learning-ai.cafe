// Reader preferences: mode, hue, font, size, width, language, book, focus.
// Stored per browser, applied to <html> as data attributes and custom properties.
import { root, locale, store, $, $$ } from "./dom.js";
import { setReader } from "./book.js";
function apply() {
  const mode =
    store.get("mode") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  root.dataset.mode = mode;
  root.dataset.font = store.get("font") || "sans";
  root.style.setProperty("--h", store.get("hue") || root.dataset.hue || "28");
  if (store.get("size"))
    root.style.setProperty("--fs", store.get("size") + "rem");
  else root.style.removeProperty("--fs");
  if (store.get("measure"))
    root.style.setProperty("--measure", store.get("measure") + "rem");
  else root.style.removeProperty("--measure");
  root.dataset.focus = store.get("focus") || "off";
  const press = (sel, val) =>
    $$(sel).forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        String(b.dataset[Object.keys(b.dataset)[0]] === val),
      ),
    );
  press("[data-set-mode]", mode);
  press("[data-set-font]", root.dataset.font);
  press("[data-set-reader]", root.dataset.reader || "off");
  $("[data-toggle-focus]")?.setAttribute(
    "aria-pressed",
    String(root.dataset.focus === "on"),
  );
  $("[data-toggle-reader]")?.setAttribute(
    "aria-pressed",
    String(root.dataset.reader === "on"),
  );
  const set = (sel, v) => {
    const el = $(sel);
    if (el && v != null) el.value = v;
  };
  set("[data-hue-range]", store.get("hue") || root.dataset.hue);
  set("[data-size-range]", store.get("size") || "1.0625");
  set("[data-measure-range]", store.get("measure") || "42");
}
document.addEventListener("click", (e) => {
  const b = e.target.closest(
    "[data-set-mode],[data-set-font],[data-set-reader],[data-set-lang],[data-reset-theme],[data-toggle-focus],[data-toggle-reader]",
  );
  if (!b) return;
  if (b.dataset.setMode) store.set("mode", b.dataset.setMode);
  if (b.dataset.setFont) store.set("font", b.dataset.setFont);
  if (b.dataset.setReader) setReader(b.dataset.setReader === "on");
  if (b.hasAttribute("data-toggle-reader"))
    setReader(root.dataset.reader !== "on");
  if (b.hasAttribute("data-toggle-focus"))
    store.set("focus", root.dataset.focus === "on" ? "off" : "on");
  if (b.hasAttribute("data-reset-theme")) {
    ["mode", "hue", "font", "size", "measure", "focus", "reader"].forEach(
      store.del,
    );
    setReader(false);
  }
  if (b.dataset.setLang) {
    store.set("lang", b.dataset.setLang);
    const alt = $(`link[rel="alternate"][hreflang="${b.dataset.setLang}"]`);
    if (alt && b.dataset.setLang !== locale) {
      location.href = new URL(alt.href).pathname; // same origin, hreflang is absolute
      return;
    }
  }
  apply();
});
document.addEventListener("input", (e) => {
  const t = e.target;
  if (t.matches("[data-hue-range]")) store.set("hue", t.value);
  if (t.matches("[data-size-range]")) store.set("size", t.value);
  if (t.matches("[data-measure-range]")) store.set("measure", t.value);
  apply();
});
(() => {
  const L = store.get("lang");
  if (L && L !== locale) {
    const alt = $(`link[rel="alternate"][hreflang="${L}"]`);
    if (alt) location.replace(new URL(alt.href).pathname);
  }
})();
apply();
