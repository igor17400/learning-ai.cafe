// A references page is a list of links, each followed by its BibTeX block. The
// block is lifted out of the flow into a dialog, and a "cite" button is added to
// the line above it, so the page stays a list and the citation is one click away.
import { $$, T } from "./dom.js";

for (const pre of $$('article.page pre[data-language="bibtex"]')) {
  const anchor = pre.previousElementSibling;
  if (!anchor) continue;
  const key =
    /@\w+\{([^,]+),/.exec(pre.textContent)?.[1] ||
    String(Math.random()).slice(2);
  const name = `cite-${key}`;

  const dialog = document.createElement("dialog");
  dialog.className = "modal";
  dialog.dataset.modal = name;
  dialog.innerHTML = `<button class="close" data-close aria-label="${T.close}">&times;</button><h3>BibTeX</h3>`;
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close();
  });
  dialog.appendChild(pre);
  document.body.appendChild(dialog);

  const btn = document.createElement("button");
  btn.className = "cite";
  btn.type = "button";
  btn.dataset.open = name;
  btn.textContent = T.cite;
  anchor.appendChild(document.createTextNode(" "));
  anchor.appendChild(btn);
}
