// A copy button on every code block. Used most on a references page, where the
// blocks hold BibTeX, but a reader copying a snippet gets it for free.
import { $$, T } from "./dom.js";

for (const pre of $$("article.page pre")) {
  const btn = document.createElement("button");
  btn.className = "copy";
  btn.type = "button";
  btn.textContent = T.copy;
  btn.setAttribute("aria-label", T.copy);
  btn.addEventListener("click", async () => {
    const text = pre.querySelector("code")?.innerText ?? pre.innerText;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = T.copied;
    } catch {
      // clipboard refused (insecure context, or permission): select it instead
      // so the reader can copy by hand rather than get a button that does nothing
      const range = document.createRange();
      range.selectNodeContents(pre);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      btn.textContent = T.copy_failed;
    }
    setTimeout(() => (btn.textContent = T.copy), 1600);
  });
  pre.appendChild(btn);
}
