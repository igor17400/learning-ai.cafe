// <dialog> plumbing: [data-open=name] opens dialog[data-modal=name],
// [data-close] or a click on the backdrop closes it.
import { $, $$ } from "./dom.js";
$$("dialog.modal").forEach((d) =>
  d.addEventListener("click", (e) => {
    if (e.target === d) d.close();
  }),
);
document.addEventListener("click", (e) => {
  const open = e.target.closest("[data-open]");
  if (open) $(`dialog[data-modal="${open.dataset.open}"]`)?.showModal();
  if (e.target.closest("[data-close]")) e.target.closest("dialog")?.close();
});
