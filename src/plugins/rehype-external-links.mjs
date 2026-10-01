// Links that leave the site open in a new tab, so a reader following a source
// does not lose their place in the chapter. Internal links are left alone.
import { visit } from "unist-util-visit";

export function rehypeExternalLinks() {
  return (tree) => {
    visit(tree, "element", (node) => {
      if (node.tagName !== "a") return;
      const href = node.properties?.href;
      if (typeof href !== "string" || !/^https?:\/\//i.test(href)) return;
      node.properties.target = "_blank";
      node.properties.rel = "noopener noreferrer";
    });
  };
}
