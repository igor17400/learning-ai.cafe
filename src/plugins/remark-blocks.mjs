// The block conventions every page uses, on top of remark-directive:
//
//   :::figure{#name .narrow}      an image line, then the caption
//   :::aside[Title]               a collapsible note
//   :::note[Title]                an open callout with a small title
//   ==text==                      the one highlighter
//
// Everything else is plain Markdown.
import { visit } from "unist-util-visit";

export function remarkBlocks() {
  return (tree) => {
    visit(tree, "containerDirective", (node) => {
      if (node.name === "figure") {
        const id = node.attributes?.id || "";
        const narrow = (node.attributes?.class || "")
          .split(" ")
          .includes("narrow");
        const [imgPara, ...rest] = node.children;
        const caption = {
          type: "paragraph",
          children: rest.flatMap((p) => p.children || []),
          data: { hName: "figcaption" },
        };
        node.data = {
          hName: "figure",
          hProperties: {
            "data-figure": id,
            class: narrow ? "narrow" : undefined,
          },
        };
        node.children = [imgPara, caption];
      }
      if (node.name === "aside" || node.name === "note") {
        const label = node.children.find((c) => c.data?.directiveLabel);
        const body = node.children.filter((c) => c !== label);
        const titleChildren = label
          ? label.children
          : [{ type: "text", value: node.name === "aside" ? "Aside" : "" }];
        if (node.name === "aside") {
          node.data = { hName: "details", hProperties: {} };
          node.children = [
            {
              type: "paragraph",
              children: titleChildren,
              data: { hName: "summary" },
            },
            { type: "parent", children: body, data: { hName: "div" } },
          ];
        } else {
          node.data = { hName: "aside", hProperties: { class: "note" } };
          node.children = [
            {
              type: "paragraph",
              children: titleChildren,
              data: { hName: "p", hProperties: { class: "eyebrow" } },
            },
            ...body,
          ];
        }
      }
    });
    // ==text== may contain inline math or emphasis, so it is matched across
    // a parent's children rather than inside one text node.
    visit(tree, (node) => {
      if (
        !node.children ||
        !node.children.some((c) => c.type === "text" && c.value.includes("=="))
      )
        return;
      const out = [];
      let open = null;
      for (const child of node.children) {
        if (child.type !== "text") {
          (open ? open.children : out).push(child);
          continue;
        }
        child.value.split("==").forEach((p, i) => {
          if (i > 0) {
            if (open) {
              out.push(open);
              open = null;
            } else
              open = { type: "strong", data: { hName: "mark" }, children: [] };
          }
          if (p) (open ? open.children : out).push({ type: "text", value: p });
        });
      }
      if (open) out.push(...open.children); // unmatched marker: keep the text, drop the highlight
      node.children = out;
    });
  };
}
