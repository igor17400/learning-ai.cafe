// Wrap everything under each <h2> in <section id=... data-section=...>, so the
// contribution chips, the on-page rail and the app can address a section.
// Runs after Astro has assigned heading ids.
export function rehypeSections() {
  return (tree) => {
    const out = [];
    let current = null;
    for (const node of tree.children) {
      if (node.type === "element" && node.tagName === "h2") {
        const id = node.properties?.id || "";
        current = {
          type: "element",
          tagName: "section",
          properties: { id, "data-section": id },
          children: [node],
        };
        node.properties.id = undefined; // the section owns the anchor
        out.push(current);
      } else if (current) {
        current.children.push(node);
      } else {
        out.push(node);
      }
    }
    tree.children = out;
  };
}
