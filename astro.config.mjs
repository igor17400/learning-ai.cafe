// @ts-check
import { defineConfig, passthroughImageService } from "astro/config";
import sitemap from "@astrojs/sitemap";
import remarkDirective from "remark-directive";
import remarkMath from "remark-math";
import rehypeMathjax from "rehype-mathjax/chtml";
import { rehypeHeadingIds } from "@astrojs/markdown-remark";
import { remarkBlocks } from "./src/plugins/remark-blocks.mjs";
import { rehypeSections } from "./src/plugins/rehype-sections.mjs";

export default defineConfig({
  site: "https://learning-ai.cafe",
  output: "static",
  integrations: [sitemap()],
  // figures are SVG and copied as they are; no native image toolchain needed
  image: { service: passthroughImageService() },
  trailingSlash: "always",
  i18n: {
    defaultLocale: "en",
    locales: ["en", "pt-BR"],
    routing: { prefixDefaultLocale: false },
  },
  markdown: {
    // ::: blocks come from remark-directive (the standard syntax); our plugin
    // turns :::figure and :::aside into <figure> and <details>. Maths is
    // rendered at build time, so pages ship no MathJax runtime.
    remarkPlugins: [remarkDirective, remarkBlocks, remarkMath],
    rehypePlugins: [
      rehypeHeadingIds,
      rehypeSections,
      [
        rehypeMathjax,
        {
          chtml: {
            fontURL:
              "https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2",
          },
        },
      ],
    ],
    smartypants: false,
  },
});
