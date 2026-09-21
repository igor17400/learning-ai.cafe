// Typed content. A tutorial that violates this schema fails the build.
// SAMPLES=1 (pnpm dev:samples) reads samples/ instead of tutorials/.
import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

export const base = process.env.SAMPLES ? "./samples" : "./tutorials";

const tutorials = defineCollection({
  loader: glob({ pattern: "*/tutorial.yaml", base }),
  schema: z.object({
    schema: z.literal(1),
    id: z.string().regex(/^[a-z0-9-]+\/[a-z0-9_]+$/, "id is <namespace>/<slug>"),
    title: z.string(),
    track: z.string().default(""),
    tagline: z.string().default(""),
    hue: z.number().min(0).max(360).default(28),
    cover: z.enum(["lion", "unicorn", "dragon", "griffin", "phoenix", "stag", "eagle"]).default("lion"), // src/covers/*.svg
    languages: z.array(z.string()).default(["en"]),
    requires: z.array(z.string()).default([]),
    related: z.array(z.string()).default([]),
    pages: z.array(z.object({
      slug: z.string().regex(/^[a-z0-9_]+$/),
      title: z.string(),
      subtitle: z.string().default(""),
      status: z.enum(["written", "draft", "planned"]).default("written"),
    })),
  }),
});

const pages = defineCollection({
  // keep the dot in "<slug>.pt-BR": the default id would drop it
  loader: glob({ pattern: "*/pages/*.md", base, generateId: ({ entry }) => entry.replace(/\.md$/, "") }),
  schema: z.object({
    title: z.string().optional(),
    subtitle: z.string().optional(),
    verified: z.string().optional(),
  }),
});

export const collections = { tutorials, pages };
