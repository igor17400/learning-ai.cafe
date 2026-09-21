// What the iPhone app reads: every tutorial, every page, every language, with
// the path of the built HTML. Page bodies come from the same static output.
import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { pathFor } from "../i18n";

export const GET: APIRoute = async () => {
  const tutorials = await getCollection("tutorials");
  const pages = await getCollection("pages");
  const out = tutorials.map((tut) => {
    const slug = tut.id.split("/")[0];
    return {
      id: tut.data.id,
      slug,
      title: tut.data.title,
      tagline: tut.data.tagline,
      track: tut.data.track,
      hue: tut.data.hue,
      languages: tut.data.languages,
      requires: tut.data.requires,
      related: tut.data.related,
      pages: tut.data.pages.map((p) => ({
        slug: p.slug,
        title: p.title,
        subtitle: p.subtitle,
        status: p.status,
        paths: Object.fromEntries(
          tut.data.languages
            .filter((l) =>
              pages.some(
                (e) =>
                  e.id.toLowerCase() ===
                  `${slug}/pages/${l.toLowerCase()}/${p.slug}`,
              ),
            )
            .map((l) => [l, pathFor(l, `${slug}/${p.slug}/`)]),
        ),
      })),
    };
  });
  return new Response(
    JSON.stringify(
      { schema: 1, generated: new Date().toISOString(), tutorials: out },
      null,
      2,
    ),
    { headers: { "Content-Type": "application/json" } },
  );
};
