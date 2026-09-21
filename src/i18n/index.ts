// i18n = internationalization (18 letters between the i and the n): the site in
// more than one language. Every piece of interface text lives in this folder,
// one file per locale; templates call t(locale, key). A key missing from a
// locale falls back to English. Tutorial prose is not here: it lives in
// pages/<slug>.<locale>.md.
//
// To add a language: copy en.ts to <code>.ts and translate it, add the code to
// `locales` below and to astro.config.mjs, and add a button to the language
// setting in Base.astro.
import en from "./en";
import ptBR from "./pt-BR";

export const locales = ["en", "pt-BR"] as const;
export type Locale = (typeof locales)[number];

export const strings = { en, "pt-BR": ptBR } as const;

export function t(locale: string, key: keyof typeof en): string {
  const d = (strings as any)[locale] || strings.en;
  return d[key] || strings.en[key] || key;
}

export function pathFor(locale: string, path: string) {
  return locale === "en" ? `/${path}` : `/${locale}/${path}`;
}
