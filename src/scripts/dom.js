// Shared by the client scripts: the root element, the locale strings, the
// preference store, and two query helpers.
import { strings } from "../i18n";

export const root = document.documentElement;
export const locale = root.lang || "en";
export const T = strings[locale] || strings.en;
export const store = {
  get: (k) => {
    try {
      return localStorage.getItem("cafe." + k);
    } catch {
      return null;
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem("cafe." + k, v);
    } catch {}
  },
  del: (k) => {
    try {
      localStorage.removeItem("cafe." + k);
    } catch {}
  },
};
export const $ = (s, el = document) => el.querySelector(s);
export const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
