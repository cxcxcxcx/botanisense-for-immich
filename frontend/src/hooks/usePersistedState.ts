// usePersistedState — useState mirrored into localStorage.
//
// For VIEW preferences that are per-browser rather than part of the
// shareable URL contract (e.g. the Level-2 cluster sort: you don't want
// `?csort=area` in a link you send someone, but you also don't want to
// re-pick "Largest crop" every time you open a cluster).
//
// The initial read is lazy, so the very first render already has the
// stored value — no flash of the default order before settling.
// Writes are best-effort: quota errors / privacy-mode localStorage must
// not break the view.

import { useCallback, useState } from 'react';

export function readStored<T>(key: string, fallback: T, isValid: (v: unknown) => v is T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null) return fallback;
    const parsed: unknown = JSON.parse(raw);
    return isValid(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

export function usePersistedState<T>(
  key: string,
  fallback: T,
  isValid: (v: unknown) => v is T,
): [T, (next: T) => void] {
  const [value, setValue] = useState<T>(() => readStored(key, fallback, isValid));

  const set = useCallback((next: T) => {
    setValue(next);
    try {
      localStorage.setItem(key, JSON.stringify(next));
    } catch { /* quota / privacy mode — ignore */ }
  }, [key]);

  return [value, set];
}
