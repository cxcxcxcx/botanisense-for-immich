import { useEffect } from 'react';

// Single-key shortcuts for power users. `b` → Browse, `r` → Run / Sync,
// `?` → open the shortcuts help modal. Multi-key chords (Ctrl+K etc.)
// are deliberately not supported yet — single-key shortcuts cover the
// two main views and help, and that's enough for the launch.
//
// Skipped when focus is in an input, textarea, or contentEditable, so
// typing "red" in the search pill doesn't accidentally flip views.

export interface ShortcutBinding {
  key: string;
  // Match `event.key` exactly (case-insensitive) for letter keys.
  // Special keys: '?', 'Escape'.
  label: string;
  action: () => void;
}

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (el.isContentEditable) return true;
  // Pico CSS dialogs use tabindex="-1" but stay in the focus tree —
  // also skip when an ancestor <dialog> is open (modal trap).
  let p: HTMLElement | null = el;
  while (p) {
    if (p.tagName === 'DIALOG' && (p as HTMLDialogElement).open) return true;
    p = p.parentElement;
  }
  return false;
}

export function useKeyboardShortcuts(bindings: ShortcutBinding[]): void {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      const key = e.key === '?' ? '?' : e.key.toLowerCase();
      const match = bindings.find(b => b.key === key);
      if (!match) return;
      e.preventDefault();
      match.action();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [bindings]);
}