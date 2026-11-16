// Toast queue — singleton Context provider mounted at the App root.
//
// Conventions enforced here:
//   - Strings: "Loading…" for data fetches, "Counting…" for previews,
//     "Saving…" for mutations. See the inline `useState`-button sites
//     for the standard.
//   - Queue: FIFO expiry by `ttlMs`; capped at MAX_TOASTS (5) so a
//     burst (e.g. an SSE reconnect replay) doesn't cover the screen.
//     Older ones fall off the bottom.
//   - Dismissal: the timer auto-dismisses; the × button dismisses
//     immediately; an action-button click runs `onClick` then
//     dismisses. Clicking the body itself does NOT dismiss.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useState,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

export type ToastKind = 'success' | 'error' | 'info';
export interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
  ttlMs: number;
  // Optional inline action — e.g. "Retry" on transient errors. The
  // button dismisses the toast when clicked.
  action?: { label: string; onClick: () => void };
}

interface ToastCtx {
  push: (toast: Omit<Toast, 'id' | 'ttlMs'>, ttlMs?: number) => void;
}

const Ctx = createContext<ToastCtx | null>(null);

// Cap at 5 visible to avoid covering the screen on a burst of errors
// (e.g. an SSE reconnect replay). Older ones fall off the bottom.
const MAX_TOASTS = 5;

function reducer(state: Toast[], action: { type: 'push'; toast: Toast } | { type: 'expire'; id: number }): Toast[] {
  switch (action.type) {
    case 'push':
      return [...state, action.toast].slice(-MAX_TOASTS);
    case 'expire':
      return state.filter(t => t.id !== action.id);
    default:
      return state;
  }
}

/** The topmost open native <dialog>, or <body> when no dialog is open. */
function toastHostElement(): HTMLElement {
  const dialogs = document.querySelectorAll<HTMLDialogElement>('dialog[open]');
  const top = dialogs[dialogs.length - 1];
  return top ?? document.body;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, dispatch] = useReducer(reducer, [] as Toast[]);
  const [toastHost, setToastHost] = useState<HTMLElement>(() =>
    typeof document === 'undefined' ? (null as unknown as HTMLElement) : document.body,
  );

  const push = useCallback((t: Omit<Toast, 'id' | 'ttlMs'>, ttlMs = 3000) => {
    const id = Date.now() + Math.random();
    const toast: Toast = { ...t, id, ttlMs };
    dispatch({ type: 'push', toast });
    setTimeout(() => dispatch({ type: 'expire', id }), ttlMs);
  }, []);
  const expire = useCallback((id: number) => dispatch({ type: 'expire', id }), []);
  const value = useMemo(() => ({ push }), [push]);

  // A native <dialog> is in the browser's top layer: a toast that is a
  // sibling of the dialog (even with a huge z-index) is painted BEHIND
  // it. Portal the toast stack into the open dialog instead. When no
  // dialog is open it lives on <body> with z-index 1001.
  useEffect(() => {
    const update = () => setToastHost(toastHostElement());
    update();
    const observer = new MutationObserver(update);
    observer.observe(document.body, {
      subtree: true,
      attributes: true,
      attributeFilter: ['open'],
    });
    return () => observer.disconnect();
  }, []);

  const stack = (
    <div className="toast-stack" role="status" aria-live="polite">
      {items.map(t => (
        <div key={t.id} className={`toast-${t.kind} flex items-center gap-3`}>
          <span className="flex-1">{t.message}</span>
          {t.action && (
            <button
              type="button"
              className="text-[0.78rem] underline hover:no-underline"
              onClick={() => {
                t.action!.onClick();
                expire(t.id);
              }}
            >
              {t.action.label}
            </button>
          )}
          <button
            type="button"
            aria-label="Dismiss"
            className="text-fg-1 hover:text-fg leading-none text-base"
            onClick={() => expire(t.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );

  return (
    <Ctx.Provider value={value}>
      {children}
      {toastHost ? createPortal(stack, toastHost) : null}
    </Ctx.Provider>
  );
}

export function useToasts(): ToastCtx['push'] {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useToasts must be used inside <ToastProvider>');
  return ctx.push;
}
