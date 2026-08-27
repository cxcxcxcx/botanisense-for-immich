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
//     dismisses. Clicking the body itself does NOT dismiss — toasts
//     are notifications, not popovers.

import { createContext, useCallback, useContext, useMemo, useReducer, type ReactNode } from 'react';

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

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, dispatch] = useReducer(reducer, [] as Toast[]);
  const push = useCallback((t: Omit<Toast, 'id' | 'ttlMs'>, ttlMs = 3000) => {
    const id = Date.now() + Math.random();
    const toast: Toast = { ...t, id, ttlMs };
    dispatch({ type: 'push', toast });
    setTimeout(() => dispatch({ type: 'expire', id }), ttlMs);
  }, []);
  const expire = useCallback((id: number) => dispatch({ type: 'expire', id }), []);
  const value = useMemo(() => ({ push }), [push]);
  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[1001] flex flex-col gap-2">
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
    </Ctx.Provider>
  );
}

export function useToasts(): ToastCtx['push'] {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useToasts must be used inside <ToastProvider>');
  return ctx.push;
}