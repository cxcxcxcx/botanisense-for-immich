import { createContext, useCallback, useContext, useMemo, useReducer, type ReactNode } from 'react';

export type ToastKind = 'success' | 'error' | 'info';
export interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
  ttlMs: number;
}

interface ToastCtx {
  push: (toast: Omit<Toast, 'id' | 'ttlMs'>, ttlMs?: number) => void;
}

const Ctx = createContext<ToastCtx | null>(null);

function reducer(state: Toast[], action: { type: 'push'; toast: Toast } | { type: 'expire'; id: number }): Toast[] {
  switch (action.type) {
    case 'push': return [...state, action.toast];
    case 'expire': return state.filter(t => t.id !== action.id);
    default: return state;
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
  const value = useMemo(() => ({ push }), [push]);
  return (
    <Ctx.Provider value={value}>
      {children}
      <div style={{ position: 'fixed', bottom: '1.5rem', left: '50%', transform: 'translateX(-50%)', zIndex: 1001, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {items.map(t => (
          <div key={t.id} className={`toast ${t.kind}`}>{t.message}</div>
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