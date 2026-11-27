// Regression tests for usePersistedState.
//
// The Level-2 cluster sort used to live in a bare useState, so every
// cluster you opened started at 'score' no matter how often you had
// picked another ordering. It is now localStorage-backed: the FIRST
// render must already carry the stored value (no flash of the default),
// and a corrupt / unknown stored value must fall back instead of
// poisoning the view.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePersistedState } from './usePersistedState';

type Sort = 'score' | 'time' | 'area';
const isSort = (v: unknown): v is Sort => v === 'score' || v === 'time' || v === 'area';

// happy-dom under vitest 4 does not expose window.localStorage (node's
// own --localstorage-file gated one is undefined), so install a minimal
// Storage instead of skipping the persistence assertions.
function memoryStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() { return m.size; },
    key: (i: number) => [...m.keys()][i] ?? null,
    getItem: (k: string) => (m.has(k) ? m.get(k)! : null),
    setItem: (k: string, v: string) => { m.set(k, String(v)); },
    removeItem: (k: string) => { m.delete(k); },
    clear: () => { m.clear(); },
  } as Storage;
}

describe('usePersistedState', () => {
  beforeEach(() => { vi.stubGlobal('localStorage', memoryStorage()); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('yields the fallback when nothing is stored', () => {
    const { result } = renderHook(() => usePersistedState<Sort>('k', 'score', isSort));
    expect(result.current[0]).toBe('score');
  });

  it('reads a stored value on the very first render', () => {
    localStorage.setItem('k', JSON.stringify('area'));
    const { result } = renderHook(() => usePersistedState<Sort>('k', 'score', isSort));
    expect(result.current[0]).toBe('area');
  });

  it('writes through on set', () => {
    const { result } = renderHook(() => usePersistedState<Sort>('k', 'score', isSort));
    act(() => result.current[1]('time'));
    expect(result.current[0]).toBe('time');
    expect(JSON.parse(localStorage.getItem('k')!)).toBe('time');
  });

  it('falls back on a corrupt or invalid stored value', () => {
    localStorage.setItem('k', '{not json');
    expect(renderHook(() => usePersistedState<Sort>('k', 'score', isSort)).result.current[0])
      .toBe('score');

    localStorage.setItem('k', JSON.stringify('bogus'));
    expect(renderHook(() => usePersistedState<Sort>('k', 'score', isSort)).result.current[0])
      .toBe('score');
  });

  it('keeps two keys independent (one per view)', () => {
    const a = renderHook(() => usePersistedState<Sort>('a', 'score', isSort));
    const b = renderHook(() => usePersistedState<Sort>('b', 'score', isSort));
    act(() => a.result.current[1]('time'));
    expect(a.result.current[0]).toBe('time');
    expect(b.result.current[0]).toBe('score');
  });
});
