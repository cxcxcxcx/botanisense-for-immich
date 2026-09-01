// Regression tests for useIntersectionObserver.
//
// The Grid renders "Loading…" before the first /api/results response
// lands, so the sentinel element is NOT in the DOM on the very first
// commit. The old implementation used a `useRef` and a useEffect whose
// deps were `[options.rootMargin, options.threshold]` — once the
// sentinel finally mounted, the deps didn't change, so the effect
// never re-ran and the observer was never attached. loadMore was never
// called, and the grid stuck at the first 60 photos.
//
// The hook now uses a callback-ref → state pattern. These tests pin
// that contract so a future "simplification" can't reintroduce the
// useRef bug.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useIntersectionObserver } from './useIntersectionObserver';

// Minimal IntersectionObserver stub. happy-dom doesn't implement one,
// so we define our own — and inspect what was observed + manually
// fire the callback to flip isIntersecting from the test.
class MockIO {
  static instances: MockIO[] = [];

  cb: IntersectionObserverCallback;
  observed: Element[] = [];
  disconnected = false;
  unobserveCalls = 0;
  root: Element | null = null;
  rootMargin = '';
  thresholds: ReadonlyArray<number> = [];

  constructor(cb: IntersectionObserverCallback, opts?: IntersectionObserverInit) {
    this.cb = cb;
    if (opts) {
      this.root = (opts.root as Element | null) ?? null;
      this.rootMargin = opts.rootMargin ?? '';
      const t = opts.threshold;
      this.thresholds = Array.isArray(t) ? t : [t ?? 0];
    }
    MockIO.instances.push(this);
  }

  observe(el: Element): void {
    this.observed.push(el);
  }

  unobserve(el: Element): void {
    this.unobserveCalls += 1;
    this.observed = this.observed.filter((x) => x !== el);
  }

  disconnect(): void {
    this.disconnected = true;
  }

  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }

  // Test-only helper: synthesize a callback firing. We always pass the
  // first observed element as `target` so callers that read it (the
  // real IntersectionObserverEntry does) don't crash.
  __fire(isIntersecting: boolean): void {
    const target = this.observed[0] as Element;
    const entry = {
      isIntersecting,
      target,
      intersectionRatio: isIntersecting ? 1 : 0,
    } as unknown as IntersectionObserverEntry;
    this.cb([entry], this as unknown as IntersectionObserver);
  }
}

beforeEach(() => {
  MockIO.instances = [];
  (globalThis as unknown as { IntersectionObserver: typeof IntersectionObserver }).IntersectionObserver =
    MockIO as unknown as typeof IntersectionObserver;
});

afterEach(() => {
  delete (globalThis as unknown as { IntersectionObserver?: typeof IntersectionObserver }).IntersectionObserver;
});

describe('useIntersectionObserver', () => {
  it('attaches an observer when the ref is wired up', () => {
    const { result } = renderHook(() =>
      useIntersectionObserver<HTMLDivElement>({ rootMargin: '300px 0px' }),
    );
    // Before the ref is attached, nothing has been mounted.
    expect(MockIO.instances).toHaveLength(0);

    const sentinel = document.createElement('div');
    act(() => {
      result.current[0](sentinel);
    });

    expect(MockIO.instances).toHaveLength(1);
    const obs = MockIO.instances[0];
    expect(obs).toBeDefined();
    expect(obs!.observed).toContain(sentinel);
    expect(obs!.rootMargin).toBe('300px 0px');
    expect(obs!.disconnected).toBe(false);
  });

  // The regression test for the timing bug. Under the old useRef impl
  // this assertion failed: the hook's effect ran once on renderHook's
  // initial render with ref.current === null, bailed out, and never
  // re-ran (its deps didn't change). MockIO.instances stayed at 0
  // forever. Under the callback-ref impl, setElement triggers a
  // re-render and the effect re-attaches.
  it('attaches an observer when the ref is wired up after the hook first runs', () => {
    const { result } = renderHook(() =>
      useIntersectionObserver<HTMLDivElement>({ rootMargin: '300px 0px' }),
    );
    // The hook has mounted and run its effect once. No element → no
    // observer. This is the state the Grid sits in while it shows
    // "Loading…" before the first /api/results lands.
    expect(MockIO.instances).toHaveLength(0);

    // First page arrives, the Grid renders the sentinel, React fires
    // the callback ref with the new node. The hook must respond by
    // creating an observer.
    const sentinel = document.createElement('div');
    act(() => {
      result.current[0](sentinel);
    });

    expect(MockIO.instances).toHaveLength(1);
    const obs = MockIO.instances[0];
    expect(obs).toBeDefined();
    expect(obs!.observed).toContain(sentinel);
  });

  it('mirrors the observer callback into the isIntersecting state', () => {
    const { result } = renderHook(() =>
      useIntersectionObserver<HTMLDivElement>({ rootMargin: '0px' }),
    );
    act(() => {
      result.current[0](document.createElement('div'));
    });
    expect(result.current[1]).toBe(false);

    const obs = MockIO.instances[0];
    expect(obs).toBeDefined();

    act(() => {
      obs!.__fire(true);
    });
    expect(result.current[1]).toBe(true);

    act(() => {
      obs!.__fire(false);
    });
    expect(result.current[1]).toBe(false);
  });

  it('disconnects the observer when the ref is unset', () => {
    const { result } = renderHook(() =>
      useIntersectionObserver<HTMLDivElement>({ rootMargin: '0px' }),
    );
    act(() => {
      result.current[0](document.createElement('div'));
    });
    expect(MockIO.instances).toHaveLength(1);
    const obs = MockIO.instances[0];
    expect(obs).toBeDefined();
    expect(obs!.disconnected).toBe(false);

    // Grid leaves infinite-scroll mode (filters.page flips > 0 on a
    // deep link, or the sentinel is unmounted entirely).
    act(() => {
      result.current[0](null);
    });

    expect(obs!.disconnected).toBe(true);
    // We don't churn out a fresh observer while nothing is mounted —
    // the next mount should reuse, not multiply.
    expect(MockIO.instances).toHaveLength(1);
  });

  it('reattaches a new observer if a new element is mounted after disconnect', () => {
    const { result } = renderHook(() =>
      useIntersectionObserver<HTMLDivElement>({ rootMargin: '0px' }),
    );
    act(() => {
      result.current[0](document.createElement('div'));
    });
    expect(MockIO.instances).toHaveLength(1);

    act(() => {
      result.current[0](null);
    });
    const first = MockIO.instances[0];
    expect(first).toBeDefined();
    expect(first!.disconnected).toBe(true);

    act(() => {
      result.current[0](document.createElement('div'));
    });
    expect(MockIO.instances).toHaveLength(2);
    const second = MockIO.instances[1];
    expect(second).toBeDefined();
    expect(second!.observed).toHaveLength(1);
    expect(second!.disconnected).toBe(false);
  });
});