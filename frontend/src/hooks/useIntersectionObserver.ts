import { useCallback, useEffect, useState } from 'react';

// Tiny IntersectionObserver wrapper for infinite-scroll sentinel triggers.
// Returns [ref, isIntersecting]; the caller wires the ref onto the
// sentinel element and watches `isIntersecting` to know when to fetch
// more. Sentinel re-attaches automatically if the element unmounts.
//
// Implementation note: we use a callback-ref → state, NOT useRef, so the
// observer effect re-runs when the sentinel element first appears in
// the DOM. The Grid renders "Loading…" before the first page lands, so
// on the very first commit the sentinel isn't in the tree — a plain
// useRef would point at null, the effect would bail out, and the
// observer would never attach (the useEffect deps don't change once the
// sentinel eventually mounts, so a one-shot useRef mount effect misses
// it entirely). The callback-ref pattern flips element state when the
// node mounts, which triggers a re-render and re-runs the effect with
// a non-null element.

export function useIntersectionObserver<T extends Element>(
  options: IntersectionObserverInit = { rootMargin: '200px 0px' },
): [React.RefCallback<T>, boolean] {
  const [element, setElement] = useState<T | null>(null);
  const [isIntersecting, setIsIntersecting] = useState(false);

  // Stable callback ref so React doesn't re-run the ref-attaching
  // effects on every render. setElement flips state when the node
  // mounts or unmounts, which is what re-runs the observer effect.
  const ref = useCallback((el: T | null) => {
    setElement(el);
  }, []);

  useEffect(() => {
    if (!element) return;
    const obs = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry) setIsIntersecting(entry.isIntersecting);
      },
      options,
    );
    obs.observe(element);
    return () => obs.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [element, options.rootMargin, options.threshold]);

  return [ref, isIntersecting];
}