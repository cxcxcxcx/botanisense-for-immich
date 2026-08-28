import { useEffect, useRef, useState } from 'react';

// Tiny IntersectionObserver wrapper for infinite-scroll sentinel triggers.
// Returns [ref, isIntersecting]; the caller wires the ref onto the
// sentinel element and watches `isIntersecting` to know when to fetch
// more. Sentinel re-attaches automatically if the element unmounts.

export function useIntersectionObserver<T extends Element>(
  options: IntersectionObserverInit = { rootMargin: '200px 0px' },
): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [isIntersecting, setIsIntersecting] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry) setIsIntersecting(entry.isIntersecting);
      },
      options,
    );
    obs.observe(el);
    return () => obs.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options.rootMargin, options.threshold]);

  return [ref, isIntersecting];
}