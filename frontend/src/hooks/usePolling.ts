// Generic poll-on-interval hook. Pauses while document is hidden so a
// backgrounded tab doesn't hammer the server.

import { useCallback, useEffect, useRef, useState } from 'react';

export interface UsePollingResult<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  refresh: () => void;
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  options: { intervalMs: number; enabled?: boolean },
): UsePollingResult<T> {
  const { intervalMs, enabled = true } = options;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const tick = useCallback(async () => {
    setLoading(true);
    try {
      const v = await fetcherRef.current();
      setData(v);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const loop = async () => {
      if (cancelled) return;
      await tick();
      if (cancelled) return;
      timer = setTimeout(loop, intervalMs);
    };
    loop();

    // Pause when tab is hidden — no point hammering /api/status when the
    // user isn't looking.
    const onVis = () => {
      if (document.hidden) {
        if (timer) { clearTimeout(timer); timer = null; }
      } else if (!timer) {
        loop();
      }
    };
    document.addEventListener('visibilitychange', onVis);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [enabled, intervalMs, tick]);

  return { data, error, loading, refresh: tick };
}