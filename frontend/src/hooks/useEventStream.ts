// Subscribe to /api/events. EventSource auto-reconnects on network
// blips and tab refocus, so we don't need our own retry loop — just
// hand back the latest parsed message.
//
// Replaces the 5 separate polling loops (usePolling + 4 ad-hoc
// setInterval sites). One connection per app; components read slices
// of the same `data` via the hook.

import { useEffect, useState } from 'react';

export type EventStreamReadyState =
  | 'connecting'
  | 'open'
  | 'closed';

export interface UseEventStreamResult<T> {
  /** Latest parsed snapshot. `null` until the first event arrives
   *  (typically < 200ms after mount). */
  data: T | null;
  /** Set if a payload failed to JSON-parse or the stream errored. The
   *  EventSource auto-reconnects; this is informational. */
  error: Error | null;
  /** Mirrors EventSource.readyState. `closed` is brief — the browser
   *  reopens the connection within ~1s. */
  readyState: EventStreamReadyState;
}

const READY_STATE: Record<number, EventStreamReadyState> = {
  0: 'connecting',
  1: 'open',
  2: 'closed',
};

export function useEventStream<T>(url: string): UseEventStreamResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [readyState, setReadyState] = useState<EventStreamReadyState>(
    'connecting',
  );

  useEffect(() => {
    if (typeof EventSource === 'undefined') {
      // SSE isn't available (e.g. SSR, ancient browser) — surface a
      // permanent error rather than silently swallowing updates.
      setError(new Error('EventSource is not supported in this environment'));
      setReadyState('closed');
      return;
    }
    const es = new EventSource(url);

    const onMessage = (ev: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(ev.data) as T;
        setData(parsed);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e : new Error(String(e)));
      }
    };
    const onError = () => {
      // EventSource auto-reconnects. We can't tell apart a transient
      // network blip from a fatal close, so just track readyState and
      // let the next onopen/onmessage clear the error.
      setReadyState(READY_STATE[es.readyState] ?? 'closed');
    };

    es.addEventListener('open', () => setReadyState('open'));
    es.addEventListener('message', onMessage);
    es.addEventListener('error', onError);

    return () => {
      es.removeEventListener('open', () => setReadyState('open'));
      es.removeEventListener('message', onMessage);
      es.removeEventListener('error', onError);
      es.close();
      setReadyState('closed');
    };
  }, [url]);

  return { data, error, readyState };
}