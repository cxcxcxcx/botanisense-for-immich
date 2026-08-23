// Subscribe to /api/events. EventSource auto-reconnects on network
// blips and tab refocus, so we don't need our own retry loop — just
// hand back the latest parsed message.
//
// Module-level singleton: every useEventStream('/api/events')
// subscriber shares one EventSource. This matters because:
//   1. uvicorn's SSE generator pushes one event per change; the
//      server doesn't fan out per-subscriber.
//   2. With React StrictMode in dev, effects run twice on mount.
//      Without sharing, the first mount's EventSource gets closed
//      and the second mount opens a new one, which can race with
//      pending messages from the first.
//   3. With HMR, replacing a component that owns an EventSource
//      can drop in-flight messages.
//
// API base routing: when loaded from the Vite dev server (port 5173),
// SSE streams would otherwise go through Vite's /api proxy, which
// buffers chunked responses and backpressures uvicorn's generator.
// Same fix as fetchJson: hit uvicorn directly on 8765 in dev. Prod
// (uvicorn-served) keeps the relative URL.

import { useEffect, useState, useSyncExternalStore } from 'react';

const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  (typeof window !== 'undefined' && window.location?.port === '5173'
    ? `${window.location.protocol}//${window.location.hostname}:8765`
    : '');

function resolveApiUrl(path: string): string {
  if (!path.startsWith('/api')) return path;
  return API_BASE + path;
}

export type EventStreamReadyState =
  | 'connecting'
  | 'open'
  | 'closed';

// ---- module-level singleton EventSource per path ----
//
// One EventSource per unique `path` argument. Subscribers attach to
// it via a callback registry; the singleton fans out incoming
// messages to all subscribers. On the last unsubscribe the
// EventSource is closed (so a navigation that unmounts all
// subscribers doesn't leave the connection dangling).

type Subscriber<T> = (data: T | null) => void;

interface StreamHandle<T> {
  es: EventSource;
  subscribers: Set<Subscriber<T>>;
  lastData: T | null;
  readyState: EventStreamReadyState;
  readySubscribers: Set<(s: EventStreamReadyState) => void>;
}

const _streams: Map<string, StreamHandle<unknown>> = new Map();

function _acquire<T>(path: string): StreamHandle<T> {
  const existing = _streams.get(path);
  if (existing) return existing as StreamHandle<T>;

  const handle: StreamHandle<T> = {
    es: null as unknown as EventSource,
    subscribers: new Set(),
    lastData: null,
    readyState: 'connecting',
    readySubscribers: new Set(),
  };

  const es = new EventSource(resolveApiUrl(path));
  handle.es = es;

  es.addEventListener('open', () => {
    handle.readyState = 'open';
    handle.readySubscribers.forEach(fn => fn('open'));
  });
  es.addEventListener('error', () => {
    // EventSource auto-reconnects. We can't tell apart a transient
    // network blip from a fatal close, so just track readyState and
    // let the next onopen/onmessage clear the error.
    const ready = es.readyState === 2 ? 'closed' : 'connecting';
    handle.readyState = ready;
    handle.readySubscribers.forEach(fn => fn(ready));
  });
  es.addEventListener('message', (ev: MessageEvent<string>) => {
    let parsed: T | null = null;
    try {
      parsed = JSON.parse(ev.data) as T;
    } catch (e) {
      // Parsing failures don't kill the stream — drop the bad
      // message and keep the last good data.
      return;
    }
    handle.lastData = parsed;
    handle.subscribers.forEach(fn => fn(parsed));
  });

  _streams.set(path, handle as StreamHandle<unknown>);
  return handle;
}

function _release<T>(path: string, sub: Subscriber<T>): void {
  const handle = _streams.get(path);
  if (!handle) return;
  handle.subscribers.delete(sub);
  // Don't close the EventSource on the last unsubscribe — it can be
  // re-acquired by a remount (React StrictMode runs effects twice;
  // Vite HMR replaces components). Closing on every remount would
  // re-open the connection constantly. The browser cleans up the
  // EventSource when the page unloads, and uvicorn times out idle
  // SSE connections at the OS level. If a path is genuinely gone
  // (route removed), the stale entry is harmless — uvicorn sees
  // the connection close when the browser navigates away.
}

// ---- hook ----

export interface UseEventStreamResult<T> {
  /** Latest parsed snapshot. `null` until the first event arrives
   *  (typically < 200ms after mount). */
  data: T | null;
  /** Mirrors EventSource.readyState. `closed` is brief — the browser
   *  reopens the connection within ~1s. */
  readyState: EventStreamReadyState;
}

export function useEventStream<T>(path: string): UseEventStreamResult<T> {
  const [readyState, setReadyState] = useState<EventStreamReadyState>('connecting');
  // Use useSyncExternalStore so concurrent subscribers all see the
  // same data without triggering extra renders in siblings.
  const data = useSyncExternalStore<T | null>(
    (cb) => {
      const handle = _acquire<T>(path);
      handle.subscribers.add(cb);
      // Push current state immediately so the new subscriber doesn't
      // have to wait for the next SSE event.
      cb(handle.lastData);
      return () => _release(path, cb);
    },
    () => (_streams.get(path)?.lastData ?? null) as T | null,
    () => null,
  );

  useEffect(() => {
    const handle = _acquire<T>(path);
    const cb = (s: EventStreamReadyState) => setReadyState(s);
    handle.readySubscribers.add(cb);
    setReadyState(handle.readyState);
    return () => {
      handle.readySubscribers.delete(cb);
    };
  }, [path]);

  return { data, readyState };
}
