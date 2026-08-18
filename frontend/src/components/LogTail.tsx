import { useEffect, useRef, useState } from 'react';
import type { EventSnapshot } from '../api/types';
import { useEventStream } from '../hooks/useEventStream';

interface Props {
  /** When true (modal open) the stream is consumed; when false the
   *  log view freezes at its last-rendered lines (and the connection
   *  is dropped on unmount, courtesy of useEventStream). */
  active: boolean;
}

export function LogTail({ active }: Props) {
  // The log tail ships in /api/events under `logs.lines`. The server
  // pushes a fresh tail only when the log file's mtime changes, so
  // this hook pays zero cost when nothing is being logged.
  const { data } = useEventStream<EventSnapshot>('/api/events');
  const [lines, setLines] = useState<string[]>([]);
  const preRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!active || !data?.logs) return;
    setLines(data.logs.lines);
  }, [data, active]);

  // Auto-scroll only if user is already at the bottom.
  useEffect(() => {
    const el = preRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <pre ref={preRef} className="m-0 p-2 px-4 bg-[#0a0a0a] text-[#bcb] text-[11px]/[1.4] font-mono max-h-[30vh] overflow-auto whitespace-pre border border-[#2a2a2a] rounded">
      {lines.join('\n')}
    </pre>
  );
}