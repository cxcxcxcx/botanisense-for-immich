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

  // Sync from SSE snapshot to local state. data.logs may be missing
  // on the very first event before the snapshot builder populates
  // the log tail — keep the previous lines in that case so we don't
  // flash empty.
  useEffect(() => {
    if (!active) return;
    const newLines = data?.logs?.lines;
    if (!newLines) return;
    setLines(prev => {
      // Avoid pointless re-renders when the snapshot is identical.
      if (
        prev.length === newLines.length &&
        prev[prev.length - 1] === newLines[newLines.length - 1]
      ) {
        return prev;
      }
      return newLines;
    });
  }, [data, active]);

  // Auto-scroll only if user is already at the bottom.
  useEffect(() => {
    const el = preRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <section className="border border-line2 rounded-lg bg-bg-1 p-3">
      <header className="flex items-center justify-between mb-2">
        <h4 className="m-0 text-[12px] uppercase tracking-wide text-fg-1">
          Pipeline log
        </h4>
        <small className="text-fg-1 font-mono text-[11px]">
          {lines.length === 0 ? 'no lines yet' : `${lines.length} lines`}
        </small>
      </header>
      <pre
        ref={preRef}
        data-testid="log-tail"
        className="m-0 p-2 px-3 bg-[#0a0a0a] text-[#bcb] text-[11px]/[1.4] font-mono h-[30vh] min-h-[160px] overflow-auto whitespace-pre border border-[#2a2a2a] rounded"
      >
        {lines.length === 0 ? '(no log lines yet — run a pipeline to populate)' : lines.join('\n')}
      </pre>
    </section>
  );
}