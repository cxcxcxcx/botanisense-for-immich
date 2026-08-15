import { useEffect, useRef, useState } from 'react';
import { getLogs } from '../api/endpoints';

export function LogTail() {
  const [open, setOpen] = useState(false);
  const [lines, setLines] = useState<string[]>([]);
  const preRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const j = await getLogs(200);
        if (!cancelled) setLines(j.lines);
      } catch { /* transient */ }
    };
    tick();
    const t = setInterval(tick, 2000);
    return () => { cancelled = true; clearInterval(t); };
  }, [open]);

  // Auto-scroll only if user is already at the bottom.
  useEffect(() => {
    const el = preRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <details className="log-tail" onToggle={e => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>{open ? '▾' : '▸'} logs (click to expand)</summary>
      {open && (
        <pre ref={preRef} className="log-pre">
          {lines.join('\n')}
        </pre>
      )}
    </details>
  );
}