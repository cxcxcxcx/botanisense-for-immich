import { useEffect, useRef, useState } from 'react';
import { getLogs } from '../api/endpoints';

interface Props {
  /** When true (modal open) the polling interval runs; when false it
   *  cleans up. Same UX as the old <details> toggle but the visibility
   *  comes from the modal lifecycle rather than a local <details> open. */
  active: boolean;
}

export function LogTail({ active }: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const preRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!active) return;
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
  }, [active]);

  // Auto-scroll only if user is already at the bottom.
  useEffect(() => {
    const el = preRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <pre ref={preRef} className="log-pre">
      {lines.join('\n')}
    </pre>
  );
}