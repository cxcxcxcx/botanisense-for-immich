import { useEffect, useState } from 'react';
import { Modal } from './Modal';
import {
  getBackfillExifLogs,
  pollBackfillExifJob,
  startBackfillExif,
} from '../api/endpoints';
import type { BackfillExifJobSnapshot } from '../api/endpoints';
import { useToasts } from './ToastProvider';

interface Props {
  open: boolean;
  onClose: () => void;
}

type Phase = 'starting' | 'running' | 'done' | 'error';

export function BackfillExifModal({ open, onClose }: Props) {
  const push = useToasts();
  const [phase, setPhase] = useState<Phase>('starting');
  const [job, setJob] = useState<BackfillExifJobSnapshot | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  // Reset on reopen so a stale job doesn't bleed across sessions.
  useEffect(() => {
    if (open) {
      setPhase('starting');
      setJob(null);
      setLogs([]);
    }
  }, [open]);

  // Kick off the job once on open. The "starting" phase is brief —
  // by the time React renders, the POST is in flight.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      try {
        const { job_id } = await startBackfillExif();
        if (cancelled) return;
        setPhase('running');
        // Poll loop: status + tail logs every second. The log tail
        // endpoint returns just the last N lines so we don't ship the
        // whole file each tick.
        const tick = async () => {
          try {
            const [snap, logResp] = await Promise.all([
              pollBackfillExifJob(job_id),
              getBackfillExifLogs(job_id, 80),
            ]);
            if (cancelled) return;
            setJob(snap);
            setLogs(logResp.lines);
            if (snap.status === 'done') {
              setPhase('done');
              push({
                kind: snap.updated_count > 0 ? 'success' : 'info',
                message: snap.updated_count > 0
                  ? `Backfilled EXIF on ${snap.updated_count.toLocaleString()} assets`
                  : 'Nothing to backfill — every asset already has capture time',
              }, 4000);
            } else if (snap.status === 'error') {
              setPhase('error');
              push({ kind: 'error', message: snap.error ?? 'backfill failed' }, 5000);
            }
          } catch (err) {
            if (cancelled) return;
            console.warn('backfill-exif poll failed', err);
          }
        };
        tick();
        const t = setInterval(tick, 1000);
        return () => clearInterval(t);
      } catch (err) {
        if (cancelled) return;
        push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 5000);
        setPhase('error');
      }
    })();
    return () => { cancelled = true; };
  }, [open, push]);

  // Summary line shown above the log tail. Hidden until the first
  // poll lands (the worker hasn't reported anything yet).
  const summary = (() => {
    if (!job) return null;
    if (phase === 'running') return <span className="text-yellow-300">Running… ({job.updated_count.toLocaleString()} updated so far)</span>;
    if (phase === 'done') return (
      <span className="text-green-300">
        ✓ Done — {job.updated_count.toLocaleString()} updated
        {job.skipped_count > 0 && `, ${job.skipped_count.toLocaleString()} skipped (already had EXIF)`}
        {job.no_exif_count > 0 && `, ${job.no_exif_count.toLocaleString()} had no EXIF`}
      </span>
    );
    if (phase === 'error') return <span className="text-red-300">✕ {job.error ?? 'failed'}</span>;
    return null;
  })();

  return (
    <Modal open={open} onClose={onClose} title="🕒 Backfill EXIF from Immich">
      <div className="flex flex-col gap-2 text-[0.85rem]">
        <p className="text-fg-1">
          Pulls EXIF capture time + GPS for every DB asset that doesn't have it yet. ~30s for the
          user's 17k library. Rows that already have <code className="font-mono text-fg">captured_at</code> are skipped.
        </p>
        <div className="flex items-center justify-between">
          <span>{summary ?? <span className="text-fg-1">Starting…</span>}</span>
          {job?.started_at && (
            <small className="text-fg-1 font-mono text-[11px]">
              started {new Date(job.started_at).toLocaleTimeString()}
            </small>
          )}
        </div>
        <pre className="m-0 p-2 px-3 bg-[#0a0a0a] text-[#bcb] text-[11px]/[1.4] font-mono max-h-80 overflow-auto whitespace-pre-wrap border border-[#2a2a2a] rounded">
          {logs.length === 0 ? '(waiting for first log lines…)' : logs.join('\n')}
        </pre>
        <div className="flex gap-2 justify-end pt-1">
          <button type="button" className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]" onClick={onClose}>
            {phase === 'running' ? 'Hide (keeps running)' : 'Close'}
          </button>
        </div>
      </div>
    </Modal>
  );
}