import { useEffect, useState } from 'react';
import { Modal } from './Modal';
import {
  getBackfillExifLogs,
  startBackfillExif,
} from '../api/endpoints';
import type { BackfillExifJobSnapshot } from '../api/endpoints';
import { useJobStream } from '../hooks/useJobStream';
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

  // Subscribe to the SSE stream for this backfill job. The server
  // pushes every progress tick (~1s cadence from the worker) so we
  // get sub-second updates without our own poll loop.
  const streamJobId = phase === 'running' && job ? job.job_id : null;
  const { job: streamJob } = useJobStream<BackfillExifJobSnapshot>(
    'backfill_exif_jobs',
    streamJobId,
  );

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
        setJob({
          job_id,
          status: 'running',
          started_at: new Date().toISOString(),
          finished_at: null,
          updated_count: 0,
          skipped_count: 0,
          no_exif_count: 0,
          missing_count: 0,
          error: null,
        });
        setPhase('running');
      } catch (err) {
        if (cancelled) return;
        push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 5000);
        setPhase('error');
      }
    })();
    return () => { cancelled = true; };
  }, [open, push]);

  // React to SSE status changes. Mirror the snapshot into local state
  // and transition to done/error.
  useEffect(() => {
    if (!streamJob) return;
    setJob(streamJob);
    if (streamJob.status === 'done') {
      setPhase('done');
      push({
        kind: streamJob.updated_count > 0 ? 'success' : 'info',
        message: streamJob.updated_count > 0
          ? `Backfilled EXIF on ${streamJob.updated_count.toLocaleString()} assets`
          : 'Nothing to backfill — every asset already has capture time',
      }, 4000);
    } else if (streamJob.status === 'error') {
      setPhase('error');
      push({ kind: 'error', message: streamJob.error ?? 'backfill failed' }, 5000);
    }
  }, [streamJob, push]);

  // Tail the log file while running.
  useEffect(() => {
    if (phase !== 'running' || !job) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const logResp = await getBackfillExifLogs(job.job_id, 80);
        if (cancelled) return;
        setLogs(logResp.lines);
      } catch (err) {
        if (cancelled) return;
        console.warn('backfill-exif log poll failed', err);
      }
    };
    tick();
    const t = setInterval(tick, 1000);
    return () => { cancelled = true; clearInterval(t); };
  }, [phase, job]);

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