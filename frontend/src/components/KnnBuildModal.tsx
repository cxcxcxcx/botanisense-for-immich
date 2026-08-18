import { useEffect, useState } from 'react';
import { Modal } from './Modal';
import {
  getKnnBuildLogs,
  getKnnStatus,
  startKnnBuild,
} from '../api/endpoints';
import type { KnnBuildArgs, KnnBuildJobSnapshot } from '../api/types';
import { useJobStream } from '../hooks/useJobStream';
import { useToasts } from './ToastProvider';

interface Props {
  open: boolean;
  onClose: () => void;
}

type Phase = 'form' | 'running' | 'done' | 'error';

const FIELD_CLS =
  'block text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[280px] flex-[1_1_280px]';
const INPUT_CLS =
  'bg-bg-2 text-fg border border-line2 px-2.5 py-[5px] rounded text-[0.85rem] font-mono focus:outline-none focus:border-accent w-full';

export function KnnBuildModal({ open, onClose }: Props) {
  const push = useToasts();
  const [phase, setPhase] = useState<Phase>('form');
  const [args, setArgs] = useState<KnnBuildArgs>({
    parquet_glob: '/data/TOL/*.parquet',
    tag: 'v1',
    index_dir: '/media/4T_Ext/bioclip_inventory/index',
    kingdoms: '',
    index_type: 'ivf',
  });
  const [job, setJob] = useState<KnnBuildJobSnapshot | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  // Subscribe to the SSE stream for this build job. While we're in
  // the running phase and have a job_id, useJobStream gives us the
  // server's latest snapshot — no 1Hz REST poll needed.
  const streamJobId = phase === 'running' && job ? job.job_id : null;
  const { job: streamJob } = useJobStream<KnnBuildJobSnapshot>(
    'knn_build_jobs',
    streamJobId,
  );

  // Reset state every time the modal reopens so a stale job doesn't
  // leak from a previous session.
  useEffect(() => {
    if (open) {
      setPhase('form');
      setJob(null);
      setLogs([]);
      setSubmitting(false);
    }
  }, [open]);

  // React to SSE status changes. Logs are still REST-polled because
  // they're a long file tail (cheaper than SSE-firing on every line).
  useEffect(() => {
    if (!streamJob) return;
    if (streamJob.status === 'done') {
      setPhase('done');
      push({ kind: 'success', message: 'KNN index built — refresh to use' }, 4000);
      getKnnStatus().catch(() => {});
    } else if (streamJob.status === 'error') {
      setPhase('error');
      push({ kind: 'error', message: streamJob.error ?? 'build failed' }, 5000);
    }
    // Mirror started_at from the server so the header timer is right
    // (client clock might drift from the server's).
    if (streamJob.started_at && (!job || job.started_at !== streamJob.started_at)) {
      setJob(streamJob);
    }
  }, [streamJob, push, job]);

  // Tail the log file while running. The log endpoint returns just the
  // tail (default 50 lines) so we don't ship the whole file each tick.
  useEffect(() => {
    if (phase !== 'running' || !job) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const logResp = await getKnnBuildLogs(job.job_id, 80);
        if (cancelled) return;
        setLogs(logResp.lines);
      } catch (err) {
        if (cancelled) return;
        console.warn('knn-build log poll failed', err);
      }
    };
    tick();
    const t = setInterval(tick, 1000);
    return () => { cancelled = true; clearInterval(t); };
  }, [phase, job]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const { job_id } = await startKnnBuild({
        ...args,
        kingdoms: args.kingdoms?.trim() || null,
        index_dir: args.index_dir?.trim() || null,
        index_type: args.index_type?.trim() || null,
      });
      setJob({ job_id, status: 'running', started_at: new Date().toISOString(), finished_at: null, error: null });
      setPhase('running');
    } catch (err) {
      push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 4000);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="🧠 Build KNN index">
      {phase === 'form' && (
        <form onSubmit={handleSubmit} className="flex flex-col gap-3 text-[0.85rem]">
          <p className="text-fg-1">
            Builds a FAISS IVF index from TreeOfLife-200M embeddings. Heavy: ~5–30 min depending on
            dataset size and CPU. Output: <code className="font-mono text-fg">{args.index_dir}</code>.
          </p>
          <label className={FIELD_CLS}>
            Parquet glob
            <input
              type="text"
              className={INPUT_CLS}
              value={args.parquet_glob}
              onChange={e => setArgs({ ...args, parquet_glob: e.target.value })}
              placeholder="/data/TOL/*.parquet"
              required
            />
          </label>
          <label className={FIELD_CLS}>
            Tag (version)
            <input
              type="text"
              className={INPUT_CLS}
              value={args.tag}
              onChange={e => setArgs({ ...args, tag: e.target.value })}
              placeholder="v1"
              required
            />
          </label>
          <label className={FIELD_CLS}>
            Index dir
            <input
              type="text"
              className={INPUT_CLS}
              value={args.index_dir ?? ''}
              onChange={e => setArgs({ ...args, index_dir: e.target.value })}
              placeholder="/media/4T_Ext/bioclip_inventory/index"
            />
          </label>
          <label className={FIELD_CLS}>
            Kingdoms (optional, comma-separated)
            <input
              type="text"
              className={INPUT_CLS}
              value={args.kingdoms ?? ''}
              onChange={e => setArgs({ ...args, kingdoms: e.target.value })}
              placeholder="Plantae,Fungi"
            />
          </label>
          <label className={FIELD_CLS}>
            Index type
            <select
              className={INPUT_CLS}
              value={args.index_type ?? 'ivf'}
              onChange={e => setArgs({ ...args, index_type: e.target.value })}
            >
              <option value="ivf">ivf (default — fast for large libraries)</option>
              <option value="flat">flat (slow but exact — small libraries only)</option>
            </select>
          </label>
          <div className="flex gap-2 justify-end pt-1">
            <button type="button" className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]" onClick={onClose}>Cancel</button>
            <button type="submit" disabled={submitting} className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-50">
              {submitting ? 'Starting…' : 'Build'}
            </button>
          </div>
        </form>
      )}

      {(phase === 'running' || phase === 'done' || phase === 'error') && (
        <div className="flex flex-col gap-2 text-[0.85rem]">
          <div className="flex items-center justify-between">
            <span>
              {phase === 'running' && (
                <span className="text-yellow-300">Building… (poll updates every second)</span>
              )}
              {phase === 'done' && (
                <span className="text-green-300">✓ Done — index written to <code className="font-mono">{args.index_dir}</code></span>
              )}
              {phase === 'error' && (
                <span className="text-red-300">✕ Failed — see log tail below</span>
              )}
            </span>
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
            {phase === 'done' || phase === 'error' ? (
              <button type="button" className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]" onClick={onClose}>Close</button>
            ) : (
              <button type="button" className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]" onClick={onClose}>Hide (keeps running)</button>
            )}
          </div>
        </div>
      )}
    </Modal>
  );
}