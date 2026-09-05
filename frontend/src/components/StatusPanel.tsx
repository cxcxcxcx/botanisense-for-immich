import { useEffect, useRef, useState } from 'react';
import { stopPipeline } from '../api/endpoints';
import type { EventSnapshot } from '../api/types';
import { useEventStream } from '../hooks/useEventStream';
import { fmtDuration } from '../lib/format';
import { useToasts } from './ToastProvider';

const STAGE_COLORS: Record<string, string> = {
  search: 'bg-stage-search',
  species: 'bg-stage-species',
  cluster: 'bg-stage-cluster',
  kingdom: 'bg-stage-kingdom',
  yolo: 'bg-stage-yolo',
  'bbox-bioclip': 'bg-stage-bioclip',
};

// Status badge classes must appear as LITERAL strings — Tailwind's
// purge only keeps the .status-* rules it can see spelled out in the
// source, and a template literal like `status-${status}` made the
// scanner drop them all, leaving the badge as an unstyled span
// (light theme: pale text floating with no chip background).
const STATUS_CLS: Record<string, string> = {
  idle: 'status-idle',
  running: 'status-running',
  done: 'status-done',
  stopped: 'status-stopped',
  error: 'status-error',
};

interface Props {
  /** When true, render only the badge + eta (browse-view header).
   *  Hides the progress bar + stage-time bar + legend. */
  compact?: boolean;
}

export function StatusPanel({ compact = false }: Props = {}) {
  // Subscribe to /api/events. The server only emits when state changes,
  // so an idle pipeline produces zero events; while running we get
  // ~200ms-cadence updates (cheaper + lower-latency than the old 1Hz
  // poll, and it stops hammering the server when nothing's happening).
  const { data } = useEventStream<EventSnapshot>('/api/events');
  const [stageTotals, setStageTotals] = useState<Record<string, number>>({});
  const prevStatusRef = useRef<string>('idle');
  const [stopping, setStopping] = useState(false);
  const push = useToasts();

  const handleStop = async () => {
    if (stopping) return;
    setStopping(true);
    try {
      const r = await stopPipeline();
      if (r.stopped) {
        push({ kind: 'info', message: 'Stop requested — will unwind at the next batch boundary' }, 3000);
      } else {
        push({ kind: 'info', message: 'No run was in progress' }, 2500);
      }
    } catch (err) {
      push({ kind: 'info', message: err instanceof Error ? err.message : String(err) }, 4000);
    } finally {
      // Re-enable after a short debounce so a rapid click doesn't
      // double-fire; the status panel will refresh on its own.
      window.setTimeout(() => setStopping(false), 1500);
    }
  };

  useEffect(() => {
    const pipeline = data?.pipeline;
    if (!pipeline) return;
    const cur = pipeline.status || 'idle';
    if (prevStatusRef.current !== 'idle' && cur !== 'running') {
      setStageTotals({});
    }
    prevStatusRef.current = cur;
    if (pipeline.stage_times) {
      setStageTotals(prev => {
        const next = { ...prev };
        for (const [k, v] of Object.entries(pipeline.stage_times)) {
          const v0 = next[k] || 0;
          next[k] = Math.max(v0, v);
        }
        return next;
      });
    }
  }, [data]);

  const pipeline = data?.pipeline;
  const status = pipeline?.status || 'idle';
  const stage = pipeline?.stage || '';
  const badgeText = stage ? `${status} · ${stage}` : status;
  const etaParts: string[] = [];
  if (pipeline?.rate != null) etaParts.push(`${pipeline.rate.toFixed(1)}/s`);
  if (pipeline?.eta_seconds != null) etaParts.push(`~${Math.ceil(pipeline.eta_seconds)}s left`);

  const projected: Record<string, number> = { ...stageTotals };
  if (pipeline && pipeline.status === 'running' && pipeline.stage && pipeline.stage_t0) {
    const elapsed = Math.max(0, Date.now() / 1000 - pipeline.stage_t0);
    projected[pipeline.stage] = (projected[pipeline.stage] || 0) + elapsed;
  }
  const totalDur = Object.values(projected).reduce((a, b) => a + b, 0);
  const bars = totalDur > 0
    ? Object.entries(projected).map(([k, dur]) => ({
        key: k, label: k, dur,
        pct: (dur / totalDur) * 100,
        cls: STAGE_COLORS[k] || '',
      }))
    : [];

  return (
    <div
      data-testid="status-panel"
      className={`flex flex-col gap-2 font-mono text-[12px]/[1.6] border border-line2 rounded-lg bg-bg-1 p-3 ${compact ? 'text-[11px]' : ''}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span data-testid="status-badge" className={STATUS_CLS[status] || 'status-idle'}>{badgeText}</span>
        {status === 'running' && (
          <button
            type="button"
            onClick={handleStop}
            disabled={stopping}
            className="px-2 py-0.5 rounded text-[11px] danger-btn"
            title="Stop the running pipeline at the next batch boundary"
          >
            {stopping ? 'stopping…' : 'Stop'}
          </button>
        )}
        <progress
          className="hidden"
          style={{ display: status === 'running' ? 'block' : 'none', width: '200px', flexShrink: 0, height: '6px' }}
          value={pipeline?.chunk_idx || 0}
          max={pipeline?.chunk_total || 1}
        />
        <small className="text-fg-1" style={{ display: pipeline?.eta_seconds != null ? 'inline' : 'none' }}>
          {etaParts.join(' · ')}
        </small>
        {/* In-app scheduler hint (UX-19b). Only renders when the
            server reports a `next_run_at` (i.e. PIPELINE_CRON is set).
            Shows the next scheduled time so the user knows when the
            next auto-run will fire. */}
        {pipeline?.next_run_at && (
          <small
            className="text-fg-2 ml-auto"
            title={`PIPELINE_CRON is set; the next run fires at ${pipeline.next_run_at}`}
          >
            Next run: {new Date(pipeline.next_run_at).toLocaleString()}
          </small>
        )}
      </div>
      {bars.length > 0 && (
        <>
          <div className={`flex h-1.5 rounded overflow-hidden bg-bg-2 ${compact ? 'hidden' : ''}`}>
            {bars.map(b => (
              <span
                key={b.key}
                className={`h-full ${b.cls}`}
                style={{ width: `${b.pct}%` }}
                title={`${b.label}: ${fmtDuration(b.dur)}`}
              />
            ))}
          </div>
          <small className={`flex flex-wrap gap-x-4 gap-y-1 w-full text-[10px] text-fg-0 ${compact ? 'hidden' : ''}`}>
            {bars.map(b => (
              <span key={b.key}>
                <span className={`inline-block w-2 h-2 rounded-sm mr-1 align-middle ${b.cls}`}></span>
                <span>{b.label}</span>
                <span className="text-fg-1"> {fmtDuration(b.dur)}</span>
              </span>
            ))}
          </small>
        </>
      )}
    </div>
  );
}