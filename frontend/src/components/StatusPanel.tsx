import { useEffect, useRef, useState } from 'react';
import { getStatus } from '../api/endpoints';
import type { PipelineStatus } from '../api/types';
import { usePolling } from '../hooks/usePolling';
import { cls, fmtDuration } from '../lib/format';

const STAGE_COLORS: Record<string, string> = {
  search: 'stage-search',
  species: 'stage-species',
  cluster: 'stage-cluster',
  kingdom: 'stage-kingdom',
  yolo: 'stage-yolo',
  'bbox-bioclip': 'stage-bbox-bioclip',
};

interface Props {
  /** When true, render only the badge + eta (used in the browse-view
   *  header where the full stage bar would be noise). The CSS class
   *  .status-compact hides the progress bar, stage-time bar, and
   *  stage legend. */
  compact?: boolean;
}

export function StatusPanel({ compact = false }: Props = {}) {
  const { data } = usePolling<PipelineStatus>(() => getStatus(), { intervalMs: 1000 });
  const [stageTotals, setStageTotals] = useState<Record<string, number>>({});
  const prevStatusRef = useRef<string>('idle');

  // Reset stageTotals when a run ends so the next run doesn't show stale slices.
  useEffect(() => {
    if (!data) return;
    const cur = data.status || 'idle';
    if (prevStatusRef.current !== 'idle' && cur !== 'running') {
      setStageTotals({});
    }
    prevStatusRef.current = cur;
    if (data.stage_times) {
      setStageTotals(prev => {
        const next = { ...prev };
        for (const [k, v] of Object.entries(data.stage_times)) {
          const v0 = next[k] || 0;
          next[k] = Math.max(v0, v);
        }
        return next;
      });
    }
  }, [data]);

  const status = data?.status || 'idle';
  const stage = data?.stage || '';
  const badgeText = stage ? `${status} · ${stage}` : status;
  const etaParts: string[] = [];
  if (data?.rate != null) etaParts.push(`${data.rate.toFixed(1)}/s`);
  if (data?.eta_seconds != null) etaParts.push(`~${Math.ceil(data.eta_seconds)}s left`);

  // Compute projected bars: snapshot totals + live running stage.
  const projected: Record<string, number> = { ...stageTotals };
  if (data && data.status === 'running' && data.stage && data.stage_t0) {
    const elapsed = Math.max(0, Date.now() / 1000 - data.stage_t0);
    projected[data.stage] = (projected[data.stage] || 0) + elapsed;
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
    <aside className={`status-panel${compact ? ' status-compact' : ''}`}>
      <span className={cls('badge', `status-${status}`)}>{badgeText}</span>
      <progress
        style={{ display: status === 'running' ? 'block' : 'none' }}
        value={data?.chunk_idx || 0}
        max={data?.chunk_total || 1}
      />
      <small style={{ display: data?.eta_seconds != null ? 'inline' : 'none' }}>
        {etaParts.join(' · ')}
      </small>
      {bars.length > 0 && (
        <div className="stage-time-bar">
          {bars.map(b => (
            <span
              key={b.key}
              className={`slice ${b.cls}`}
              style={{ width: `${b.pct}%` }}
              title={`${b.label}: ${fmtDuration(b.dur)}`}
            />
          ))}
        </div>
      )}
      {bars.length > 0 && (
        <small className="stage-time-legend">
          {bars.map(b => (
            <span key={b.key}>
              <span className={`swatch ${b.cls}`}></span>
              <span>{b.label}</span>
              <span className="stage-sec"> {fmtDuration(b.dur)}</span>
            </span>
          ))}
        </small>
      )}
    </aside>
  );
}