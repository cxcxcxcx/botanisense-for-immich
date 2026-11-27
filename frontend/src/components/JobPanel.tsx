import { useEffect, useRef, useState } from 'react';
import {
  pollPersonBackfillJob,
  previewKingdomAlbum,
  previewPersonBackfill,
  previewPurgeMissing,
  purgeMissing,
  startPersonBackfill,
  startKingdomAlbum,
  stopPersonBackfill,
} from '../api/endpoints';
import type { PersonBackfillJobSnapshot } from '../api/endpoints';
import type { JobSnapshot, PurgeMissingPreview, PurgeMissingResult } from '../api/types';

// Compact human label for the optional membership filter. Mirrors
// the ImmichSync component's PEOPLE_LABELS so the legacy JobPanel
// (purge-missing, person-backfill, etc.)
// shows the same scope the new Run-page UI uses. Returns '' when no
// filter is set, so existing call sites that don't carry the field
// render unchanged.
function peopleLabel(pf: unknown): string {
  if (pf === 0) return 'w/o people';
  if (pf === 1) return 'w/ people';
  return '';
}
import { ApiError } from '../api/client';
import { useApp, useJob, type JobKind } from '../state/AppContext';
import type { JobRegistryKey } from '../hooks/useJobStream';
import { useJobStream } from '../hooks/useJobStream';
import { useToasts } from './ToastProvider';

interface JobConfig {
  title: () => string;
  confirmText: (n: number) => string;
  runningText: (added: number, total: number) => string;
  doneSubject: () => string;
  hasPreview: boolean;
  hasAlbumLink: boolean;
  // `true` for synchronous jobs (e.g. purge-missing) that complete in
  // a single round-trip — `start()` returns the final result and we
  // dispatch JOB_DONE directly without entering the stream.
  isSync?: boolean;
  // Registry key in the SSE snapshot where this job's progress lives.
  // Null for sync jobs (no stream subscription needed). Also null for
  // person-backfill, which runs in a child process that writes its
  // status to a JSON file — the in-memory dict doesn't update on its
  // own, so we keep the REST poll for that one.
  registryKey: JobRegistryKey | null;
  // Optional REST poll for jobs whose state doesn't flow through the
  // in-memory registry (currently just person-backfill). When set,
  // takes precedence over the SSE registry.
  poll?: (jobId: string) => Promise<JobSnapshot>;
  preview: () => Promise<{ n_assets: number; sample?: PurgeMissingPreview['sample'] }>;
  // Sync jobs return their own result shape; async jobs return
  // {job_id: string} (or {job_id: null} for noop short-circuits, which
  // the start() handler in JobPanel recognizes).
  start: () => Promise<{ job_id: string | null } | PurgeMissingResult>;
  extractProgress: (snap: JobSnapshot) => { added: number; total: number };
}

function buildConfigs(): Record<Exclude<JobKind, 'idle'>, JobConfig> {
  return {
    plants: {
      title: () => 'Tag as Plants album',
      confirmText: n => `Tag every Plantae-classified photo (${n.toLocaleString()}) as an Immich album?`,
      runningText: (added, total) => `Adding ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos added to Plantae album',
      hasPreview: true,
      hasAlbumLink: true,
      registryKey: 'kingdom_jobs',
      preview: () => previewKingdomAlbum('Plantae'),
      start: () => startKingdomAlbum('Plantae', {}),
      extractProgress: snap => ({ added: (snap.n_added as number | undefined) ?? 0, total: (snap.n_assets as number | undefined) ?? 0 }),
    },
    fungi: {
      title: () => 'Tag as Fungi album',
      confirmText: n => `Tag every Fungi-classified photo (${n.toLocaleString()}) as an Immich album?`,
      runningText: (added, total) => `Adding ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos added to Fungi album',
      hasPreview: true,
      hasAlbumLink: true,
      registryKey: 'kingdom_jobs',
      preview: () => previewKingdomAlbum('Fungi'),
      start: () => startKingdomAlbum('Fungi', {}),
      extractProgress: snap => ({ added: (snap.n_added as number | undefined) ?? 0, total: (snap.n_assets as number | undefined) ?? 0 }),
    },
    kingdom: {
      title: () => 'Tag kingdom album',
      confirmText: n => `Tag ${n.toLocaleString()} photos as the kingdom album?`,
      runningText: (added, total) => `Adding ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos added to kingdom album',
      hasPreview: true,
      hasAlbumLink: true,
      registryKey: 'kingdom_jobs',
      preview: () => previewKingdomAlbum('Plantae'),
      start: () => startKingdomAlbum('Plantae', {}),
      extractProgress: snap => ({ added: (snap.n_added as number | undefined) ?? 0, total: (snap.n_assets as number | undefined) ?? 0 }),
    },
    'purge-missing': {
      title: () => 'Purge assets deleted from Immich',
      confirmText: n => `${n.toLocaleString()} assets are in the local DB but not in Immich. Delete them? (If you just trashed something in Immich, wait for the trash TTL to expire before purging.)`,
      runningText: (added, total) => `Deleting ${added.toLocaleString()} / ${total.toLocaleString()} assets…`,
      doneSubject: () => 'assets purged',
      hasPreview: true,
      hasAlbumLink: false,
      // Synchronous job — start() returns the final result directly,
      // no stream, no progress bar.
      isSync: true,
      registryKey: null,
      preview: () => previewPurgeMissing(),
      start: () => purgeMissing(),
      extractProgress: () => ({ added: 0, total: 0 }),
    },
    // Standalone person-detection backfill — runs ONLY the YOLO
    // person stage (no BioCLIP). Fills `assets.has_person` for every
    // asset where it's currently NULL so the "has person" filter
    // works on the existing library.
    'person-backfill': {
      title: () => 'Detect people in existing photos',
      confirmText: n => `Run YOLO over ${n.toLocaleString()} photos to detect people (~70 min for ~19k photos). No BioCLIP is involved — just the 17-label person+plant model. Skip if your library is small or you've never enabled the person filter.`,
      runningText: (added, total) => `Scanning ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos scanned for person',
      hasPreview: true,
      hasAlbumLink: false,
      // Multiprocess job — child writes status JSON, parent reads it
      // via /api/person/backfill/job/{id}. In-memory dict doesn't get
      // updated by the child, so the SSE stream wouldn't see progress.
      // Falls through to the REST poll below.
      registryKey: null,
      poll: id => pollPersonBackfillJob(id) as unknown as Promise<JobSnapshot>,
      preview: () => previewPersonBackfill(),
      start: () => startPersonBackfill().then(r => {
        // Backend returns {noop: true, n_to_scan: 0, job_id: null}
        // when nothing needs scanning — fall through to JOB_DONE
        // immediately so the panel shows "nothing to do" instead of
        // hanging on "running".
        if (r.noop || !r.job_id) {
          return { job_id: null };
        }
        return { job_id: r.job_id };
      }),
      extractProgress: snap => {
        // Cast through unknown — PersonBackfillJobSnapshot is the
        // concrete shape returned by the poll endpoint; JobSnapshot
        // is the generic union.
        const s = snap as unknown as PersonBackfillJobSnapshot;
        return {
          added: s.scanned_count ?? 0,
          total: s.n_to_scan ?? 0,
        };
      },
    },
  };
}

export function JobPanel() {
  const { state, dispatch } = useApp();
  const job = useJob();
  const push = useToasts();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [pollError, setPollError] = useState<string>('');

  // Resolve the active job's SSE registry key (or null if it should
  // use REST polling / no polling). Held outside the useEffect so the
  // hook order is stable across renders.
  const activeCfg =
    job.kind === 'idle'
      ? null
      : buildConfigs()[job.kind];
  const sseRegistryKey: JobRegistryKey | null =
    activeCfg && !activeCfg.isSync && !activeCfg.poll
      ? activeCfg.registryKey
      : null;
  const sseJobId: string | null =
    sseRegistryKey && job.kind !== 'idle' && job.jobId ? job.jobId : null;
  // useJobStream always opens the EventSource (it's a hook, can't be
  // conditional), but returns null until both key + jobId are set.
  // We pass 'kingdom_jobs' as a dummy fallback that will never match
  // since jobId is null in that case.
  const { job: streamJob } = useJobStream<JobSnapshot>(
    sseRegistryKey ?? 'kingdom_jobs',
    sseJobId,
  );
  // Stable per-run identity for effect deps: the whole `job` object is
  // replaced on every dispatch, so keying effects on it restarts them
  // at dispatch cadence (RTT) instead of their intended cadence.
  const jobRunKey =
    job.kind === 'idle'
      ? ''
      : `${job.kind}:${job.state}:${job.jobId ?? ''}:${job.payload.kingdom ?? ''}`;

  // When the panel enters the confirm state, kick off the preview (if any).
  // Guarded by a signature ref: JOB_PREVIEW_OK replaces the job object
  // (state stays 'confirm'), so keying this effect on `job` alone would
  // re-fetch in a loop — for purge-missing each fetch is a server-side
  // Immich dry-run scan. Re-fetch only when the panel identity changes.
  const previewedForRef = useRef<string | null>(null);
  useEffect(() => {
    if (job.kind === 'idle') return;
    const activeJob = job;
    if (activeJob.state !== 'confirm') return;
    const cfg = buildConfigs()[activeJob.kind];
    if (!cfg.hasPreview) return;
    const sig = `${activeJob.kind}:${JSON.stringify(activeJob.payload)}`;
    if (previewedForRef.current === sig) return;
    previewedForRef.current = sig;
    let cancelled = false;
    cfg.preview()
      .then(r => {
        if (cancelled) return;
        if (!r.n_assets) dispatch({ type: 'JOB_PREVIEW_ERR', errorMsg: 'No matching photos at the current threshold.' });
        else dispatch({ type: 'JOB_PREVIEW_OK', total: r.n_assets, sample: r.sample });
      })
      .catch(e => {
        if (cancelled) return;
        const msg = e instanceof ApiError ? e.detail : (e instanceof Error ? e.message : String(e));
        dispatch({ type: 'JOB_PREVIEW_ERR', errorMsg: msg });
      });
    return () => { cancelled = true; };
  }, [job, dispatch]);

  // Track the running job. Two paths:
  //   - registryKey != null: subscribe to the SSE stream and react to
  //     snapshots that include this jobId. Sub-second latency, no
  //     network traffic when the snapshot hasn't changed.
  //   - poll != null: fall back to a REST poll (currently only used
  //     for person-backfill, which runs in a child process that writes
  //     its status to a JSON file).
  useEffect(() => {
    if (job.kind === 'idle') return;
    const activeJob = job;
    if (activeJob.state !== 'running' || !activeJob.jobId) return;
    const cfg = buildConfigs()[activeJob.kind];
    // Sync jobs complete in start() — nothing to poll.
    if (cfg.isSync) return;
    const jobId = activeJob.jobId;

    // ---- REST poll branch (person-backfill) --------------------------
    if (cfg.poll) {
      const loop = async () => {
        try {
          const snap = await cfg.poll!(jobId);
          setPollError('');
          if (snap.status === 'running') {
            const { added, total } = cfg.extractProgress(snap);
            dispatch({ type: 'JOB_PROGRESS', added, total });
            timerRef.current = setTimeout(loop, 400);
            return;
          }
          if (snap.status === 'done') {
            const { added } = cfg.extractProgress(snap);
            dispatch({
              type: 'JOB_DONE',
              added,
              album_id: (snap.album_id as string | null | undefined) ?? null,
              album_url: (snap.album_url as string | null | undefined) ?? null,
              reused: Boolean(snap.reused_existing),
            });
            push({
              kind: 'success',
              message: `Done — ${added.toLocaleString()} photos added`,
            }, 4000);
            return;
          }
          if (snap.status === 'error') {
            dispatch({ type: 'JOB_ERROR', errorMsg: (snap.error as string | undefined) ?? 'unknown error' });
            return;
          }
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) {
            // Job aged out — treat as best-effort done.
            dispatch({ type: 'JOB_HIDE' });
            return;
          }
          const msg = e instanceof Error ? e.message : String(e);
          setPollError(msg);
          dispatch({ type: 'JOB_ERROR', errorMsg: msg });
        }
      };
      loop();
      return () => {
        if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
      };
    }

    // ---- SSE branch (plants / fungi / kingdom) ------------------------
    // Handled by the streamJob-driven effect below — no imperative
    // loop needed.
    return undefined;
    // Keyed on the stable per-run identity — the whole `job` object is
    // replaced by every JOB_PROGRESS dispatch, which would restart this
    // effect (and re-poll) at RTT cadence instead of every 400 ms.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobRunKey, dispatch, push]);

  // React to SSE snapshots for the running job. `streamJob` is the
  // job's current state from the latest server-emitted snapshot; each
  // new value means the server pushed an update.
  useEffect(() => {
    if (!streamJob) return;
    const cfg = activeCfg;
    if (!cfg) return;
    // Don't re-dispatch once we've moved to a terminal state — the
    // server keeps emitting snapshots of finished jobs (until TTL)
    // and we don't want to clobber JOB_DONE with stale JOB_PROGRESS.
    if (job.kind === 'idle' || job.state === 'done' || job.state === 'error') return;
    setPollError('');
    if (streamJob.status === 'done') {
      const { added } = cfg.extractProgress(streamJob);
      dispatch({
        type: 'JOB_DONE',
        added,
        album_id: (streamJob.album_id as string | null | undefined) ?? null,
        album_url: (streamJob.album_url as string | null | undefined) ?? null,
        reused: Boolean(streamJob.reused_existing),
      });
      push({
        kind: 'success',
        message: `Done — ${added.toLocaleString()} photos added`,
      }, 4000);
      return;
    }
    if (streamJob.status === 'error') {
      dispatch({
        type: 'JOB_ERROR',
        errorMsg: (streamJob.error as string | undefined) ?? 'unknown error',
      });
      return;
    }
    // status === 'running' (or any other in-progress state)
    const { added, total } = cfg.extractProgress(streamJob);
    // Only dispatch when the numbers actually moved. JOB_PROGRESS
    // always returns a new state object, so an unconditional dispatch
    // here re-rendered the app, re-ran this effect (its deps include
    // the fresh `job`), and dispatched again — an infinite loop that
    // pegged a core for the whole job duration.
    if (job.progress.added === added && job.progress.total === total) return;
    dispatch({ type: 'JOB_PROGRESS', added, total });
  }, [streamJob, activeCfg, job, dispatch, push]);

  if (job.kind === 'idle') return null;
  const active = job;

  const cfg = buildConfigs()[active.kind];

  const start = async () => {
    try {
      const r = await cfg.start();
      // Sync jobs (e.g. purge-missing) complete in this round-trip —
      // dispatch JOB_DONE directly with the deleted count. Async jobs
      // enter the polling loop via JOB_STARTED.
      if (cfg.isSync) {
        const syncResult = r as PurgeMissingResult;
        const deleted = syncResult.deleted_count ?? 0;
        dispatch({
          type: 'JOB_DONE',
          added: deleted,
          album_id: null,
          album_url: null,
          reused: false,
        });
        push({
          kind: deleted > 0 ? 'success' : 'info',
          message: deleted > 0
            ? `Purged ${deleted.toLocaleString()} deleted assets`
            : 'Nothing to purge — local DB is in sync with Immich',
        }, 4000);
        return;
      }
      // Noop short-circuit for jobs whose start() endpoint can
      // immediately resolve to "nothing to do" (e.g. person-backfill
      // when all assets already scanned).
      const jobId = (r as { job_id: string | null }).job_id;
      if (!jobId) {
        dispatch({
          type: 'JOB_DONE',
          added: active.progress.total,
          album_id: null,
          album_url: null,
          reused: false,
        });
        push({ kind: 'info', message: 'Nothing to scan — already complete' }, 3000);
        return;
      }
      dispatch({ type: 'JOB_STARTED', jobId, total: active.progress.total });
    } catch (e) {
      const msg = e instanceof ApiError ? e.detail : (e instanceof Error ? e.message : String(e));
      dispatch({ type: 'JOB_ERROR', errorMsg: msg });
    }
  };

  const hide = () => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    dispatch({ type: 'JOB_HIDE' });
  };

  return (
    <aside className="fixed bottom-6 right-6 left-6 sm:left-auto z-[100] sm:max-w-[32rem]">
      <article className="bg-bg-1 border border-line2 rounded-lg p-4 shadow-[0_8px_32px_rgba(0,0,0,0.4)]">
        <header className="flex justify-between items-center mb-2 border-b border-line pb-2">
          <h4 className="m-0 text-base">{cfg.title()}</h4>
          <button
            className="text-fg-1 hover:text-fg text-lg leading-none px-2"
            aria-label="Close"
            onClick={hide}
          >×</button>
        </header>

        {active.state === 'confirm' && (
          <div>
            <p>{cfg.confirmText(active.progress.total)}</p>
            {active.progress.sample.length > 0 && (
              <ul className="list-none p-0 mt-1 mb-2 border border-line2 rounded bg-bg-0 max-h-40 overflow-auto text-[11px] font-mono">
                {active.progress.sample.map(s => (
                  <li key={s.asset_id} className="flex justify-between gap-2 py-0.5 px-2 border-t border-line first:border-t-0">
                    <span className="text-fg overflow-hidden text-ellipsis whitespace-nowrap flex-1 min-w-0">{s.original_file_name}</span>
                    {s.cluster_id != null && (
                      <small className="text-fg-1 flex-shrink-0">cluster #{s.cluster_id}</small>
                    )}
                  </li>
                ))}
              </ul>
            )}
            <footer className="flex gap-2 mt-2 text-right">
              <button type="button" className="px-2.5 py-1 rounded border border-line2 bg-transparent text-fg text-sm hover:bg-bg-2 hover:border-accent" onClick={hide}>Cancel</button>
              <button type="button" className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover" onClick={start}>Confirm</button>
            </footer>
          </div>
        )}

        {active.state === 'preview-error' && (
          <div>
            <p>✕ <span className="job-error">{active.errorMsg}</span></p>
            <footer className="flex gap-2 mt-2 justify-end">
              <button type="button" className="px-2.5 py-1 rounded border border-line2 bg-transparent text-fg text-sm hover:bg-bg-2 hover:border-accent" onClick={hide}>Dismiss</button>
              <button type="button" className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover" onClick={() => dispatch({ type: 'JOB_PREVIEW_OK', total: 0 })}>Retry</button>
            </footer>
          </div>
        )}

        {active.state === 'running' && (
          <div>
            <p>
              {cfg.runningText(active.progress.added, active.progress.total)}
              {/* Show the membership scope next to the running text when
                  the server stored one. For ImmichSync-driven jobs the
                  scope is already visible in its inline progress, so
                  this is mostly cosmetic. */}
              {streamJob && peopleLabel(streamJob.people_filter) && (
                <small className="text-fg-1 ml-2">
                  ({peopleLabel(streamJob.people_filter)})
                </small>
              )}
            </p>
            <progress value={active.progress.added} max={Math.max(1, active.progress.total)} className="w-full h-1.5" />
            <p className="flex gap-2">
              <button
                type="button"
                className="px-2.5 py-1 rounded border border-line2 bg-transparent text-fg text-sm hover:bg-bg-2 hover:border-accent"
                onClick={hide}
                title="Hide this panel — the job keeps running in the background"
              >
                Hide
              </button>
              {/* Person-backfill has its own Stop endpoint (the main
                  pipeline uses the existing /api/run/stop). Show the
                  button only for the kinds that support stopping. */}
              {active.kind === 'person-backfill' && (
                <button
                  type="button"
                  className="px-2.5 py-1 rounded text-sm danger-btn"
                  onClick={async () => {
                    try {
                      await stopPersonBackfill(active.jobId!);
                      push({ kind: 'info', message: 'Stop requested — will unwind at the next batch boundary' }, 3000);
                    } catch (e) {
                      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
                    }
                  }}
                  title="Stop the backfill at the next batch boundary"
                >
                  Stop
                </button>
              )}
            </p>
          </div>
        )}

        {active.state === 'done' && (
          <div>
            <p>
              ✓ Done — <strong>{active.progress.added.toLocaleString()}</strong> {cfg.doneSubject()}
              {cfg.hasAlbumLink && active.progress.album_id && active.progress.album_url && (
                <>
                  {' '}
                  <a href={active.progress.album_url} target="_blank" rel="noopener noreferrer" className="text-[#7ad] hover:underline">
                    {active.progress.album_id.slice(0, 8)}…
                  </a>
                </>
              )}
              {active.progress.reused && <small className="text-fg-1"> (reused existing album)</small>}
            </p>
            <footer className="flex gap-2 mt-2 justify-end">
              <button type="button" className="px-2.5 py-1 rounded border border-line2 bg-transparent text-fg text-sm hover:bg-bg-2 hover:border-accent" onClick={hide}>Dismiss</button>
              <button type="button" className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover" onClick={() => dispatch({ type: 'JOB_PREVIEW_OK', total: active.progress.total })}>
                Run again
              </button>
            </footer>
          </div>
        )}

        {active.state === 'error' && (
          <div>
            <p>✕ <span className="text-red-300">{active.errorMsg || pollError}</span></p>
            <footer className="flex gap-2 mt-2 justify-end">
              <button type="button" className="px-2.5 py-1 rounded border border-line2 bg-transparent text-fg text-sm hover:bg-bg-2 hover:border-accent" onClick={hide}>Dismiss</button>
              <button type="button" className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover" onClick={start}>Retry</button>
            </footer>
          </div>
        )}
      </article>
    </aside>
  );
}