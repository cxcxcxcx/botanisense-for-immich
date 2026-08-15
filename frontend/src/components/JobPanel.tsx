import { useEffect, useRef, useState } from 'react';
import {
  pollPlantsAlbumJob,
  pollKingdomAlbumJob,
  pollClusterAlbumJob,
  previewPlantsAlbum,
  previewKingdomAlbum,
  previewClusterAlbum,
  startPlantsAlbum,
  startKingdomAlbum,
  startClusterAlbumSync,
  startSyncAllClusterAlbums,
} from '../api/endpoints';
import type { JobSnapshot } from '../api/types';
import { ApiError } from '../api/client';
import { useApp, useJob, type JobKind } from '../state/AppContext';
import { useToasts } from './ToastProvider';

interface JobConfig {
  title: () => string;
  confirmText: (n: number) => string;
  runningText: (added: number, total: number) => string;
  doneSubject: () => string;
  hasPreview: boolean;
  hasAlbumLink: boolean;
  preview: () => Promise<{ n_assets: number }>;
  start: () => Promise<{ job_id: string }>;
  poll: (jobId: string) => Promise<JobSnapshot>;
  extractProgress: (snap: JobSnapshot) => { added: number; total: number };
}

function buildConfigs(clusterId: number | undefined): Record<Exclude<JobKind, 'idle'>, JobConfig> {
  return {
    plants: {
      title: () => 'Tag as Plants album',
      confirmText: n => `Tag every Plantae-classified photo (${n.toLocaleString()}) as an Immich album?`,
      runningText: (added, total) => `Adding ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos added to Plantae album',
      hasPreview: true,
      hasAlbumLink: true,
      preview: () => previewPlantsAlbum(),
      start: () => startPlantsAlbum({}),
      poll: id => pollPlantsAlbumJob(id),
      extractProgress: snap => ({ added: (snap.n_added as number | undefined) ?? 0, total: (snap.n_assets as number | undefined) ?? 0 }),
    },
    fungi: {
      title: () => 'Tag as Fungi album',
      confirmText: n => `Tag every Fungi-classified photo (${n.toLocaleString()}) as an Immich album?`,
      runningText: (added, total) => `Adding ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos added to Fungi album',
      hasPreview: true,
      hasAlbumLink: true,
      preview: () => previewKingdomAlbum('Fungi'),
      start: () => startKingdomAlbum('Fungi', {}),
      poll: id => pollKingdomAlbumJob(id),
      extractProgress: snap => ({ added: (snap.n_added as number | undefined) ?? 0, total: (snap.n_assets as number | undefined) ?? 0 }),
    },
    kingdom: {
      title: () => 'Tag kingdom album',
      confirmText: n => `Tag ${n.toLocaleString()} photos as the kingdom album?`,
      runningText: (added, total) => `Adding ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos added to kingdom album',
      hasPreview: true,
      hasAlbumLink: true,
      preview: () => previewKingdomAlbum('Plantae'),
      start: () => startKingdomAlbum('Plantae', {}),
      poll: id => pollKingdomAlbumJob(id),
      extractProgress: snap => ({ added: (snap.n_added as number | undefined) ?? 0, total: (snap.n_assets as number | undefined) ?? 0 }),
    },
    cluster: {
      title: () => `Sync cluster #${clusterId ?? '?'} as album`,
      confirmText: n => `Sync every photo in cluster #${clusterId} (${n.toLocaleString()}) as an Immich album?`,
      runningText: (added, total) => `Adding ${added.toLocaleString()} / ${total.toLocaleString()} photos…`,
      doneSubject: () => 'photos added to cluster album',
      hasPreview: true,
      hasAlbumLink: true,
      preview: () => {
        if (clusterId == null) return Promise.resolve({ n_assets: 0 });
        return previewClusterAlbum(clusterId);
      },
      start: () => {
        if (clusterId == null) return Promise.reject(new Error('no cluster id'));
        return startClusterAlbumSync(clusterId);
      },
      poll: id => pollClusterAlbumJob(id),
      extractProgress: snap => ({ added: (snap.n_added as number | undefined) ?? 0, total: (snap.n_assets as number | undefined) ?? 0 }),
    },
    'sync-all': {
      title: () => 'Sync all clusters as albums',
      confirmText: () => 'Create one Immich album per cluster and add every photo?',
      runningText: (added, total) => `Syncing ${added.toLocaleString()} / ${total.toLocaleString()} photos across clusters…`,
      doneSubject: () => 'photos added across all cluster albums',
      hasPreview: false,
      hasAlbumLink: false,
      preview: () => Promise.reject(new Error('no preview')),
      start: () => startSyncAllClusterAlbums(),
      poll: id => pollClusterAlbumJob(id),
      extractProgress: snap => ({
        added: (snap.photos_added as number | undefined) ?? (snap.clusters_done as number | undefined) ?? 0,
        total: (snap.n_assets_total as number | undefined) ?? 0,
      }),
    },
  };
}

export function JobPanel() {
  const { state, dispatch } = useApp();
  const job = useJob();
  const push = useToasts();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [pollError, setPollError] = useState<string>('');

  // When the panel enters the confirm state, kick off the preview (if any).
  useEffect(() => {
    if (job.kind === 'idle') return;
    const activeJob = job;
    if (activeJob.state !== 'confirm') return;
    const cfg = buildConfigs(activeJob.payload.clusterId)[activeJob.kind];
    if (!cfg.hasPreview) return;
    let cancelled = false;
    cfg.preview()
      .then(r => {
        if (cancelled) return;
        if (!r.n_assets) dispatch({ type: 'JOB_PREVIEW_ERR', errorMsg: 'No matching photos at the current threshold.' });
        else dispatch({ type: 'JOB_PREVIEW_OK', total: r.n_assets });
      })
      .catch(e => {
        if (cancelled) return;
        const msg = e instanceof ApiError ? e.detail : (e instanceof Error ? e.message : String(e));
        dispatch({ type: 'JOB_PREVIEW_ERR', errorMsg: msg });
      });
    return () => { cancelled = true; };
    // job.kind & job.payload change when the user opens a new panel; we want
    // the preview to re-fetch when that happens.
  }, [job, dispatch]);

  // Poll the running job.
  useEffect(() => {
    if (job.kind === 'idle') return;
    const activeJob = job;
    if (activeJob.state !== 'running' || !activeJob.jobId) return;
    const cfg = buildConfigs(activeJob.payload.clusterId)[activeJob.kind];
    const jobId = activeJob.jobId;

    const loop = async () => {
      try {
        const snap = await cfg.poll(jobId);
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
  }, [job, dispatch, push]);

  if (job.kind === 'idle') return null;
  const active = job;

  const cfg = buildConfigs(active.payload.clusterId)[active.kind];

  const start = async () => {
    try {
      const r = await cfg.start();
      dispatch({ type: 'JOB_STARTED', jobId: r.job_id, total: active.progress.total });
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
    <aside className="job-panel-slot">
      <article>
        <header>
          <h4>{cfg.title()}</h4>
          <button className="close" rel="prev" aria-label="Close" onClick={hide}>✕</button>
        </header>

        {active.state === 'confirm' && (
          <div>
            <p>{cfg.confirmText(active.progress.total)}</p>
            <footer>
              <button type="button" className="secondary" onClick={hide}>Cancel</button>
              <button type="button" onClick={start}>Confirm</button>
            </footer>
          </div>
        )}

        {active.state === 'preview-error' && (
          <div>
            <p>✕ <span className="error">{active.errorMsg}</span></p>
            <footer>
              <button type="button" className="secondary" onClick={hide}>Dismiss</button>
              <button type="button" onClick={() => dispatch({ type: 'JOB_PREVIEW_OK', total: 0 })}>Retry</button>
            </footer>
          </div>
        )}

        {active.state === 'running' && (
          <div>
            <p>{cfg.runningText(active.progress.added, active.progress.total)}</p>
            <progress value={active.progress.added} max={Math.max(1, active.progress.total)} />
            <p>
              <button
                type="button"
                className="secondary outline"
                onClick={hide}
                title="Hide this panel — the job keeps running in the background"
              >
                Hide
              </button>
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
                  <a href={active.progress.album_url} target="_blank" rel="noopener noreferrer">
                    {active.progress.album_id.slice(0, 8)}…
                  </a>
                </>
              )}
              {active.progress.reused && <small className="muted"> (reused existing album)</small>}
            </p>
            <footer>
              <button type="button" className="secondary" onClick={hide}>Dismiss</button>
              <button type="button" onClick={() => dispatch({ type: 'JOB_PREVIEW_OK', total: active.progress.total })}>
                Run again
              </button>
            </footer>
          </div>
        )}

        {active.state === 'error' && (
          <div>
            <p>✕ <span className="error">{active.errorMsg || pollError}</span></p>
            <footer>
              <button type="button" className="secondary" onClick={hide}>Dismiss</button>
              <button type="button" onClick={start}>Retry</button>
            </footer>
          </div>
        )}
      </article>
    </aside>
  );
}