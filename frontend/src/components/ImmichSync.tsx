// "Sync to IMMICH" section on the Run page. Four actions (Sync
// clusters all / Sync Plantae / Sync Fungi / Remove synced clusters)
// each openable from a button. Defaults (people_filter per action +
// min_score per kingdom) are persisted server-side via
// GET/PUT /api/settings/sync-defaults and pre-fill the per-action
// confirm modals.
//
// Per-action overrides live in this component's local state — they
// intentionally don't go through the global reducer because they
// shouldn't survive a remount or tab switch.
//
// Progress is shown inline under each button so multiple actions can
// run concurrently without a shared JobPanel competing for the same
// DOM. The legacy JobPanel still serves the per-cluster pill sync
// (cluster kind) and the unrelated purge / person-backfill flows.

import { useEffect, useState } from 'react';
import {
  getSyncDefaults,
  putSyncDefaults,
  previewKingdomAlbum,
  previewPlantsAlbum,
  listClusterAlbums,
  startSyncAllClusterAlbums,
  startKingdomAlbum,
  startPlantsAlbum,
  removeClusterAlbums,
} from '../api/endpoints';
import type { SyncDefaults } from '../api/types';
import { useApp, useSyncDefaults } from '../state/AppContext';
import { useEventStream } from '../hooks/useEventStream';
import type { EventSnapshot, JobSnapshot } from '../api/types';
import { useToasts } from './ToastProvider';
import { Modal } from './Modal';

// ---------- people-filter chip group ---------------------------------------
// Per the mock: each rule row has a 3-button segmented control
// (Any / Without people / With people) that spans the full width of
// the section. Replaces the older radio-button row which felt less
// like an "options" picker and more like a form field.

type PeopleFilter = 0 | 1 | null;
const PEOPLE_OPTS: { value: PeopleFilter; label: string }[] = [
  { value: null, label: 'Any' },
  { value: 0,    label: 'Without people' },
  { value: 1,    label: 'With people' },
];

function PeopleFilterChips({
  value,
  onChange,
}: {
  value: PeopleFilter;
  onChange: (v: PeopleFilter) => void;
}) {
  return (
    <div className="flex bg-bg-2 p-1 rounded-lg border border-line2/50" role="radiogroup" aria-label="People filter">
      {PEOPLE_OPTS.map(opt => (
        <button
          key={String(opt.value)}
          type="button"
          role="radio"
          aria-checked={value === opt.value}
          onClick={() => onChange(opt.value)}
          className={`flex-1 py-1.5 text-[10px] font-medium rounded-md transition-all ${
            value === opt.value ? 'bg-bg-1 text-fg' : 'text-fg-1 hover:text-fg-0'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

// ---------- per-action progress slot --------------------------------------
// Each of the four buttons can be in one of these states:
//   idle  — no job has run yet / last job settled
//   running — a job_id is in flight; the inline progress bar shows
//   done  — terminal success; render the album link inline
//   error — terminal failure; render the error message
//
// Keeping state per-action in this component (not the global reducer)
// lets multiple actions run at once without stepping on each other.

interface ActionSlot {
  jobId: string | null;
  status: 'idle' | 'running' | 'done' | 'error';
  total: number;
  added: number;
  albumId: string | null;
  albumUrl: string | null;
  error: string;
}

function emptySlot(): ActionSlot {
  return { jobId: null, status: 'idle', total: 0, added: 0, albumId: null, albumUrl: null, error: '' };
}

// Per-action identifier. Each maps to a registry key in the SSE
// snapshot so we can read the live progress without an extra REST
// poll loop.
type ActionKey = 'sync-all' | 'plantae' | 'fungi' | 'remove';

function registryFor(action: ActionKey): 'cluster_album_jobs' | 'kingdom_jobs' | 'plants_jobs' | null {
  if (action === 'sync-all') return 'cluster_album_jobs';
  if (action === 'fungi') return 'kingdom_jobs';
  if (action === 'plantae') return 'plants_jobs';
  return null; // remove is synchronous, no stream
}

// The SSE jobId is keyed by the server's job uuid4 — we pass it
// straight through. For the bulk remove action we don't get a
// job_id, so the slot stays synchronous.

function progressFromSnap(
  action: ActionKey,
  snap: JobSnapshot,
): { added: number; total: number } {
  if (action === 'sync-all') {
    return {
      added: (snap.photos_added as number | undefined) ?? (snap.clusters_done as number | undefined) ?? 0,
      total: (snap.n_assets_total as number | undefined) ?? 0,
    };
  }
  return {
    added: (snap.n_added as number | undefined) ?? 0,
    total: (snap.n_assets as number | undefined) ?? 0,
  };
}

// ---------- main component ------------------------------------------------

export function ImmichSync() {
  const { dispatch } = useApp();
  const stored = useSyncDefaults();
  const push = useToasts();

  // Live SSE snapshot — used to drive progress for the three async
  // actions. We only pull the registries we care about; the rest of the
  // payload is irrelevant to this component.
  const { data: snap } = useEventStream<EventSnapshot>('/api/events');

  // ---- defaults (local mirror of the server state, edited in-place) ----
  // `stored` is the canonical version; `draft` is the in-flight edit.
  // Save copies draft → server, then reloads.
  const [draft, setDraft] = useState<SyncDefaults | null>(stored);
  // Open by default — the section exists to let the user configure
// these defaults, so hiding them behind a disclosure forces an extra
// click before they can do anything useful. CPU-only users in
// particular benefit from seeing the min_score slider up front.
const [defaultsOpen, setDefaultsOpen] = useState(true);
  const [savingDefaults, setSavingDefaults] = useState(false);
  const [defaultsDirty, setDefaultsDirty] = useState(false);
  const [seeded, setSeeded] = useState(false);

  // Load defaults on mount. The server seeds from env on first read;
  // we capture the `seeded` flag so we can show a one-time hint.
  useEffect(() => {
    let cancelled = false;
    getSyncDefaults()
      .then(r => {
        if (cancelled) return;
        dispatch({ type: 'SET_SYNC_DEFAULTS', defaults: r.defaults });
        setDraft(r.defaults);
        setSeeded(r.seeded);
      })
      .catch(e => {
        push({ kind: 'error', message: `Defaults load failed: ${e instanceof Error ? e.message : String(e)}` }, 4000);
      });
    return () => { cancelled = true; };
    // Run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep `draft` in sync with `stored` when it changes externally
  // (e.g. after Save reloads). Don't clobber an in-flight edit.
  useEffect(() => {
    if (stored && !defaultsDirty) setDraft(stored);
  }, [stored, defaultsDirty]);

  const updateSlot = (slot: 'plants' | 'fungi' | 'clusters' | 'sync_all', patch: Partial<{ people_filter: PeopleFilter; min_score: number }>) => {
    setDraft(d => {
      if (!d) return d;
      const next: SyncDefaults = {
        ...d,
        [slot]: { ...d[slot], ...patch },
      };
      setDefaultsDirty(true);
      return next;
    });
  };

  const saveDefaults = async () => {
    if (!draft) return;
    setSavingDefaults(true);
    try {
      const r = await putSyncDefaults(draft);
      dispatch({ type: 'SET_SYNC_DEFAULTS', defaults: r.defaults });
      setDraft(r.defaults);
      setDefaultsDirty(false);
      push({ kind: 'success', message: 'Defaults saved' }, 2500);
    } catch (e) {
      push({ kind: 'error', message: `Save failed: ${e instanceof Error ? e.message : String(e)}` }, 4000);
    } finally {
      setSavingDefaults(false);
    }
  };

  // ---- per-action slots ------------------------------------------------
  const [slots, setSlots] = useState<Record<ActionKey, ActionSlot>>({
    'sync-all': emptySlot(),
    plantae: emptySlot(),
    fungi: emptySlot(),
    remove: emptySlot(),
  });

  // Drive each async slot's progress from the SSE snapshot. We can't
  // do this with useEffect-per-slot because three subscriptions would
  // each re-render the whole component; instead walk the slots here.
  useEffect(() => {
    if (!snap) return;
    setSlots(prev => {
      let changed = false;
      const next = { ...prev };
      (['sync-all', 'plantae', 'fungi'] as ActionKey[]).forEach(action => {
        const slot = next[action];
        if (slot.status !== 'running' || !slot.jobId) return;
        const reg = registryFor(action);
        if (!reg) return;
        const serverJob = snap[reg]?.[slot.jobId];
        if (!serverJob) return;
        const { added, total } = progressFromSnap(action, serverJob);
        if (serverJob.status === 'done') {
          next[action] = {
            ...slot,
            status: 'done',
            added,
            total,
            albumId: (serverJob.album_id as string | null | undefined) ?? null,
            albumUrl: (serverJob.album_url as string | null | undefined) ?? null,
          };
          changed = true;
        } else if (serverJob.status === 'error') {
          next[action] = {
            ...slot,
            status: 'error',
            added,
            total,
            error: (serverJob.error as string | undefined) ?? 'unknown error',
          };
          changed = true;
        } else if (added !== slot.added || total !== slot.total) {
          next[action] = { ...slot, added, total };
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [snap]);

  // ---- confirm modals --------------------------------------------------
  // A single modal re-used for the four actions; `modalFor` selects
  // which action the modal is currently showing. Keeps the JSX small
  // and lets the per-action preview logic live in one place.
  const [modalFor, setModalFor] = useState<ActionKey | null>(null);
  const [overridePeople, setOverridePeople] = useState<PeopleFilter>(null);
  const [overrideScore, setOverrideScore] = useState<number>(0.6);
  const [overrideForce, setOverrideForce] = useState(false);
  const [overrideRemoteDelete, setOverrideRemoteDelete] = useState(false);
  const [previewCount, setPreviewCount] = useState<number | null>(null);
  const [previewing, setPreviewing] = useState(false);

  // Synced counts shown on each button (the unfiltered total — gives
  // the user a sense of scale before clicking). We refresh on mount
  // and on defaults save.
  const [syncedClusterCount, setSyncedClusterCount] = useState<number | null>(null);
  const refreshSyncedCount = async () => {
    try {
      const r = await listClusterAlbums();
      setSyncedClusterCount(r.albums.length);
    } catch {
      // non-fatal — the button still works, just no badge
    }
  };
  useEffect(() => {
    refreshSyncedCount();
  }, []);

  // Reload after a sync-all or remove settles (slot status flip).
  useEffect(() => {
    if (slots['sync-all'].status === 'done' || slots.remove.status === 'done') {
      refreshSyncedCount();
    }
  }, [slots['sync-all'].status, slots.remove.status]);

  const openModal = (action: ActionKey) => {
    if (!draft) return;
    // Pull the per-action default. Branch on action explicitly so the
    // TypeScript narrowing sees only the matching shape — SyncDefaults
    // distinguishes kingdom slots (with min_score) from cluster slots
    // (without).
    let defaultPeople: PeopleFilter = null;
    let defaultScore = 0.6;
    if (action === 'plantae') {
      defaultPeople = (draft.plants.people_filter ?? null) as PeopleFilter;
      defaultScore = draft.plants.min_score ?? 0.6;
    } else if (action === 'fungi') {
      defaultPeople = (draft.fungi.people_filter ?? null) as PeopleFilter;
      defaultScore = draft.fungi.min_score ?? 0.6;
    } else if (action === 'sync-all') {
      defaultPeople = (draft.sync_all.people_filter ?? null) as PeopleFilter;
    } else {
      defaultPeople = (draft.clusters.people_filter ?? null) as PeopleFilter;
    }
    setOverridePeople(defaultPeople);
    setOverrideScore(defaultScore);
    setOverrideForce(false);
    setOverrideRemoteDelete(false);
    setPreviewCount(null);
    setModalFor(action);
    setPreviewing(true);
    // Compute a preview count using the CURRENT (default) overrides so
    // the modal shows a sensible number immediately. The user can then
    // tweak the overrides and we'll re-preview (handled by the
    // useEffect on overridePeople/overrideScore below).
    doPreview(action, defaultPeople,
      action === 'plantae' || action === 'fungi' ? defaultScore : undefined,
    ).catch(() => { /* surfaced in modal as null */ });
  };

  const doPreview = async (action: ActionKey, pf: PeopleFilter, score?: number) => {
    setPreviewing(true);
    try {
      let n = 0;
      if (action === 'sync-all') {
        const r = await previewSyncAllClusters(pf);
        n = r.n_assets;
      } else if (action === 'plantae') {
        const r = await previewPlantsAlbum(score, pf);
        n = r.n_assets;
      } else if (action === 'fungi') {
        const r = await previewKingdomAlbum('Fungi', score, pf);
        n = r.n_assets;
      }
      setPreviewCount(n);
    } catch (e) {
      push({ kind: 'error', message: `Preview failed: ${e instanceof Error ? e.message : String(e)}` }, 3000);
      setPreviewCount(null);
    } finally {
      setPreviewing(false);
    }
  };

  // Re-preview on override change (debounced via setTimeout so a
  // slider drag doesn't fire 30 requests).
  useEffect(() => {
    if (!modalFor || modalFor === 'remove') return;
    const t = setTimeout(() => {
      doPreview(modalFor, overridePeople,
        modalFor === 'plantae' || modalFor === 'fungi' ? overrideScore : undefined,
      );
    }, 200);
    return () => clearTimeout(t);
  }, [modalFor, overridePeople, overrideScore]);

  const closeModal = () => setModalFor(null);

  const confirmAction = async () => {
    if (!modalFor) return;
    const action = modalFor;
    closeModal();
    try {
      if (action === 'sync-all') {
        const r = await startSyncAllClusterAlbums({
          force_recreate: overrideForce,
          people_filter: overridePeople,
        });
        setSlots(s => ({ ...s, 'sync-all': {
          ...emptySlot(),
          jobId: r.job_id,
          status: 'running',
          total: r.n_assets ?? 0,
        }}));
        push({ kind: 'info', message: `Syncing all clusters (${(r.n_assets ?? 0).toLocaleString()} photos)` }, 2500);
      } else if (action === 'plantae') {
        const r = await startPlantsAlbum({
          min_score: overrideScore,
          force_recreate: overrideForce,
          people_filter: overridePeople,
        });
        setSlots(s => ({ ...s, plantae: {
          ...emptySlot(),
          jobId: r.job_id,
          status: 'running',
          total: r.n_assets ?? 0,
        }}));
        push({ kind: 'info', message: `Syncing Plantae photos` }, 2000);
      } else if (action === 'fungi') {
        const r = await startKingdomAlbum('Fungi', {
          min_score: overrideScore,
          force_recreate: overrideForce,
          people_filter: overridePeople,
        });
        setSlots(s => ({ ...s, fungi: {
          ...emptySlot(),
          jobId: r.job_id,
          status: 'running',
          total: r.n_assets ?? 0,
        }}));
        push({ kind: 'info', message: `Syncing Fungi photos` }, 2000);
      } else if (action === 'remove') {
        setSlots(s => ({ ...s, remove: { ...s.remove, status: 'running' } }));
        const r = await removeClusterAlbums({
          cluster_ids: null,
          delete_in_immich: overrideRemoteDelete,
        });
        setSlots(s => ({ ...s, remove: {
          ...emptySlot(),
          status: r.errors.length ? 'error' : 'done',
          added: r.removed,
          total: r.removed,
          error: r.errors.length
            ? `${r.errors.length} cluster(s) failed: ${r.errors.slice(0, 2).map(e => `#${e.cluster_id}`).join(', ')}`
            : '',
        }}));
        if (r.errors.length) {
          push({
            kind: 'error',
            message: `Removed ${r.removed} (${r.errors.length} failed)`,
          }, 4000);
        } else {
          push({ kind: 'success', message: `Removed ${r.removed} cluster album(s)` }, 3000);
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setSlots(s => ({ ...s, [action]: { ...emptySlot(), status: 'error', error: msg } }));
      push({ kind: 'error', message: `${action}: ${msg}` }, 4000);
    }
  };

  // ---- render -----------------------------------------------------------
  // Buttons are disabled while loading defaults OR while their slot is
  // running (to avoid double-clicks). Loading-state is shown inline
  // below each button so the user sees progress.

  const buttonCls = 'px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345] hover:border-accent disabled:opacity-50';

  return (
    <section className="border border-line2 rounded-lg bg-bg-1 p-4 flex flex-col gap-3 w-full">
      <header className="flex items-center justify-between gap-2 flex-wrap">
        <h3 className="m-0 text-base">Sync to IMMICH</h3>
        <button
          type="button"
          className="text-fg-1 hover:text-fg text-sm"
          onClick={() => setDefaultsOpen(o => !o)}
        >
          {defaultsOpen ? '▼' : '▸'} Defaults
          {defaultsDirty && <span className="ml-1 text-yellow-300">●</span>}
        </button>
      </header>

      {seeded && (
        <p className="text-[0.72rem] text-fg-2 m-0">
          Defaults initialised from environment — edit below or just go.
        </p>
      )}

      {defaultsOpen && (
        <div className="flex flex-col gap-2 border-t border-[#222] pt-3">
          {/* Render the panel even before the server returns the
              defaults — placeholder values are overwritten by the
              useEffect on `stored` once it lands. Gating on `draft`
              made the disclosure appear "broken" if the user clicked
              the toggle while the GET was still in flight. */}
          <SlotRow
            label="Plantae"
            showScore
            score={draft?.plants.min_score ?? 0.6}
            people={draft?.plants.people_filter ?? null}
            onScore={v => updateSlot('plants', { min_score: v })}
            onPeople={v => updateSlot('plants', { people_filter: v })}
          />
          <SlotRow
            label="Fungi"
            showScore
            score={draft?.fungi.min_score ?? 0.6}
            people={draft?.fungi.people_filter ?? null}
            onScore={v => updateSlot('fungi', { min_score: v })}
            onPeople={v => updateSlot('fungi', { people_filter: v })}
          />
          <SlotRow
            label="Clusters (per-pill)"
            showScore={false}
            people={draft?.clusters.people_filter ?? null}
            onPeople={v => updateSlot('clusters', { people_filter: v })}
          />
          <SlotRow
            label="Sync clusters (all)"
            showScore={false}
            people={draft?.sync_all.people_filter ?? null}
            onPeople={v => updateSlot('sync_all', { people_filter: v })}
          />
          <div className="flex gap-2 items-center">
            <button
              type="button"
              className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-50"
              onClick={saveDefaults}
              disabled={!defaultsDirty || savingDefaults}
            >
              {savingDefaults ? 'Saving…' : 'Save defaults'}
            </button>
            {defaultsDirty && (
              <button
                type="button"
                className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]"
                onClick={() => { setDraft(stored); setDefaultsDirty(false); }}
              >
                Discard
              </button>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <button type="button" className={buttonCls} onClick={() => openModal('sync-all')} disabled={!draft}>
          Sync clusters (all)
        </button>
        <button type="button" className={buttonCls} onClick={() => openModal('plantae')} disabled={!draft}>
          Sync Plantae photos
        </button>
        <button type="button" className={buttonCls} onClick={() => openModal('fungi')} disabled={!draft}>
          Sync Fungi photos
        </button>
        <button
          type="button"
          className={buttonCls}
          onClick={() => openModal('remove')}
          disabled={!draft || !syncedClusterCount}
          title={!syncedClusterCount ? 'No clusters are currently synced' : 'One-click remove all synced cluster albums'}
        >
          Remove synced clusters{syncedClusterCount != null ? ` (${syncedClusterCount})` : ''}
        </button>
      </div>

      {/* Inline progress rows — one per async action. Remove shows
          its result inline too (synchronous), so we render all four. */}
      <ActionProgress
        label={slots['sync-all'].status === 'done' ? '✓ clusters synced' : 'Sync clusters'}
        slot={slots['sync-all']}
      />
      <ActionProgress
        label={slots.plantae.status === 'done' ? '✓ Plantae synced' : 'Plantae'}
        slot={slots.plantae}
      />
      <ActionProgress
        label={slots.fungi.status === 'done' ? '✓ Fungi synced' : 'Fungi'}
        slot={slots.fungi}
      />
      {slots.remove.status !== 'idle' && (
        <div className="text-[0.8rem]">
          {slots.remove.status === 'running' && <span className="text-fg-1">Removing synced clusters…</span>}
          {slots.remove.status === 'done' && (
            <span className="text-green-300">
              ✓ Removed {slots.remove.added.toLocaleString()} cluster album mapping{slots.remove.added === 1 ? '' : 's'}
            </span>
          )}
          {slots.remove.status === 'error' && (
            <span className="text-red-300">✕ {slots.remove.error}</span>
          )}
        </div>
      )}

      {modalFor && (
        <ConfirmModal
          action={modalFor}
          onClose={closeModal}
          onConfirm={confirmAction}
          overridePeople={overridePeople}
          setOverridePeople={setOverridePeople}
          overrideScore={overrideScore}
          setOverrideScore={setOverrideScore}
          overrideForce={overrideForce}
          setOverrideForce={setOverrideForce}
          overrideRemoteDelete={overrideRemoteDelete}
          setOverrideRemoteDelete={setOverrideRemoteDelete}
          previewCount={previewCount}
          previewing={previewing}
          syncedClusterCount={syncedClusterCount}
        />
      )}
    </section>
  );
}

// ---------- SlotRow: one rule in the Defaults panel ----------------------
// Per the mock: each rule is laid out as two stacked rows.
//   • Top row: rule label on the left, "score ≥" threshold on the right.
//     Only the Plantae / Fungi rules have a score; clusters / sync-all
//     just have the people filter.
//   • Bottom row: full-width segmented chip group (Any / Without people
//     / With people).

function SlotRow({
  label, showScore, score, people, onScore, onPeople,
}: {
  label: string;
  showScore: boolean;
  score?: number;
  people: PeopleFilter;
  onScore?: (v: number) => void;
  onPeople: (v: PeopleFilter) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-fg">{label}</span>
        {showScore && score != null && onScore && (
          <label className="inline-flex items-center gap-1.5">
            <span className="text-[10px] text-fg-1">score ≥</span>
            <input
              type="number"
              className="w-12 bg-bg-2 border border-line2 rounded text-center text-xs py-0.5 text-fg focus:outline-none focus:border-accent font-mono"
              value={score}
              min={0}
              max={1}
              step={0.05}
              onChange={e => onScore(Number(e.target.value))}
            />
          </label>
        )}
      </div>
      <PeopleFilterChips value={people} onChange={onPeople} />
    </div>
  );
}

// ---------- ActionProgress: inline progress bar per action ---------------

function ActionProgress({ label, slot }: { label: string; slot: ActionSlot }) {
  if (slot.status === 'idle') return null;
  return (
    <div className="text-[0.8rem] flex flex-col gap-1 border-t border-[#1f1f1f] pt-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-fg-1">{label}</span>
        {slot.status === 'running' && (
          <span className="font-mono">{slot.added.toLocaleString()} / {slot.total.toLocaleString()}</span>
        )}
        {slot.status === 'done' && slot.albumUrl && (
          <a href={slot.albumUrl} target="_blank" rel="noopener noreferrer" className="text-[#7ad] hover:underline">
            album {slot.albumId?.slice(0, 8)}…
          </a>
        )}
      </div>
      {slot.status === 'running' && (
        <progress value={slot.added} max={Math.max(1, slot.total)} className="w-full h-1.5" />
      )}
      {slot.status === 'error' && <span className="text-red-300">✕ {slot.error}</span>}
    </div>
  );
}

// ---------- ConfirmModal: per-action confirm dialog ---------------------
// Shown when the user clicks any of the four action buttons. Pre-fills
// from the persisted defaults; user can tweak for this single run.

function ConfirmModal(props: {
  action: ActionKey;
  onClose: () => void;
  onConfirm: () => void;
  overridePeople: PeopleFilter;
  setOverridePeople: (v: PeopleFilter) => void;
  overrideScore: number;
  setOverrideScore: (v: number) => void;
  overrideForce: boolean;
  setOverrideForce: (v: boolean) => void;
  overrideRemoteDelete: boolean;
  setOverrideRemoteDelete: (v: boolean) => void;
  previewCount: number | null;
  previewing: boolean;
  syncedClusterCount: number | null;
}) {
  const {
    action, onClose, onConfirm,
    overridePeople, setOverridePeople,
    overrideScore, setOverrideScore,
    overrideForce, setOverrideForce,
    overrideRemoteDelete, setOverrideRemoteDelete,
    previewCount, previewing,
    syncedClusterCount,
  } = props;

  const titles: Record<ActionKey, string> = {
    'sync-all': 'Sync clusters (all)',
    plantae: 'Sync Plantae photos',
    fungi: 'Sync Fungi photos',
    remove: 'Remove synced clusters',
  };

  const showScore = action === 'plantae' || action === 'fungi';
  const isRemove = action === 'remove';

  return (
    <Modal open={true} onClose={onClose} title={titles[action]}>
      <div className="flex flex-col gap-3 text-[0.85rem]">
        {isRemove ? (
          <>
            <p className="m-0">
              {syncedClusterCount == null
                ? 'Counting synced clusters…'
                : `Remove ${syncedClusterCount} cluster→album mapping${syncedClusterCount === 1 ? '' : 's'} from the local DB.`}
            </p>
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={overrideRemoteDelete}
                onChange={e => setOverrideRemoteDelete(e.target.checked)}
                className="accent-accent"
              />
              Also delete the Immich albums (default off — safer)
            </label>
          </>
        ) : (
          <>
            <p className="m-0 text-fg-1">
              {previewing
                ? 'Counting matching photos…'
                : previewCount == null
                  ? 'Preview unavailable — see server response after click.'
                  : `Will tag ${previewCount.toLocaleString()} photo${previewCount === 1 ? '' : 's'}.`}
            </p>
            <div className="flex flex-col gap-1.5">
              <span className="text-fg-2 text-[0.78rem]">Membership</span>
              <PeopleFilterChips value={overridePeople} onChange={setOverridePeople} />
            </div>
            {showScore && (
              <label className="inline-flex items-center gap-1.5">
                <span className="text-fg-2 min-w-[80px]">min score</span>
                <input
                  type="number"
                  className="bg-bg-2 text-fg border border-line2 px-1.5 py-0.5 rounded w-[90px] text-[0.8rem] font-mono focus:outline-none focus:border-accent"
                  value={overrideScore}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={e => setOverrideScore(Number(e.target.value))}
                />
              </label>
            )}
            <label className="inline-flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={overrideForce}
                onChange={e => setOverrideForce(e.target.checked)}
                className="accent-accent"
              />
              Force-recreate the Immich album (ignore saved id)
            </label>
          </>
        )}
        <footer className="flex gap-2 mt-2 justify-end">
          <button
            type="button"
            className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="button"
            className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover"
            onClick={onConfirm}
            disabled={isRemove ? syncedClusterCount == null : previewing}
          >
            {isRemove ? 'Remove all' : 'Sync'}
          </button>
        </footer>
      </div>
    </Modal>
  );
}

// previewSyncAllClusters — wraps /api/album/cluster/sync-all's
// preview. The backend endpoint for sync-all currently doesn't have a
// dedicated preview; we use the per-cluster count sum via the existing
// list_cluster_albums + count_assets_in_cluster round-trips. To keep
// this component dependency-light we just call listClusterAlbums for
// the cluster count badge and skip the deeper preview for sync-all —
// the user will see the actual numbers in the JobPanel anyway.
//
// Implemented here rather than in endpoints.ts to keep the change
// small; if sync-all preview is needed in another caller, promote it.
async function previewSyncAllClusters(_pf: PeopleFilter): Promise<{ n_assets: number }> {
  // The sync-all endpoint has no preview; we approximate by reading
  // list_cluster_albums and summing the stored n_assets. That's not
  // exact (it reflects the LAST sync, not the current picker's
  // people_filter), but it's a reasonable "before" picture. The
  // worker reports the real number as it goes.
  const r = await listClusterAlbums();
  const n = r.albums.reduce((s, a) => s + (a.n_assets ?? 0), 0);
  return { n_assets: n };
}