// Single source of truth for UI state. Server-driven data (results,
// clusters, etc.) is NOT held here — each component fetches its own data
// and the reducer only owns the few pieces of UI state that need to be
// shared across the tree: filters, modal, job panel, active cluster.
//
// We use Context + useReducer (no Zustand/Redux). Selectors via
// useFilters/useJobState/useModalState keep polling re-renders from
// re-rendering the grid.

import {
  createContext,
  useContext,
  useReducer,
  type Dispatch,
  type ReactNode,
} from 'react';
import type { PublicConfig, SyncDefaults } from '../api/types';

// ---- Types ---------------------------------------------------------------

export type GroupBy = 'cluster' | 'species' | 'genus' | 'family';
// `time` = newest captured_at first (EXIF dateTimeOriginal, falling
// back to fileCreatedAt). Photos without captured_at sort last.
// 'rank'  — server-defined order (Immich rank then cluster_id tiebreak)
// 'plantae' — highest Plantae confidence first
// 'score' — highest BioCLIP top-1 species score first
// 'time'  — newest EXIF capture time first (nulls last) — server-side
// 'added' — newest first_seen_at first (nulls last) — server-side
export type SortMode = 'rank' | 'plantae' | 'score' | 'time' | 'added';

export interface Filters {
  group_by: GroupBy;
  sort: SortMode;
  q: string;
  page: number;
  page_size: number;
  // At most one of these is set, dictated by group_by:
  cluster_id?: number | undefined;
  species?: string | undefined;
  genus?: string | undefined;
  family?: string | undefined;
  // Person-filter: top-level facet, independent of group_by.
  // 0 = no person, 1 = has person, undefined = any (and unscanned).
  // Drives /api/results?has_person= and the toolbar chip group.
  has_person?: 0 | 1 | undefined;
  // Plantae-tier facet: independent of group_by, narrows both the
  // filter cards and the photo grid to the same population so counts
  // reconcile. undefined = all tiers.
  plantae?: 'plant' | 'non_plant' | 'uncertain' | undefined;
  // Manual opt-in to surface non-plant photos in the unfiltered
  // all-photo view. Default false — the project-wide "no non-plant
  // counts" policy scopes the grid to Plantae >= 0.35 unless the user
  // flips this on. Has no effect when an explicit `plantae` tier is
  // given; `/api/results?include_non_plants=true` mirrors this flag.
  include_non_plants?: boolean | undefined;
}

export interface ModalState {
  open: boolean;
  // The modal slot only carries the asset detail popup (the photo +
  // library_neighbors + KNN + detections). Pipeline params and server
  // logs live on the #run view, not in a modal.
  kind: 'asset' | null;
  assetId: string | null;
}

export type JobKind = 'plants' | 'fungi' | 'kingdom' | 'cluster' | 'sync-all' | 'purge-missing' | 'person-backfill';

// Up to 5 sample rows from /api/assets/purge-missing, used by the
// confirm modal so the user can spot-check filenames before purging.
export interface PurgeMissingSample {
  asset_id: string;
  original_file_name: string;
  cluster_id: number | null;
}

export interface JobProgress {
  added: number;
  total: number;
  album_id: string | null;
  album_url: string | null;
  reused: boolean;
  // For purge-missing: filename samples returned by the preview.
  sample: PurgeMissingSample[];
}

export type JobPanelState =
  | { kind: 'idle' }
  | {
      kind: Exclude<JobKind, 'idle'>;
      payload: { clusterId?: number; kingdom?: string };
      state: 'confirm' | 'preview-error' | 'running' | 'done' | 'error';
      progress: JobProgress;
      jobId: string | null;
      errorMsg: string;
    };

export interface AppState {
  filters: Filters;
  modal: ModalState;
  job: JobPanelState;
  // When a filter is active, FilterCards collapses by default behind a
  // single active-bar. ClusterAlbumBar reads this so it doesn't
  // duplicate the cluster identity shown in that bar.
  filtersCollapsed: boolean;
  // Sync-to-IMMICH defaults, persisted on the server (album_meta key
  // "sync_defaults"). Loaded on Run-page mount; mutated by the
  // ImmichSync component's "Save defaults" button. Per-action
  // overrides (e.g. "this one time, tag only the with-people ones")
  // live in ImmichSync's local state — they don't go through the
  // reducer because they shouldn't survive a remount.
  syncDefaults: SyncDefaults | null;
  // /api/config, fetched once at App mount and refreshed after the
  // connection form saves. Drives the connection form's prefill and
  // the env-pinned notice. Null until the first fetch resolves.
  config: PublicConfig | null;
  // The "Fix connection" modal (ConnectForm inside a Modal). Opened
  // from HealthBanner's Immich blockers and ImmichSync's connection
  // row; the Welcome wizard embeds ConnectForm directly instead.
  connectionModalOpen: boolean;
  // Multi-select state (UX-17). Set of asset_ids the user has
  // checked via the Card checkbox. Lives in the global reducer so
  // the floating action bar (rendered by Grid) and the cards can
  // both read/write it without prop drilling.
  selection: Set<string>;
}

export type Action =
  | { type: 'SET_FILTERS'; patch: Partial<Filters> }
  | { type: 'OPEN_MODAL'; assetId: string }
  | { type: 'CLOSE_MODAL' }
  | { type: 'SET_FILTERS_COLLAPSED'; collapsed: boolean }
  | { type: 'OPEN_JOB'; kind: Exclude<JobKind, 'idle'>; payload: { clusterId?: number; kingdom?: string } }
  | { type: 'JOB_PREVIEW_OK'; total: number; sample?: PurgeMissingSample[] }
  | { type: 'JOB_PREVIEW_ERR'; errorMsg: string }
  | { type: 'JOB_STARTED'; jobId: string; total: number }
  | { type: 'JOB_PROGRESS'; added: number; total: number }
  | { type: 'JOB_DONE'; added: number; album_id: string | null; album_url: string | null; reused: boolean }
  | { type: 'JOB_ERROR'; errorMsg: string }
  | { type: 'JOB_HIDE' }
  | { type: 'SET_SYNC_DEFAULTS'; defaults: SyncDefaults | null }
  | { type: 'SET_CONFIG'; config: PublicConfig | null }
  | { type: 'OPEN_CONNECTION_MODAL' }
  | { type: 'CLOSE_CONNECTION_MODAL' }
  | { type: 'TOGGLE_SELECT'; assetId: string }
  | { type: 'SELECT_RANGE'; assetIds: string[]; selected: boolean }
  | { type: 'CLEAR_SELECTION' };

// ---- Initial state -------------------------------------------------------

export const DEFAULT_FILTERS: Filters = {
  group_by: 'cluster',
  // Browse-list default is "Newest first" (capture date DESC). Photos
  // without EXIF captured_at fall to the end of the page, but they
  // still appear. Reason: on a personal photo library the user almost
  // always wants to start with their most-recent shot — Immich's
  // "search rank" was a confusing default for that. Stay in lock-step
  // with the SORT_OPTIONS pill order (Grid.tsx).
  sort: 'time',
  q: '',
  page: 0,
  page_size: 60,
};

function defaultJob(): JobPanelState {
  return { kind: 'idle' };
}

const INITIAL: AppState = {
  filters: DEFAULT_FILTERS,
  modal: { open: false, kind: null, assetId: null },
  job: defaultJob(),
  filtersCollapsed: false,
  syncDefaults: null,
  config: null,
  connectionModalOpen: false,
  selection: new Set<string>(),
};

// ---- Reducer -------------------------------------------------------------

function clearTaxonomyFilter(f: Filters): Partial<Filters> {
  return {
    cluster_id: undefined,
    species: undefined,
    genus: undefined,
    family: undefined,
  };
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_FILTERS': {
      const next = { ...state.filters, ...action.patch };
      // Switching group_by must atomically clear the stale taxonomy filter
      // and reset the page. The hook handles the URL push separately.
      if (action.patch.group_by && action.patch.group_by !== state.filters.group_by) {
        Object.assign(next, clearTaxonomyFilter(state.filters), { group_by: action.patch.group_by, page: 0 });
      }
      return { ...state, filters: next };
    }
    case 'OPEN_MODAL':
      return { ...state, modal: { open: true, kind: 'asset', assetId: action.assetId } };
    case 'CLOSE_MODAL':
      return { ...state, modal: { open: false, kind: null, assetId: null } };
    case 'SET_FILTERS_COLLAPSED':
      return { ...state, filtersCollapsed: action.collapsed };
    case 'OPEN_JOB':
      return {
        ...state,
        job: {
          kind: action.kind,
          payload: action.payload,
          state: 'confirm',
          progress: { added: 0, total: 0, album_id: null, album_url: null, reused: false, sample: [] },
          jobId: null,
          errorMsg: '',
        },
      };
    case 'JOB_PREVIEW_OK':
      if (state.job.kind === 'idle') return state;
      return {
        ...state,
        job: {
          ...state.job,
          state: 'confirm',
          progress: {
            ...state.job.progress,
            total: action.total,
            sample: action.sample ?? state.job.progress.sample,
          },
        },
      };
    case 'JOB_PREVIEW_ERR':
      if (state.job.kind === 'idle') return state;
      return { ...state, job: { ...state.job, state: 'preview-error', errorMsg: action.errorMsg } };
    case 'JOB_STARTED':
      if (state.job.kind === 'idle') return state;
      return { ...state, job: { ...state.job, state: 'running', jobId: action.jobId, progress: { ...state.job.progress, total: action.total } } };
    case 'JOB_PROGRESS':
      if (state.job.kind === 'idle') return state;
      return { ...state, job: { ...state.job, progress: { ...state.job.progress, added: action.added, total: action.total } } };
    case 'JOB_DONE':
      if (state.job.kind === 'idle') return state;
      return {
        ...state,
        job: {
          ...state.job,
          state: 'done',
          progress: { ...state.job.progress, added: action.added, album_id: action.album_id, album_url: action.album_url, reused: action.reused },
        },
      };
    case 'JOB_ERROR':
      if (state.job.kind === 'idle') return state;
      return { ...state, job: { ...state.job, state: 'error', errorMsg: action.errorMsg } };
    case 'JOB_HIDE':
      return { ...state, job: defaultJob() };
    case 'SET_SYNC_DEFAULTS':
      return { ...state, syncDefaults: action.defaults };
    case 'SET_CONFIG':
      return { ...state, config: action.config };
    case 'OPEN_CONNECTION_MODAL':
      return { ...state, connectionModalOpen: true };
    case 'CLOSE_CONNECTION_MODAL':
      return { ...state, connectionModalOpen: false };
    case 'TOGGLE_SELECT': {
      const next = new Set(state.selection);
      if (next.has(action.assetId)) next.delete(action.assetId);
      else next.add(action.assetId);
      return { ...state, selection: next };
    }
    case 'SELECT_RANGE': {
      const next = new Set(state.selection);
      for (const id of action.assetIds) {
        if (action.selected) next.add(id);
        else next.delete(id);
      }
      return { ...state, selection: next };
    }
    case 'CLEAR_SELECTION':
      if (state.selection.size === 0) return state;
      return { ...state, selection: new Set() };
    default:
      return state;
  }
}

// ---- Context + hook ------------------------------------------------------

interface Ctx {
  state: AppState;
  dispatch: Dispatch<Action>;
}

const AppCtx = createContext<Ctx | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  return <AppCtx.Provider value={{ state, dispatch }}>{children}</AppCtx.Provider>;
}

export function useApp(): Ctx {
  const ctx = useContext(AppCtx);
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>');
  return ctx;
}

// Narrow selector hooks — components subscribe to one slice so the polling
// reducer doesn't re-render the grid.
export function useFilters() {
  return useApp().state.filters;
}
export function useModal() {
  return useApp().state.modal;
}
export function useJob() {
  return useApp().state.job;
}
export function useSyncDefaults() {
  return useApp().state.syncDefaults;
}
export function useConfig() {
  return useApp().state.config;
}
export function useConnectionModal() {
  return useApp().state.connectionModalOpen;
}

export function useSelection() {
  return useApp().state.selection;
}