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

// ---- Types ---------------------------------------------------------------

export type GroupBy = 'cluster' | 'species' | 'genus' | 'family';
export type SortMode = 'rank' | 'plantae' | 'score';

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
}

export interface ModalState {
  open: boolean;
  // Which dialog is currently mounted. `asset` is the photo detail;
  // `params` and `logs` are the new modal-launched forms. Only one is
  // open at a time (a single <Modal> slot is reused).
  kind: 'asset' | 'params' | 'logs' | null;
  assetId: string | null;
}

export type JobKind = 'plants' | 'fungi' | 'kingdom' | 'cluster' | 'sync-all';

export interface JobProgress {
  added: number;
  total: number;
  album_id: string | null;
  album_url: string | null;
  reused: boolean;
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
}

export type Action =
  | { type: 'SET_FILTERS'; patch: Partial<Filters> }
  | { type: 'OPEN_MODAL'; assetId: string }
  | { type: 'OPEN_PARAMS_MODAL' }
  | { type: 'OPEN_LOGS_MODAL' }
  | { type: 'CLOSE_MODAL' }
  | { type: 'OPEN_JOB'; kind: Exclude<JobKind, 'idle'>; payload: { clusterId?: number; kingdom?: string } }
  | { type: 'JOB_PREVIEW_OK'; total: number }
  | { type: 'JOB_PREVIEW_ERR'; errorMsg: string }
  | { type: 'JOB_STARTED'; jobId: string; total: number }
  | { type: 'JOB_PROGRESS'; added: number; total: number }
  | { type: 'JOB_DONE'; added: number; album_id: string | null; album_url: string | null; reused: boolean }
  | { type: 'JOB_ERROR'; errorMsg: string }
  | { type: 'JOB_HIDE' };

// ---- Initial state -------------------------------------------------------

export const DEFAULT_FILTERS: Filters = {
  group_by: 'cluster',
  sort: 'rank',
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
    case 'OPEN_PARAMS_MODAL':
      return { ...state, modal: { open: true, kind: 'params', assetId: null } };
    case 'OPEN_LOGS_MODAL':
      return { ...state, modal: { open: true, kind: 'logs', assetId: null } };
    case 'CLOSE_MODAL':
      return { ...state, modal: { open: false, kind: null, assetId: null } };
    case 'OPEN_JOB':
      return {
        ...state,
        job: {
          kind: action.kind,
          payload: action.payload,
          state: 'confirm',
          progress: { added: 0, total: 0, album_id: null, album_url: null, reused: false },
          jobId: null,
          errorMsg: '',
        },
      };
    case 'JOB_PREVIEW_OK':
      if (state.job.kind === 'idle') return state;
      return { ...state, job: { ...state.job, state: 'confirm', progress: { ...state.job.progress, total: action.total } } };
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