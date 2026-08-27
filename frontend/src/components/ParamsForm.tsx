// Run-page "Pipeline" form. The four pipeline params (query,
// n_samples, top_k, batch_size) are collapsed by default — most users
// just want "run with server defaults". Unfolding the disclosure
// reveals the fields; the Run button always works and uses server
// defaults for any field left blank.
//
// The disclosure state lives in the parent (App.tsx Run section) so
// the toggle button can sit in the outer Pipeline card header —
// matching the mock's card layout where the "Edit params" button is
// in the header next to the title.
//
// The Run button's visual state tracks the actual pipeline run via
// the SSE stream — when /api/events reports status="running" the
// button shows "Running…" and is disabled; when the run finishes
// (status="done") it reverts to "Run pipeline". Tracking a local
// `busy` boolean during the click was useless because /api/run
// returns immediately (the child runs in the background).

import { useEffect, useRef, useState } from 'react';
import { runPipeline } from '../api/endpoints';
import { useEventStream } from '../hooks/useEventStream';
import type { EventSnapshot } from '../api/types';
import { useToasts } from './ToastProvider';

interface FieldHelp {
  label: string;
  help: string;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
}

const FIELDS: Record<string, FieldHelp> = {
  query: {
    label: 'query',
    help: 'Free-text search phrase sent to Immich to pick the candidate photos. Leave blank to use the server default (the IMMICH_QUERY env var).',
    placeholder: 'intentional plant photos',
  },
  n_samples: {
    label: 'photos to fetch',
    help: 'How many candidate photos to pull from Immich for this run. Higher = more to classify, longer runtime. No hard cap — the server processes whatever you give it (default 10000).',
    placeholder: '10000',
    min: 1,
  },
  top_k: {
    label: 'predictions per photo',
    help: 'How many species-rank predictions to keep per photo. Shown in each card. 5 is plenty for the UI; raising it just stores more rows.',
    placeholder: '5',
    min: 1,
    max: 20,
  },
  batch_size: {
    label: 'batch',
    help: 'How many images BioCLIP processes at once. Most users run on CPU — 16 is a sensible default (CPUs plateau around 16-32). Lower if you are tight on RAM; GPU users can push higher (32-64) until VRAM fills.',
    placeholder: '16',
    min: 1,
    max: 64,
  },
};

interface Props {
  /** When true, render the full set of param fields. When false, just
   *  show the Run button (uses server defaults). The parent owns the
   *  toggle so the toggle button can sit in the outer card header. */
  paramsOpen: boolean;
}

export function ParamsForm({ paramsOpen }: Props) {
  const [query, setQuery] = useState('');
  const [n, setN] = useState('');
  const [topK, setTopK] = useState('');
  const [batch, setBatch] = useState('');
  // The three "advanced" params are tucked behind a separate
  // disclosure so the most-used knob (`query`) gets the visual
  // weight. Session-only state — no need to persist.
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Subscribe to the SSE pipeline status so the button reflects the
  // actual run state, not just "click is being processed".
  const { data: snapshot } = useEventStream<EventSnapshot>('/api/events');
  const running = snapshot?.pipeline?.status === 'running';
  // `awaiting` covers the window from click → SSE confirms a terminal
  // state. It is NOT cleared when the POST returns, because there's a
  // brief gap between /api/run returning and the server's first SSE
  // push. Without this, the button would flicker back to "Run pipeline"
  // before the user sees "Running…". Cleared in three places:
  //   1. handleRun's catch block — POST failed, let the user retry.
  //   2. The useEffect below — SSE reports done/error/stopped.
  //   3. (Implicit) when the SSE's `running` flag flips on, `busy` is
  //      already true via `running`; we don't need to clear awaiting.
  const [awaiting, setAwaiting] = useState(false);
  const busy = running || awaiting;
  // Synchronous "busy" guard — `awaiting` is React state and only
  // applies to the next render. Between click and re-render the
  // button is still enabled in the DOM, so a rapid second click
  // (or pressing Enter inside an input field, which submits the
  // form even when the submit button is disabled) would re-enter
  // handleRun and fire /api/run twice. `busyRef` is a ref — set
  // synchronously on click — so the second invocation early-returns
  // before the second POST goes out.
  const busyRef = useRef(false);
  const push = useToasts();

  // Clear `awaiting` (and the ref) once the server reports a
  // terminal pipeline state. We don't react to "idle" — that's also
  // the initial state, and a momentary SSE reconnect could surface
  // "idle" right before the run actually starts.
  useEffect(() => {
    if (!awaiting) return;
    const status = snapshot?.pipeline?.status;
    if (status === 'done' || status === 'error' || status === 'stopped') {
      setAwaiting(false);
      busyRef.current = false;
    }
  }, [snapshot, awaiting]);

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busyRef.current) return;
    busyRef.current = true;
    setAwaiting(true);
    try {
      const resp = await runPipeline({
        query: query.trim() || null,
        n_samples: n ? Number(n) : null,
        top_k: topK ? Number(topK) : null,
        batch_size: batch ? Number(batch) : null,
      });
      const jobId = (resp as { job_id?: string })?.job_id;
      push(
        { kind: 'success', message: `Pipeline started (job ${jobId ?? '?'})` },
        2500,
      );
      // Note: we deliberately don't `setAwaiting(false)` here. The SSE
      // will tell us when the run actually finishes (done/error/stopped),
      // at which point the useEffect above clears awaiting.
    } catch (err) {
      setAwaiting(false);
      busyRef.current = false;
      push(
        { kind: 'info', message: err instanceof Error ? err.message : String(err) },
        4000,
      );
    }
  };

  const inputCls =
    'bg-bg-2 text-fg border border-line2 px-2.5 py-[5px] rounded text-[0.85rem] font-mono focus:outline-none focus:border-accent';

  return (
    <form onSubmit={handleRun} className="flex flex-col gap-3 w-full">
      {/* Single Run button — uses server defaults unless the user has
          unfolded the params and edited a field. Blank fields = server
          default, so the collapsed state is fully functional. The
          button's text/disabled state tracks the actual run via SSE so
          the user gets immediate feedback that the click registered. */}
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={busy}
          className="run-pipeline-btn"
        >
          <PlayIcon />
          {busy ? 'Running…' : 'Run Pipeline'}
        </button>
      </div>

      {paramsOpen && (
        <fieldset role="group" className="flex flex-col gap-3 border-0 p-0 m-0">
          {/* `query` is the most consequential knob — visible
              whenever params is open, with a one-line helper to nudge
              new users toward concrete examples. */}
          <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px]">
            {FIELDS.query?.label}
            <input
              type="text"
              className={`${inputCls} min-w-[220px] w-full`}
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder={FIELDS.query?.placeholder}
              title={FIELDS.query?.help}
            />
            <small className="text-fg-2 mt-0.5 text-[0.72rem] leading-[1.4]">
              What to ask Immich's smart-search. Try{' '}
              <em>"red flowers"</em>, <em>"moss on rocks"</em>, or{' '}
              <em>"tree bark close-up"</em>.
            </small>
          </label>

          {/* The other three knobs are tucked behind an Advanced
              disclosure — most users just want "run with server
              defaults". */}
          <details
            open={advancedOpen}
            onToggle={e => setAdvancedOpen((e.target as HTMLDetailsElement).open)}
            className="text-fg-1 text-[0.8rem]"
          >
            <summary className="cursor-pointer text-fg-1 hover:text-fg text-[0.8rem] py-1">
              Advanced (batch, top-K, sample count)
            </summary>
            <div className="flex flex-wrap items-start gap-x-5 gap-y-3 pt-2">
              <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[180px] flex-[1_1_180px]">
                {FIELDS.n_samples?.label}
                <input
                  type="number"
                  className={`${inputCls} w-[110px]`}
                  value={n}
                  onChange={e => setN(e.target.value)}
                  min={FIELDS.n_samples?.min}
                  placeholder={FIELDS.n_samples?.placeholder}
                  title={FIELDS.n_samples?.help}
                />
              </label>

              <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[180px] flex-[1_1_180px]">
                {FIELDS.top_k?.label}
                <input
                  type="number"
                  className={`${inputCls} w-[110px]`}
                  value={topK}
                  onChange={e => setTopK(e.target.value)}
                  min={FIELDS.top_k?.min}
                  max={FIELDS.top_k?.max}
                  placeholder={FIELDS.top_k?.placeholder}
                  title={FIELDS.top_k?.help}
                />
              </label>

              <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[180px] flex-[1_1_180px]">
                {FIELDS.batch_size?.label}
                <input
                  type="number"
                  className={`${inputCls} w-[110px]`}
                  value={batch}
                  onChange={e => setBatch(e.target.value)}
                  min={FIELDS.batch_size?.min}
                  max={FIELDS.batch_size?.max}
                  placeholder={FIELDS.batch_size?.placeholder}
                  title={FIELDS.batch_size?.help}
                />
              </label>
            </div>
          </details>
        </fieldset>
      )}
    </form>
  );
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <polygon points="6 4 20 12 6 20" />
    </svg>
  );
}