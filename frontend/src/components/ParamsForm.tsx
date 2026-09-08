// Run-page "Pipeline" form.
//
// Layout (when the outer "Edit params" toggle is open):
//   • Run button — always visible.
//   • Stop-criteria threshold — the main knob. Always visible when
//     params are open. Lower % = the search stops sooner (stricter
//     plant/fungi density requirement per batch), so the run gathers
//     fewer photos; higher % = the search keeps going through more
//     pages of mixed content.
//   • Query disclosure — folded by default. Inside:
//       - One-line suggestion chips ("intentional flower photos",
//         "moss on rocks", "tree bark close-up"). Click to fill the
//         query input; the chip vanishes from the row once that text
//         is in the box.
//       - Query text input.
//       - "Photos to fetch" — the Immich candidate-pool cap. Default
//         20k (UI placeholder; the server's env-configured default
//         is unchanged unless the user types a value).
//   • Advanced disclosure — only the two technical knobs most users
//     never touch (top_k, batch_size).
//
// The Run button always works and uses server defaults for any field
// left blank.
//
// The disclosure state (`paramsOpen`) lives in the parent (App.tsx
// Run section) so the toggle button can sit in the outer Pipeline
// card header.
//
// The Run button's visual state tracks the actual pipeline run via
// the SSE stream — when /api/events reports status="running" the
// button shows "Running…" and is disabled; when the run finishes
// (status="done") it reverts to "Run Pipeline". Tracking a local
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
  // UI suffix shown after the input (e.g. "%").
  suffix?: string;
}

const FIELDS: Record<string, FieldHelp> = {
  // Main knob — exposed at the top of the params panel, NOT inside a
  // disclosure. Expressed as a percent (0-100).
  stop_criteria_pct: {
    label: 'stop-criteria threshold',
    help: "Percent of each batch that must look like plants-or-fungi for the search to keep going. Lower = stricter (search stops sooner, fewer photos gathered). 30% is a sensible default — drop to 10-15% if your library is mostly non-plants and you want the run to terminate fast; raise to 50%+ to keep pulling even sparse batches.",
    placeholder: '30',
    min: 0,
    max: 100,
    step: 5,
    suffix: '%',
  },
  query: {
    label: 'query',
    help: "Free-text search phrase sent to Immich to pick the candidate photos. Leave blank to use the server default (the QUERY env var, typically 'intentional plant photos').",
    placeholder: 'intentional plant photos',
  },
  n_samples: {
    label: 'photos to fetch',
    help: 'How many candidate photos to pull from Immich for this run. Higher = more to classify, longer runtime. The search also stops earlier if it hits the stop-criteria boundary, so this is a soft cap rather than a hard target.',
    placeholder: '20000',
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

// One-click query presets. Clicking a chip fills the query input
// (overwriting whatever's there); once the input matches one of
// the chips exactly, that chip is hidden so the row doesn't
// duplicate. Keep these short — they're buttons, not paragraphs.
const QUERY_SUGGESTIONS = [
  'intentional flower photos',
  'moss on rocks',
  'tree bark close-up',
];

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
  // The new "main" knob lives at the top of the panel; the query +
  // photos-to-fetch pair sits behind a folded disclosure so a
  // single click on "Run Pipeline" with default everything still
  // works.
  const [stopCriteria, setStopCriteria] = useState('');
  const [queryOpen, setQueryOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Subscribe to the SSE pipeline status so the button reflects the
  // actual run state, not just "click is being processed".
  const { data: snapshot } = useEventStream<EventSnapshot>('/api/events');
  const running = snapshot?.pipeline?.status === 'running';
  // `awaiting` covers the window from click → the SSE confirms the run
  // is live. It is NOT cleared when the POST returns (there's a gap
  // between /api/run returning and the server's first SSE push) and it
  // is NOT cleared by just any terminal status: the snapshot the SSE
  // already holds at click time still shows the PREVIOUS run's
  // "done", and letting that stale value clear `awaiting` re-enabled
  // the button seconds before the new run ever showed up. So a
  // terminal status only counts once the SSE has reported "running"
  // for this click (`runConfirmedRef`). Cleared in:
  //   1. handleRun's catch block — POST failed, let the user retry.
  //   2. The useEffect below — SSE confirms "running" (busy visuals
  //      continue via `running`), then releases on done/error/stopped.
  //   3. The backstop timer in handleRun — SSE never confirmed (child
  //      died at boot), so don't latch the button forever.
  const [awaiting, setAwaiting] = useState(false);
  const busy = running || awaiting;
  // Synchronous "busy" guard — `awaiting` is React state and only
  // applies to the next render. Between click and re-render the
  // button is still enabled in the DOM, so a rapid second click
  // (or pressing Enter inside an input field, which submits the
  // form even when the submit button is disabled) would re-enter
  // handleRun and fire /api/run twice. `busyRef` is a ref — set
  // synchronously on click — so the second invocation early-returns
  // before the second POST goes out. It stays latched until the
  // confirmed run reaches a terminal state, so Enter still can't
  // double-submit mid-run.
  const busyRef = useRef(false);
  // Set once the SSE has shown "running" since the click. Only a
  // terminal status seen AFTER this latch releases the button.
  const runConfirmedRef = useRef(false);
  // Backstop timer armed after a successful POST (see handleRun).
  const releaseTimerRef = useRef<number | null>(null);
  const push = useToasts();

  useEffect(() => () => {
    // Unmount: don't leave the backstop timer firing post-unmount.
    if (releaseTimerRef.current !== null) window.clearTimeout(releaseTimerRef.current);
  }, []);

  useEffect(() => {
    if (!awaiting && !busyRef.current) return;
    const status = snapshot?.pipeline?.status;
    if (status === 'running') {
      // Our run is live — `running` owns the busy visuals from here,
      // so awaiting can drop. busyRef stays latched (Enter-in-input
      // submits even with the button disabled) until the run ends.
      runConfirmedRef.current = true;
      if (releaseTimerRef.current !== null) {
        window.clearTimeout(releaseTimerRef.current);
        releaseTimerRef.current = null;
      }
      setAwaiting(false);
      return;
    }
    if (
      !runConfirmedRef.current ||
      (status !== 'done' && status !== 'error' && status !== 'stopped')
    ) {
      // Either a pre-run snapshot value (idle / the previous run's
      // "done") or still mid-run — keep the button latched.
      return;
    }
    runConfirmedRef.current = false;
    setAwaiting(false);
    busyRef.current = false;
    if (releaseTimerRef.current !== null) {
      window.clearTimeout(releaseTimerRef.current);
      releaseTimerRef.current = null;
    }
  }, [snapshot, awaiting]);

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busyRef.current) return;
    busyRef.current = true;
    runConfirmedRef.current = false;
    setAwaiting(true);
    try {
      const resp = await runPipeline({
        query: query.trim() || null,
        n_samples: n ? Number(n) : null,
        top_k: topK ? Number(topK) : null,
        batch_size: batch ? Number(batch) : null,
        stop_criteria_pct: stopCriteria === '' ? null : Number(stopCriteria),
      });
      const jobId = (resp as { job_id?: string })?.job_id;
      push(
        { kind: 'success', message: `Pipeline started (job ${jobId ?? '?'})` },
        2500,
      );
      // Note: we deliberately don't `setAwaiting(false)` here — the
      // useEffect above clears it when the SSE confirms the run. The
      // timer below is only the backstop for the case where the SSE
      // NEVER confirms (child died before its first status write, or
      // the SSE connection is gone): release the button rather than
      // latching the user out of the form forever.
      releaseTimerRef.current = window.setTimeout(() => {
        releaseTimerRef.current = null;
        if (runConfirmedRef.current) return; // effect owns the release
        setAwaiting(false);
        busyRef.current = false;
      }, 30_000);
    } catch (err) {
      setAwaiting(false);
      busyRef.current = false;
      runConfirmedRef.current = false;
      push(
        { kind: 'info', message: err instanceof Error ? err.message : String(err) },
        4000,
      );
    }
  };

  const inputCls =
    'bg-bg-2 text-fg border border-line2 px-2.5 py-[5px] rounded text-[0.85rem] font-mono focus:outline-none focus:border-accent';

  // Hide a suggestion chip once the user has the same text in the
  // query box (case-insensitive, trimmed). Avoids the "I just
  // clicked 'flower' and the chip is still there" feedback loop.
  const visibleSuggestions = QUERY_SUGGESTIONS.filter(
    s => s.toLowerCase().trim() !== query.toLowerCase().trim(),
  );

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
          {/* Stop-criteria threshold — the main knob. Always visible
              when params are open (not inside any disclosure) so the
              most consequential per-run decision sits at the top.
              A 0-100 number input with a % suffix; placeholder shows
              the server default (30%) so blank = "use default". */}
          <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px]">
            <span className="flex items-center gap-1.5">
              {FIELDS.stop_criteria_pct?.label}
              <HintBadge text={FIELDS.stop_criteria_pct?.help ?? ''} />
            </span>
            <span className="flex items-center gap-1.5">
              <input
                type="number"
                className={`${inputCls} w-[110px]`}
                value={stopCriteria}
                onChange={e => setStopCriteria(e.target.value)}
                min={FIELDS.stop_criteria_pct?.min}
                max={FIELDS.stop_criteria_pct?.max}
                step={FIELDS.stop_criteria_pct?.step}
                placeholder={FIELDS.stop_criteria_pct?.placeholder}
                title={FIELDS.stop_criteria_pct?.help}
              />
              <span className="text-fg-2 text-[0.8rem] font-mono">
                {FIELDS.stop_criteria_pct?.suffix}
              </span>
            </span>
            <small className="text-fg-2 mt-0.5 text-[0.72rem] leading-[1.4]">
              How strict the plant-density check is before the search
              stops. Lower = the run terminates sooner with fewer
              photos; higher = it keeps crawling through mixed batches.
            </small>
          </label>

          {/* Query disclosure — folded by default. New users hit
              "Run Pipeline" with everything blank and get sensible
              server defaults; power users open this to set the
              Immich search phrase and the candidate-pool cap
              together. Photos-to-fetch moved INTO this block because
              it pairs naturally with the query (both shape what
              Immich returns). */}
          <details
            open={queryOpen}
            onToggle={e => setQueryOpen((e.target as HTMLDetailsElement).open)}
            className="text-fg-1 text-[0.8rem]"
          >
            <summary className="cursor-pointer text-fg-1 hover:text-fg text-[0.8rem] py-1">
              Query &amp; photos to fetch
            </summary>
            <div className="flex flex-col gap-3 pt-2">
              {/* Suggestion chips — one click fills the query.
                  Hide a chip once its text matches the current
                  query. Empty list = no extra UI shown (don't render
                  an empty row). */}
              {visibleSuggestions.length > 0 && (
                <div className="flex flex-col gap-[3px]">
                  <span className="text-fg-1 text-[0.8rem]">
                    query suggestions
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {visibleSuggestions.map(s => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => setQuery(s)}
                        title={`Use "${s}" as the query`}
                        className="text-fg-1 bg-bg-2 hover:bg-bg-3 border border-line2 hover:border-accent px-2 py-[3px] rounded-full text-[0.75rem] font-mono transition-colors"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

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
                  <em>"red flowers"</em>, <em>"ferns in shade"</em>, or{' '}
                  <em>"succulents on a windowsill"</em>.
                </small>
              </label>

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
            </div>
          </details>

          {/* The truly technical knobs — tucked behind their own
              disclosure so the panel doesn't scare off casual users. */}
          <details
            open={advancedOpen}
            onToggle={e => setAdvancedOpen((e.target as HTMLDetailsElement).open)}
            className="text-fg-1 text-[0.8rem]"
          >
            <summary className="cursor-pointer text-fg-1 hover:text-fg text-[0.8rem] py-1">
              Advanced (batch, top-K)
            </summary>
            <div className="flex flex-wrap items-start gap-x-5 gap-y-3 pt-2">
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

// Tiny "?" badge that opens the field's full help text on hover/focus
// (title attribute). Used inline next to the stop-criteria label so
// the long description stays one click away without bloating the
// visible row. Re-using the browser-native tooltip keeps the UI
// consistent with the rest of the form (every other field already
// uses `title=` for the same help string).
function HintBadge({ text }: { text: string }) {
  return (
    <span
      role="img"
      aria-label="help"
      title={text}
      className="inline-flex items-center justify-center w-[14px] h-[14px] rounded-full border border-line2 text-fg-2 text-[0.65rem] leading-none cursor-help"
      style={{ fontFamily: 'inherit' }}
    >
      ?
    </span>
  );
}
