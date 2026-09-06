import { useEffect, useRef, useState } from 'react';
import { getHealth, getResults, getStatus } from '../api/endpoints';
import { navigate } from '../hooks/useRoute';
import { useEventStream } from '../hooks/useEventStream';
import type { EventSnapshot } from '../api/types';
import { ConnectForm } from './ConnectForm';
import { StatusPanel } from './StatusPanel';

// Onboarding wizard shown on the browse-list view when the pipeline has
// not yet produced any data. Two steps — Connect → First run — with a
// server-state-driven visibility rule:
//
//   - Loads invisible until both API calls (status + results) resolve
//     (no flicker of wizard while data is still in flight).
//   - Hides itself once the pipeline has produced at least one asset.
//     Server-state-driven so it works the same way on every port,
//     every browser, every device — no per-origin localStorage trap.
//   - While a pipeline run is ACTIVE this whole wizard is replaced by
//     the live run status — "welcome, connect, run" is the wrong thing
//     to show someone whose run is already chewing through their
//     library.
//
// The wizard's step state is component-local — it resets each session,
// which is fine because we're only using it for the in-progress
// walk-through cursor.
//
// Steps:
//   1. Connect Immich — the connection form (URL + API key), saved via
//      POST /api/settings/immich and live-checked. No config files.
//   2. First run — a CTA that jumps to the Run page. The seed query
//      lives there (ParamsForm), so no separate step for it.

type WizardStep = 'connect' | 'run';

const STEP_ORDER: WizardStep[] = ['connect', 'run'];

export function Welcome() {
  const [step, setStep] = useState<WizardStep>('connect');
  const [hasRuns, setHasRuns] = useState<boolean | null>(null);
  const [totalAssets, setTotalAssets] = useState<number | null>(null);
  // null = still checking; true once /api/health reports the connection
  // works (either pre-configured via env, or just saved via the form).
  const [connected, setConnected] = useState<boolean | null>(null);
  // Mount-time run state; the SSE snapshot (shared singleton) overrides
  // it live, so a run started elsewhere flips this card to run status
  // without a remount.
  const [statusAtMount, setStatusAtMount] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const { data: events } = useEventStream<EventSnapshot>('/api/events');

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getStatus().catch(() => null),
      getResults({ page: 0, page_size: 1 }).catch(() => null),
      getHealth().catch(() => null),
    ]).then(([status, resultsResp, health]) => {
      if (cancelled) return;
      const ranSomething = status != null && (
        Object.keys(status.stage_times ?? {}).length > 0 ||
        status.status === 'running'
      );
      setHasRuns(ranSomething);
      setTotalAssets(resultsResp?.pagination?.total ?? 0);
      setStatusAtMount(status?.status ?? null);
      setConnected(
        health?.immich?.ok === true && health?.api_key?.ok === true,
      );
    });
    return () => { cancelled = true; };
  }, [reloadKey]);

  const liveStatus = events?.pipeline?.status;
  const running = (liveStatus ?? statusAtMount) === 'running';
  const wasRunning = useRef(running);
  useEffect(() => {
    // Run just finished while we were watching → refetch once so the
    // wizard re-hides if the run produced data (totalAssets > 0).
    if (wasRunning.current && !running) setReloadKey(k => k + 1);
    wasRunning.current = running;
  }, [running]);

  // A run in progress IS the home-page message — replaces the wizard.
  if (running) {
    return (
      <section
        className="m-4 sm:m-6 p-6 sm:p-8 rounded-xl bg-bg-1 border border-line2 max-w-[720px]"
        aria-label="Run in progress"
      >
        <h2 className="text-xl sm:text-2xl m-0 mb-3">Pipeline running</h2>
        <StatusPanel />
        <p className="text-fg-2 text-[11.5px] m-0 mt-3">
          Results appear as the run commits them — the Browse grid fills
          in batch by batch, no need to wait for the whole run.
        </p>
      </section>
    );
  }

  // Loading state — keep the wizard hidden until both API calls resolve,
  // so users don't see a one-frame flash of "Welcome to BotaniSense" on
  // every page load.
  if (hasRuns === null || totalAssets === null) return null;

  // Once the pipeline has produced assets, the wizard is done forever —
  // same on every port, browser, device, regardless of any localStorage
  // whether they've "clicked through" the steps before.
  if (hasRuns && totalAssets > 0) return null;

  const advance = (next: WizardStep) => setStep(next);

  return (
    <section
      className="m-4 sm:m-6 p-6 sm:p-8 rounded-xl bg-bg-1 border border-line2 max-w-[720px]"
      aria-label="Setup wizard"
    >
      <header className="flex items-baseline justify-between gap-3 mb-3">
        <h2 className="text-xl sm:text-2xl m-0">Welcome to BotaniSense</h2>
        <span className="text-[11px] text-fg-1 font-mono">
          Step {STEP_ORDER.indexOf(step) + 1} / {STEP_ORDER.length}
        </span>
      </header>

      <Stepper current={step} />

      {step === 'connect' && (
        <Step
          title="Connect Immich"
          help="Enter where the server reaches your Immich, and an API key it can use. Takes effect immediately — no restart."
        >
          <p className="text-fg-1 text-[13px] m-0 mb-3">
            The API key is generated in Immich → Administration →
            Settings → API Keys (mark Asset Read + Album Read/Write).
            It is stored on the server and never shown again.
          </p>
          <ConnectForm
            onSaved={(_cfg, health) =>
              setConnected(health.immich?.ok === true && health.api_key?.ok === true)
            }
          />
          <StepActions
            onNext={() => advance('run')}
            nextLabel="Next: start first run"
            nextDisabled={connected !== true}
            nextHint={
              connected === null
                ? 'Checking the connection…'
                : connected === false
                ? 'Save a working connection to continue'
                : undefined
            }
          />
        </Step>
      )}

      {step === 'run' && (
        <Step
          title="Start first run"
          help="BioCLIP-2 loads lazily on first use — expect ~5 GB RSS for the duration of the run."
        >
          <p className="text-fg-1 text-[13px] m-0 mb-4">
            Run the pipeline with your seed query. It will pull the top
            {` `}
            <strong>10 000</strong>
            {` `}matching photos from Immich and classify each one
            against the TreeOfLife-200M label set. After it finishes,
            you'll land back here on the Browse view.
          </p>
          <button
            type="button"
            className="px-3 py-1.5 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover"
            onClick={() => navigate('run')}
          >
            Open Run page →
          </button>
        </Step>
      )}
    </section>
  );
}

function Step({ title, help, children }: { title: string; help: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-base m-0 mb-1">{title}</h3>
      <p className="text-fg-2 text-[11.5px] m-0 mb-3">{help}</p>
      {children}
    </div>
  );
}

function StepActions({
  onBack,
  onNext,
  nextLabel,
  nextDisabled,
  nextHint,
}: {
  onBack?: () => void;
  onNext: () => void;
  nextLabel: string;
  nextDisabled?: boolean;
  nextHint?: string;
}) {
  return (
    <div className="flex gap-2 mt-4 items-center">
      {onBack && (
        <button
          type="button"
          className="px-2.5 py-1 rounded border border-line2 text-fg text-sm hover:border-accent"
          onClick={onBack}
        >
          Back
        </button>
      )}
      <button
        type="button"
        disabled={nextDisabled}
        title={nextDisabled ? nextHint : undefined}
        className="px-3 py-1.5 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed"
        onClick={onNext}
      >
        {nextLabel}
      </button>
      {nextDisabled && nextHint && (
        <span className="text-fg-2 text-[11px]">{nextHint}</span>
      )}
    </div>
  );
}

function Stepper({ current }: { current: WizardStep }) {
  // Two-dot indicator above the active step.
  const labels: Record<WizardStep, string> = {
    connect: 'Connect',
    run: 'First run',
  };
  const order: WizardStep[] = [...STEP_ORDER];
  const activeIdx = order.indexOf(current);
  return (
    <ol className="m-0 mb-4 p-0 list-none flex items-center gap-2 text-[11px] text-fg-1">
      {order.map((s, i) => (
        <li key={s} className="flex items-center gap-2">
          <span
            aria-hidden="true"
            className={
              'inline-flex items-center justify-center w-5 h-5 rounded-full border text-[10px] font-mono ' +
              (i < activeIdx
                ? 'bg-accent/30 border-accent text-accent'
                : i === activeIdx
                ? 'bg-bg-2 border-accent text-fg'
                : 'bg-bg-2 border-line2 text-fg-1')
            }
          >
            {i < activeIdx ? '✓' : i + 1}
          </span>
          <span className={i === activeIdx ? 'text-fg' : ''}>{labels[s]}</span>
          {i < order.length - 1 && (
            <span aria-hidden="true" className="text-fg-2">›</span>
          )}
        </li>
      ))}
    </ol>
  );
}


