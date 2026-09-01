import { useEffect, useState } from 'react';
import { getResults, getStatus } from '../api/endpoints';
import { navigate } from '../hooks/useRoute';

// Onboarding wizard shown on the browse-list view when the pipeline has
// not yet produced any data. Three steps — Connect → Seed query → First
// run — with a server-state-driven visibility rule:
//
//   - Loads invisible until both API calls (status + results) resolve
//     (no flicker of wizard while data is still in flight).
//   - Hides itself once the pipeline has produced at least one asset.
//     Server-state-driven so it works the same way on every port,
//     every browser, every device — no per-origin localStorage trap.
//
// localStorage is used ONLY for the seed-query text (a UX preference,
// not a state). The wizard's step state is component-local — it
// resets each session, which is fine because we're only using it for
// the in-progress walk-through cursor.
//
// Steps:
//   1. Connect Immich — show the exact lines to paste into
//      config.toml and .env (no server endpoint to write them, by
//      design — secrets stay on disk).
//   2. Seed query — text input; value persists to localStorage and
//      is read by ParamsForm as the default.
//   3. First run — a CTA that jumps to the Run page with the seed
//      query pre-filled.

type WizardStep = 'connect' | 'seed' | 'run';

const STEP_ORDER: WizardStep[] = ['connect', 'seed', 'run'];

// localStorage is only for the seed-query text — not for any
// visibility state.
const STORAGE_KEY_SEED = 'botanisense:seed-query';

function readSeed(): string {
  if (typeof window === 'undefined') return 'intentional plant photos';
  return localStorage.getItem(STORAGE_KEY_SEED) || 'intentional plant photos';
}

export function Welcome() {
  const [step, setStep] = useState<WizardStep>('connect');
  const [seedQuery, setSeedQuery] = useState<string>(readSeed);
  const [hasRuns, setHasRuns] = useState<boolean | null>(null);
  const [totalAssets, setTotalAssets] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getStatus().catch(() => null),
      getResults({ page: 0, page_size: 1 }).catch(() => null),
    ]).then(([status, resultsResp]) => {
      if (cancelled) return;
      const ranSomething = status != null && (
        Object.keys(status.stage_times ?? {}).length > 0 ||
        status.status === 'running'
      );
      setHasRuns(ranSomething);
      setTotalAssets(resultsResp?.pagination?.total ?? 0);
    });
    return () => { cancelled = true; };
  }, []);

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
          Step {STEP_ORDER.indexOf(step) + 1} / 3
        </span>
      </header>

      <Stepper current={step} />

      {step === 'connect' && (
        <Step
          title="Connect Immich"
          help="Two config files, both gitignored. Edit them once, restart the server."
        >
          <p className="text-fg-1 text-[13px] m-0 mb-3">
            Copy the templates and fill in your Immich URL + admin API
            key. The API key is generated in Immich → Administration →
            Settings → API Keys.
          </p>
          <CodeBlock
            lines={[
              '# In config.toml',
              '[immich]',
              'base_url = "http://192.168.1.10:2283"',
              '',
              '# In .env',
              'IMMICH_API_KEY=your-key-here',
            ]}
          />
          <p className="text-fg-2 text-[12px] m-0 mt-3">
            After editing, restart uvicorn (Ctrl+C, then re-run).
          </p>
          <StepActions onNext={() => advance('seed')} nextLabel="Next: pick a seed query" />
        </Step>
      )}

      {step === 'seed' && (
        <Step
          title="Pick a seed query"
          help="This is what the pipeline asks Immich's smart-search for. Try descriptive phrases — plants, scenes, or photo style."
        >
          <p className="text-fg-1 text-[13px] m-0 mb-3">
            The default covers most libraries. Replace it with
            something narrower if your photos need it.
          </p>
          <input
            type="text"
            value={seedQuery}
            onChange={e => {
              setSeedQuery(e.target.value);
              localStorage.setItem(STORAGE_KEY_SEED, e.target.value);
            }}
            placeholder="intentional plant photos"
            className="w-full bg-bg-2 text-fg border border-line2 px-2.5 py-2 rounded text-[0.9rem] font-mono focus:outline-none focus:border-accent"
          />
          <p className="text-fg-2 text-[11px] m-0 mt-2">
            Examples: <em>red flowers</em>, <em>moss on rocks</em>, <em>tree bark close-up</em>, <em>succulents in pots</em>.
          </p>
          <StepActions
            onBack={() => advance('connect')}
            onNext={() => advance('run')}
            nextLabel="Next: start first run"
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
}: {
  onBack?: () => void;
  onNext: () => void;
  nextLabel: string;
}) {
  return (
    <div className="flex gap-2 mt-4">
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
        className="px-3 py-1.5 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover"
        onClick={onNext}
      >
        {nextLabel}
      </button>
    </div>
  );
}

function Stepper({ current }: { current: WizardStep }) {
  // Three-dot indicator above the active step.
  const labels: Record<WizardStep, string> = {
    connect: 'Connect',
    seed: 'Seed query',
    run: 'First run',
  };
  const order: WizardStep[] = ['connect', 'seed', 'run'];
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

function CodeBlock({ lines }: { lines: string[] }) {
  return (
    <pre className="m-0 p-3 rounded bg-bg-2 border border-line2 text-[12px] font-mono text-fg overflow-x-auto">
{lines.join('\n')}
    </pre>
  );
}
