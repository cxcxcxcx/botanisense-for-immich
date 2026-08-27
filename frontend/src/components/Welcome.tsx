import { useEffect, useState } from 'react';
import { getResults, getStatus } from '../api/endpoints';
import { useApp } from '../state/AppContext';
import { navigate } from '../hooks/useRoute';

// Onboarding panel shown on the browse-list view when the local DB is
// empty (no pipeline runs yet). The user lands here after a fresh
// install and is greeted with what to do next, instead of an
// unexplained empty grid.

export function Welcome() {
  const { dispatch } = useApp();
  const [hasRuns, setHasRuns] = useState<boolean | null>(null);
  const [totalAssets, setTotalAssets] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getStatus().catch(() => null),
      // 1-row probe to learn the global asset count without paying for
      // the full results payload.
      getResults({ page: 0, page_size: 1 }).catch(() => null),
    ]).then(([status, resultsResp]) => {
      if (cancelled) return;
      // hasRuns: best-effort signal that the pipeline has actually run.
      // The status payload doesn't carry a run count, so we infer from
      // stage_times (any pipeline run records at least one stage time).
      const ranSomething = status != null && (
        Object.keys(status.stage_times ?? {}).length > 0 ||
        status.status === 'running'
      );
      setHasRuns(ranSomething);
      setTotalAssets(resultsResp?.pagination?.total ?? 0);
    });
    return () => { cancelled = true; };
  }, []);

  // If the probe hasn't returned yet, render a lightweight placeholder
  // so the empty grid below doesn't flash. Otherwise check both: a
  // populated DB with no runs is the normal "ready to browse" state
  // (we still show the welcome panel since they haven't seen a run).
  const showWelcome = hasRuns === false && totalAssets === 0;
  if (!showWelcome) return null;

  return (
    <section
      className="m-4 sm:m-6 p-6 sm:p-8 rounded-xl bg-bg-1 border border-line2 max-w-[640px]"
      aria-label="Welcome"
    >
      <h2 className="text-xl sm:text-2xl m-0 mb-2">Welcome to BotaniSense</h2>
      <p className="text-fg-1 text-[13px] sm:text-[14px] m-0 mb-5">
        Browse your Immich library by plant species. BioCLIP-2 ranks
        every photo against the TreeOfLife-200M label set; you group the
        results, drill into clusters, and optionally push curated
        albums back into Immich.
      </p>

      <ol className="m-0 p-0 list-none counter-reset-step space-y-3 mb-5">
        <Step n={1} title="Connect Immich">
          Set <code className="font-mono">[immich].base_url</code> in
          {' '}<code className="font-mono">config.toml</code> and
          {' '}<code className="font-mono">IMMICH_API_KEY</code> in
          {' '}<code className="font-mono">.env</code>, then restart the
          server.
        </Step>
        <Step n={2} title="Run your first pipeline">
          Open the Run page, pick a seed query
          {' '}(<em>red flowers</em>, <em>moss on rocks</em>, …), and
          click <strong>Run</strong>. BioCLIP-2 loads lazily on first
          use — expect ~5 GB RSS for the duration of the run.
        </Step>
        <Step n={3} title="Browse and sync">
          Come back here to browse by cluster, species, genus, or
          family. From the Run page you can push Plantae or per-cluster
          albums back into Immich.
        </Step>
      </ol>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className="px-3 py-1.5 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover"
          onClick={() => {
            dispatch({ type: 'SET_FILTERS_COLLAPSED', collapsed: false });
            navigate('run');
          }}
        >
          Open Run page →
        </button>
        <a
          href="https://github.com/cxcxcxcx/botanisense-for-immich#quickstart"
          target="_blank"
          rel="noopener noreferrer"
          className="px-3 py-1.5 rounded border border-line2 text-fg text-sm no-underline hover:border-accent"
        >
          View setup guide ↗
        </a>
      </div>
    </section>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="flex gap-3 items-start">
      <span
        aria-hidden="true"
        className="shrink-0 w-6 h-6 rounded-full bg-bg-2 border border-line2 text-fg-1 text-[12px] flex items-center justify-center font-mono"
      >
        {n}
      </span>
      <div>
        <strong className="block text-[14px]">{title}</strong>
        <span className="block text-fg-1 text-[12.5px]">{children}</span>
      </div>
    </li>
  );
}