import { useEffect, useState } from 'react';
import { getHealth } from '../api/endpoints';
import type { Health } from '../api/types';
import { useApp, useConnectionModal } from '../state/AppContext';

// Top-of-page banner that surfaces blocking misconfigurations before
// the user starts a 5-minute pipeline run. Fetched once on mount (not
// via SSE — health is slow-changing and a one-shot is cheap), and
// re-fetched when the "Fix connection" modal closes so a saved
// connection clears the banner without a reload.
//
// Tone:
//   - red:    Immich unreachable OR API key invalid (blockers — pipeline
//             cannot run). Carries a "Fix connection" action opening the
//             connection modal (same form as the Welcome wizard).
//   - yellow: BioCLIP weights not yet downloaded (warning — the pipeline
//             still works; KNN state is surfaced in its own section on
//             the Run page rather than as a top-level highlight).

interface Blocker {
  text: string;
  fixConnection?: boolean;
}

export function HealthBanner() {
  const [health, setHealth] = useState<Health | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const { dispatch } = useApp();
  const connectionModalOpen = useConnectionModal();

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then(h => { if (!cancelled) setHealth(h); })
      .catch(() => { /* silently null — no banner */ });
    return () => { cancelled = true; };
  }, [connectionModalOpen]);

  if (!health || dismissed) return null;

  // Categorise findings.
  const blockers: Blocker[] = [];
  const warnings: string[] = [];

  if (health.immich?.ok === false) {
    blockers.push({
      text:
        `Immich unreachable${health.immich.error ? ` — ${health.immich.error}` : ''}. ` +
        (health.immich.error === 'not configured'
          ? 'Enter your Immich URL and API key to get started.'
          : 'Check that Immich is up and the URL is right.'),
      fixConnection: true,
    });
  }
  if (health.api_key?.ok === false) {
    blockers.push({
      text:
        `Immich API key invalid${health.api_key.error ? ` (${health.api_key.error})` : ''}. ` +
        (health.api_key.error === 'not configured'
          ? ''
          : 'Generate a new one in Immich → Administration → API Keys.'),
      fixConnection: true,
    });
  }
  if (health.bioclip?.weights_present === false) {
    warnings.push(
      'BioCLIP-2 weights not downloaded yet — the first pipeline run will fetch them ' +
      `(expected under ${health.bioclip.cache_path ?? '~/.cache/huggingface'}).`,
    );
  }

  if (blockers.length === 0 && warnings.length === 0) return null;

  const isRed = blockers.length > 0;

  return (
    <div
      role={isRed ? 'alert' : 'status'}
      className={
        isRed
          ? 'm-3 p-3 rounded-md border border-[#a44] bg-[#2a1414] text-[#fbb] text-[13px]'
          : 'm-3 p-3 rounded-md border border-[#aa8] bg-[#2a2414] text-[#fdc] text-[13px]'
      }
    >
      <div className="flex items-start gap-2">
        <span aria-hidden="true" className="font-mono text-[14px] mt-[1px]">
          {isRed ? '⚠' : 'ⓘ'}
        </span>
        <div className="flex-1 min-w-0">
          <strong className="block mb-1">
            {isRed ? 'Configuration problem' : 'Heads up'}
          </strong>
          <ul className="m-0 p-0 list-none space-y-1">
            {blockers.map((b, i) => (
              <li key={i} className="flex items-center gap-2 flex-wrap">
                <span>{b.text}</span>
                {b.fixConnection && (
                  <button
                    type="button"
                    className="px-2 py-0.5 rounded border border-[#a44] text-[#fbb] text-[11px] hover:bg-[#3a1c1c]"
                    onClick={() => dispatch({ type: 'OPEN_CONNECTION_MODAL' })}
                  >
                    Fix connection
                  </button>
                )}
              </li>
            ))}
            {warnings.map((w, i) => (
              <li key={`w${i}`}>{w}</li>
            ))}
          </ul>
        </div>
        <button
          type="button"
          aria-label="Dismiss health banner"
          className="text-fg-1 hover:text-fg text-sm leading-none px-1"
          onClick={() => setDismissed(true)}
        >
          ×
        </button>
      </div>
    </div>
  );
}
