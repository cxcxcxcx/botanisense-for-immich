import { useEffect, useState } from 'react';
import { getHealth } from '../api/endpoints';
import type { Health } from '../api/types';

// Top-of-page banner that surfaces blocking misconfigurations before
// the user starts a 5-minute pipeline run. Polled once on mount (not
// via SSE — health is slow-changing and a one-shot is cheap).
//
// Tone:
//   - red:    Immich unreachable OR API key invalid (blockers — pipeline
//             cannot run).
//   - yellow: KNN index missing OR BioCLIP weights not yet downloaded
//             (warnings — pipeline still works in zero-shot mode).

export function HealthBanner() {
  const [health, setHealth] = useState<Health | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then(h => { if (!cancelled) setHealth(h); })
      .catch(() => { /* silently null — no banner */ });
    return () => { cancelled = true; };
  }, []);

  if (!health || dismissed) return null;

  // Categorise findings.
  const blockers: string[] = [];
  const warnings: string[] = [];

  if (health.immich?.ok === false) {
    blockers.push(
      `Immich unreachable${health.immich.error ? ` — ${health.immich.error}` : ''}. ` +
      'Check `[immich].base_url` in config.toml.',
    );
  }
  if (health.api_key?.ok === false) {
    blockers.push(
      `Immich API key invalid${health.api_key.error ? ` (${health.api_key.error})` : ''}. ` +
      'Regenerate in Immich → Administration → API Keys.',
    );
  }
  if (health.knn?.configured && health.knn?.built === false) {
    warnings.push(
      `KNN reference index missing at ${health.knn.path}. ` +
      'Build it from the Run page to enable re-classification.',
    );
  }
  if (health.bioclip?.weights_present === false) {
    warnings.push(
      'BioCLIP-2 weights not downloaded yet — the first pipeline run will fetch them ' +
      `(expected under ${health.bioclip.cache_path ?? '~/.cache/huggingface'}).`,
    );
  }

  if (blockers.length === 0 && warnings.length === 0) return null;

  const items = [...blockers, ...warnings];
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
            {items.map((m, i) => (
              <li key={i}>{m}</li>
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