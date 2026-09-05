import { useEffect, useState } from 'react';
import { getConfig, saveImmichConnection } from '../api/endpoints';
import type { Health, PublicConfig } from '../api/types';
import { useToasts } from './ToastProvider';

// Immich connection form — the web-UI replacement for editing
// config.toml / .env. Two inputs (Immich URL + API key), one save that
// POSTs /api/settings/immich and immediately shows whether the server
// can reach Immich with the saved pair (the response carries the fresh
// health payload).
//
// Used in two places:
//   - Welcome wizard step "Connect Immich" (first-run setup, inline)
//   - the "Fix connection" modal (HealthBanner blockers / ImmichSync row)
//
// The API key is write-only from the browser's point of view: the
// server only ever reports api_key_set, so a previously-saved key shows
// as a placeholder and blank input = keep it.

const INPUT_CLS =
  'bg-bg-2 text-fg border border-line2 px-2.5 py-1 rounded text-[0.85rem] font-mono focus:outline-none focus:border-accent w-full';
const LABEL_CLS = 'block text-fg-1 text-[0.8rem] flex flex-col gap-[2px]';

interface Props {
  // Called after a successful save with the fresh config + health, so
  // the parent can gate "Next" / close the modal / refresh state.
  onSaved?: (config: PublicConfig, health: Health) => void;
}

export function ConnectForm({ onSaved }: Props) {
  const push = useToasts();
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<
    { ok: boolean; message: string; envPinned: boolean } | null
  >(null);

  // Prefill from the server's non-secret view. The URL is returned in
  // the clear so an existing setup shows where it points; the key is
  // only a boolean.
  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then(cfg => {
        if (cancelled) return;
        setConfig(cfg);
        setBaseUrl(cfg.base_url);
      })
      .catch(() => { /* form starts empty; save still works */ });
    return () => { cancelled = true; };
  }, []);

  const canSave = baseUrl.trim().length > 0 && !saving;

  const save = async () => {
    setSaving(true);
    setResult(null);
    try {
      const r = await saveImmichConnection({
        base_url: baseUrl.trim(),
        api_key: apiKey.trim(), // blank = keep the saved key
      });
      const ok = r.health.immich?.ok === true && r.health.api_key?.ok === true;
      setResult({
        ok,
        message: ok
          ? `Connected${r.health.immich?.version ? ` — Immich ${r.health.immich.version}` : ''}`
          : `Saved, but the check failed: ${
              r.health.immich?.error ?? r.health.api_key?.error ?? 'unknown error'
            }`,
        envPinned: r.env_pinned,
      });
      const fresh = await getConfig().catch(() => null);
      if (fresh) {
        setConfig(fresh);
        setBaseUrl(fresh.base_url);
      }
      setApiKey('');
      if (ok) push({ kind: 'success', message: 'Immich connection saved' }, 2500);
      if (fresh && onSaved) onSaved(fresh, r.health);
    } catch (e) {
      setResult({
        ok: false,
        message: e instanceof Error ? e.message : 'Saving failed',
        envPinned: false,
      });
    } finally {
      setSaving(false);
    }
  };

  const envPinnedNow = (config?.env_pinned ?? false) && !result?.ok;

  return (
    <div className="flex flex-col gap-2.5">
      <label className={LABEL_CLS}>
        Immich URL
        <input
          type="text"
          value={baseUrl}
          onChange={e => setBaseUrl(e.target.value)}
          placeholder="http://192.168.1.10:2283"
          autoComplete="url"
          className={INPUT_CLS}
        />
      </label>
      <label className={LABEL_CLS}>
        Immich API key
        <span className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder={
              config?.api_key_set
                ? '•••••••• saved — leave blank to keep it'
                : 'paste the key from Immich → Administration → API Keys'
            }
            autoComplete="off"
            className={INPUT_CLS + ' pr-12'}
          />
          <button
            type="button"
            aria-label={showKey ? 'Hide API key' : 'Show API key'}
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-fg-2 hover:text-fg text-[11px] px-1"
            onClick={() => setShowKey(s => !s)}
          >
            {showKey ? 'hide' : 'show'}
          </button>
        </span>
      </label>

      {envPinnedNow && (
        <p className="m-0 text-[11px] text-fg-2">
          IMMICH_BASE_URL / IMMICH_API_KEY are set in the environment, so
          they pin the connection — the values saved here take effect only
          after those variables are removed.
        </p>
      )}

      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={!canSave}
          className="px-3 py-1.5 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed"
          onClick={save}
        >
          {saving ? 'Saving…' : 'Save & test connection'}
        </button>
        {result && (
          <span
            role="status"
            className={
              'text-[12px] ' + (result.ok ? 'text-green-400' : 'text-red-400')
            }
          >
            {result.ok ? '✓ ' : '✗ '}
            {result.message}
          </span>
        )}
      </div>
    </div>
  );
}
