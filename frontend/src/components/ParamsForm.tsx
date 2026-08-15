import { useState } from 'react';
import { runPipeline, refreshPipeline } from '../api/endpoints';
import { useToasts } from './ToastProvider';

interface FieldHelp {
  label: string;
  help: string;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
}

// Each tunable documented for non-BioCLIP users. These appear as
// placeholder text + a small tooltip on the right of the input.
const FIELDS: Record<string, FieldHelp> = {
  query: {
    label: 'query',
    help: 'Free-text search phrase sent to Immich to pick the candidate photos. Leave blank to use the server default (the IMMICH_QUERY env var).',
    placeholder: 'intentional plant photos',
  },
  n_samples: {
    label: 'photos to fetch',
    help: 'How many candidate photos to pull from Immich for this run. Higher = more to classify, longer runtime. No hard cap — the server processes whatever you give it (default 120).',
    placeholder: '120',
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
    help: 'How many images BioCLIP processes at once. Higher = faster but more GPU memory. Server default is 16. 32 is usually fine on a 12 GB GPU.',
    placeholder: '16',
    min: 1,
    max: 64,
  },
};

export function ParamsForm() {
  const [query, setQuery] = useState('');
  const [n, setN] = useState('');
  const [topK, setTopK] = useState('');
  const [batch, setBatch] = useState('');
  const [busy, setBusy] = useState(false);
  const push = useToasts();

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      await runPipeline({
        query: query.trim() || null,
        n_samples: n ? Number(n) : null,
        top_k: topK ? Number(topK) : null,
        batch_size: batch ? Number(batch) : null,
      });
      push({ kind: 'success', message: 'Pipeline started' }, 2500);
    } catch (err) {
      push({ kind: 'info', message: err instanceof Error ? err.message : String(err) }, 4000);
    } finally {
      setBusy(false);
    }
  };

  const handleRefresh = async () => {
    try {
      await refreshPipeline();
      push({ kind: 'info', message: 'Refresh queued' }, 2000);
    } catch (err) {
      push({ kind: 'info', message: err instanceof Error ? err.message : String(err) }, 4000);
    }
  };

  const HelpText = ({ k }: { k: keyof typeof FIELDS }) => (
    <small className="muted" style={{ display: 'block', marginTop: '0.15rem', fontSize: '0.7rem' }}>
      {FIELDS[k]?.help}
    </small>
  );

  return (
    <details className="params-details">
      <summary>Pipeline parameters</summary>
      <form onSubmit={handleRun}>
        <fieldset role="group">
          <label>
            {FIELDS.query?.label}
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder={FIELDS.query?.placeholder}
              title={FIELDS.query?.help}
            />
            <HelpText k="query" />
          </label>

          <label>
            {FIELDS.n_samples?.label}
            <input
              type="number"
              value={n}
              onChange={e => setN(e.target.value)}
              min={FIELDS.n_samples?.min}
              max={FIELDS.n_samples?.max}
              placeholder={FIELDS.n_samples?.placeholder}
              title={FIELDS.n_samples?.help}
            />
            <HelpText k="n_samples" />
          </label>

          <label>
            {FIELDS.top_k?.label}
            <input
              type="number"
              value={topK}
              onChange={e => setTopK(e.target.value)}
              min={FIELDS.top_k?.min}
              max={FIELDS.top_k?.max}
              placeholder={FIELDS.top_k?.placeholder}
              title={FIELDS.top_k?.help}
            />
            <HelpText k="top_k" />
          </label>

          <label>
            {FIELDS.batch_size?.label}
            <input
              type="number"
              value={batch}
              onChange={e => setBatch(e.target.value)}
              min={FIELDS.batch_size?.min}
              max={FIELDS.batch_size?.max}
              placeholder={FIELDS.batch_size?.placeholder}
              title={FIELDS.batch_size?.help}
            />
            <HelpText k="batch_size" />
          </label>

          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '0.5rem' }}>
            <button type="submit" disabled={busy}>{busy ? 'Running…' : 'Run'}</button>
            <button type="button" className="secondary outline" onClick={handleRefresh}>
              ↻ Refresh (server defaults)
            </button>
            <small className="muted">empty fields = use server default</small>
          </div>
        </fieldset>
      </form>
    </details>
  );
}