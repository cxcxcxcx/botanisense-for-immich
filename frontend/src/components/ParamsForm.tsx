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
    <small className="text-fg-2 mt-0.5 text-[0.72rem] leading-[1.4]">
      {FIELDS[k]?.help}
    </small>
  );

  // Shared label class — vertical-stack input + help text. min-width
  // 220px + flex: 1 lets the four fields wrap onto multiple lines on
  // narrow viewports while sharing the row width.
  const labelCls = 'text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[220px] flex-[1_1_220px]';
  const inputCls = 'bg-bg-2 text-fg border border-line2 px-2.5 py-[5px] rounded text-[0.85rem] font-mono focus:outline-none focus:border-accent';

  return (
    <form onSubmit={handleRun} className="p-0">
      <fieldset role="group" className="flex flex-wrap items-start gap-x-5 gap-y-3 border-0 p-0 m-0">
        <label className={labelCls}>
          {FIELDS.query?.label}
          <input
            type="text"
            className={`${inputCls} min-w-[220px] w-full`}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={FIELDS.query?.placeholder}
            title={FIELDS.query?.help}
          />
          <HelpText k="query" />
        </label>

        <label className={labelCls}>
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
          <HelpText k="n_samples" />
        </label>

        <label className={labelCls}>
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
          <HelpText k="top_k" />
        </label>

        <label className={labelCls}>
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
          <HelpText k="batch_size" />
        </label>

        <div className="flex gap-2 items-center mt-2 flex-wrap">
          <button type="submit" disabled={busy} className="px-2.5 py-1 min-w-[80px] rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-50">
            {busy ? 'Running…' : 'Run'}
          </button>
          <button type="button" className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345] hover:border-accent" onClick={handleRefresh}>
            ↻ Refresh (server defaults)
          </button>
          <small className="text-fg-1">empty fields = use server default</small>
        </div>
      </fieldset>
    </form>
  );
}