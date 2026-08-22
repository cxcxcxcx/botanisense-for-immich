// Run-page "Pipeline" section. The four pipeline params (query,
// n_samples, top_k, batch_size) are collapsed by default — most users
// just want "run with server defaults". Unfolding the disclosure
// reveals the fields; the Run button always works and uses server
// defaults for any field left blank.
//
// The previous version had a separate "Refresh (server defaults)"
// button, which was confusing (it did exactly what Run does when all
// fields are blank). It's gone.

import { useState } from 'react';
import { runPipeline } from '../api/endpoints';
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

export function ParamsForm() {
  const [query, setQuery] = useState('');
  const [n, setN] = useState('');
  const [topK, setTopK] = useState('');
  const [batch, setBatch] = useState('');
  const [busy, setBusy] = useState(false);
  const [paramsOpen, setParamsOpen] = useState(false);
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

  const inputCls =
    'bg-bg-2 text-fg border border-line2 px-2.5 py-[5px] rounded text-[0.85rem] font-mono focus:outline-none focus:border-accent';

  return (
    <form onSubmit={handleRun} className="border border-line2 rounded-lg bg-bg-1 p-4 flex flex-col gap-3 w-full">
      <header className="flex items-center justify-between gap-2">
        <h3 className="m-0 text-base">Pipeline</h3>
        <button
          type="button"
          className="text-fg-1 hover:text-fg text-sm"
          onClick={() => setParamsOpen(o => !o)}
          aria-expanded={paramsOpen}
        >
          {paramsOpen ? '▼ Hide params' : '▸ Edit params'}
        </button>
      </header>

      {/* Single Run button — uses server defaults unless the user has
          unfolded the params and edited a field. Blank fields = server
          default, so the collapsed state is fully functional. */}
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={busy}
          className="px-3 py-1.5 min-w-[100px] rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-50"
        >
          {busy ? 'Running…' : 'Run pipeline'}
        </button>
        <small className="text-fg-1">
          Uses server defaults unless you edit the params.
        </small>
      </div>

      {paramsOpen && (
        <fieldset role="group" className="flex flex-wrap items-start gap-x-5 gap-y-3 border-0 p-0 m-0">
          <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[220px] flex-[1_1_220px]">
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
              {FIELDS.query?.help}
            </small>
          </label>

          <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[220px] flex-[1_1_220px]">
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
            <small className="text-fg-2 mt-0.5 text-[0.72rem] leading-[1.4]">
              {FIELDS.n_samples?.help}
            </small>
          </label>

          <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[220px] flex-[1_1_220px]">
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
            <small className="text-fg-2 mt-0.5 text-[0.72rem] leading-[1.4]">
              {FIELDS.top_k?.help}
            </small>
          </label>

          <label className="text-fg-1 text-[0.8rem] flex flex-col gap-[3px] min-w-[220px] flex-[1_1_220px]">
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
            <small className="text-fg-2 mt-0.5 text-[0.72rem] leading-[1.4]">
              {FIELDS.batch_size?.help}
            </small>
          </label>
        </fieldset>
      )}
    </form>
  );
}