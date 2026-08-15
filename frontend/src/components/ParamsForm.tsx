import { useState } from 'react';
import { runPipeline, refreshPipeline } from '../api/endpoints';
import { useToasts } from './ToastProvider';

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

  return (
    <details className="params-details">
      <summary>Pipeline parameters</summary>
      <form onSubmit={handleRun}>
        <fieldset role="group">
          <label>query <input type="text" value={query} onChange={e => setQuery(e.target.value)} placeholder="intentional plant photos" /></label>
          <label>n <input type="number" value={n} onChange={e => setN(e.target.value)} min={1} max={1000} /></label>
          <label>top_k <input type="number" value={topK} onChange={e => setTopK(e.target.value)} min={1} max={20} /></label>
          <label>batch <input type="number" value={batch} onChange={e => setBatch(e.target.value)} min={1} max={64} /></label>
          <button type="submit" disabled={busy}>{busy ? 'Running…' : 'Run'}</button>
          <button type="button" className="secondary outline" onClick={handleRefresh}>Refresh</button>
          <small>empty fields = use server default</small>
        </fieldset>
      </form>
    </details>
  );
}