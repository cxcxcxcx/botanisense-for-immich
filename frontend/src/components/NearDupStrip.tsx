import { useState } from 'react';
import type { ResultRow } from '../api/types';
import { thumbUrl } from '../lib/thumb';
import { useToasts } from './ToastProvider';

interface Props {
  row: ResultRow;
  primaryAid: string;
  onPickPrimary: (aid: string) => void;
}

export function NearDupStrip({ row, primaryAid, onPickPrimary }: Props) {
  const [open, setOpen] = useState(false);
  const push = useToasts();
  const dups = row.near_dups ?? [];
  if (dups.length === 0) return null;
  return (
    <div className="near-dup">
      <button
        type="button"
        className="near-dup-toggle"
        onClick={e => { e.stopPropagation(); setOpen(o => !o); }}
      >
        {open ? '▾' : '▸'} {dups.length} similar image{dups.length > 1 ? 's' : ''} (cosine ≥ 0.99)
      </button>
      {open && (
        <div className="near-dup-panel">
          {dups.map(d => (
            <div key={d.asset_id} className="near-dup-item">
              <img src={thumbUrl(d.asset_id)} alt={d.original_file_name ?? ''} loading="lazy" />
              <div className="near-dup-name" title={d.original_file_name ?? ''}>
                {d.original_file_name ?? d.asset_id}
              </div>
              <button
                type="button"
                className="near-dup-pick"
                onClick={e => {
                  e.stopPropagation();
                  if (d.asset_id !== primaryAid) onPickPrimary(d.asset_id);
                  push({ kind: 'info', message: 'Primary image updated (local only)' }, 2000);
                }}
              >
                use as primary
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}