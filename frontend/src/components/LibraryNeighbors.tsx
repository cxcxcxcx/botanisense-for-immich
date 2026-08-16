import type { LibraryNeighbor } from '../api/types';
import { thumbUrl } from '../lib/thumb';
import { useApp } from '../state/AppContext';
import { formatPct } from '../lib/format';

interface Props {
  items: LibraryNeighbor[];
}

export function LibraryNeighbors({ items }: Props) {
  const { dispatch } = useApp();
  if (!items || items.length === 0) return null;
  return (
    <>
      <h5 className="m-4 mb-1 text-[0.95rem] text-[#ad9]">Library neighbors</h5>
      <div className="grid gap-1.5 grid-cols-[repeat(auto-fill,minmax(100px,1fr))]">
        {items.map(n => (
          <div
            key={n.asset_id}
            className="flex flex-col items-center gap-0.5 cursor-pointer"
            onClick={() => dispatch({ type: 'OPEN_MODAL', assetId: n.asset_id })}
            title={n.original_file_name ?? n.asset_id}
          >
            <img src={thumbUrl(n.asset_id)} alt={n.original_file_name ?? ''} loading="lazy" className="w-full h-20 object-cover rounded cursor-pointer" />
            <small className="text-[10px] text-fg-1 font-mono">{formatPct(n.similarity, 1)}</small>
          </div>
        ))}
      </div>
    </>
  );
}