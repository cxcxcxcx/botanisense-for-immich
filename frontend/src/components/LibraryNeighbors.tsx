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
      <h5>Library neighbors</h5>
      <div className="library-neighbors">
        {items.map(n => (
          <div
            key={n.asset_id}
            className="lib-neighbor"
            onClick={() => dispatch({ type: 'OPEN_MODAL', assetId: n.asset_id })}
            title={n.original_file_name ?? n.asset_id}
          >
            <img src={thumbUrl(n.asset_id)} alt={n.original_file_name ?? ''} loading="lazy" />
            <small>{formatPct(n.similarity, 1)}</small>
          </div>
        ))}
      </div>
    </>
  );
}