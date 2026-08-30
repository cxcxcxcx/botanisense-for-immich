// The K nearest reference photos behind a KNN prediction, rendered as a
// thumbnail grid. This is the "show your work" view: the top-1 species
// string alone gives a user no way to tell a good match from a confident
// wrong one, but seeing the actual reference images makes it obvious.
//
// Thumbnails go through /api/knn/ref-thumb (see app.py) rather than
// hotlinking the gbif/eol URL directly — that proxy caches, and it keeps
// the user's IP off a few hundred herbarium hosts.

import { useState } from 'react';
import type { KnnNeighbor } from '../api/types';
import { refThumbUrl } from '../lib/thumb';

interface Props {
  neighbors: KnnNeighbor[];
  // How many to show. The rest stay hidden behind the count line.
  limit?: number;
}

function RefThumb({ n }: { n: KnnNeighbor }) {
  // A reference image can fail for reasons we can't fix from here: the
  // source host is down, the media URL 404s (common in older gbif
  // records), or the row has no identifier at all (bioscan). Fall back
  // to a labelled placeholder instead of a broken-image icon.
  const [broken, setBroken] = useState(false);
  const src = n.identifier ? refThumbUrl(n.identifier) : null;

  return (
    <div className="knn-ref">
      {src && !broken ? (
        <img
          src={src}
          alt={`Reference photo for ${n.species}`}
          loading="lazy"
          className="knn-ref-img"
          onError={() => setBroken(true)}
        />
      ) : (
        <div className="knn-ref-img knn-ref-missing" aria-hidden="true">
          no photo
        </div>
      )}
      <div className="knn-ref-meta">
        <div className="knn-ref-sp" title={n.species}>{n.species}</div>
        {n.common_name ? (
          <div className="knn-ref-cn" title={n.common_name}>{n.common_name}</div>
        ) : null}
        {/* Raw similarity, not a percentage. The index is inner-product
            over unnormalized reference vectors, so `sim` is not bounded
            to [0,1] and rendering it as a percent produces nonsense
            like "2001.9%". */}
        <div className="knn-ref-sim">sim {n.sim.toFixed(2)}</div>
      </div>
    </div>
  );
}

export function KnnRefNeighbors({ neighbors, limit = 6 }: Props) {
  if (!neighbors?.length) return null;
  const shown = neighbors.slice(0, limit);
  const hidden = neighbors.length - shown.length;

  return (
    <div>
      <div className="knn-ref-grid">
        {shown.map((n, i) => (
          <RefThumb key={`${n.uuid ?? n.species}-${i}`} n={n} />
        ))}
      </div>
      {hidden > 0 && (
        <div className="knn-ref-more">+ {hidden} more neighbours</div>
      )}
    </div>
  );
}
