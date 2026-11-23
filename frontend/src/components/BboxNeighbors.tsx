import { useEffect, useRef, useState } from 'react';
import { getBboxNeighbors } from '../api/endpoints';
import type { BboxNeighbor, BboxNeighborsResponse } from '../api/types';
import { formatPct } from '../lib/format';

interface Props {
  detectionId: number;
  onOpenAsset: (assetId: string) => void;
  onClose?: () => void;
}

function NeighborCard({
  item,
  onOpenAsset,
}: {
  item: BboxNeighbor;
  onOpenAsset: (assetId: string) => void;
}) {
  const title = item.species || item.common_name || item.label || item.original_file_name || '—';
  const sub = item.kind === 'box'
    ? (item.common_name || item.label || '')
    : (item.original_file_name || 'whole photo');
  const nBoxes = item.n_boxes ?? (item.kind === 'box' ? 1 : 0);
  const cropInfo = item.kind === 'box'
    ? (nBoxes > 1 ? `${nBoxes} crops · best` : 'crop')
    : (nBoxes > 0 ? `photo · ${nBoxes} crops` : 'photo');
  return (
    <button
      type="button"
      onClick={() => onOpenAsset(item.asset_id)}
      className="group text-left rounded-md border border-line2 bg-bg-2 p-1.5 hover:border-accent transition-colors"
      title={`${item.kind} · ${title} · ${nBoxes} crop${nBoxes === 1 ? '' : 's'} · similarity ${(item.similarity * 100).toFixed(1)}%`}
    >
      <div className="aspect-square w-full overflow-hidden rounded bg-bg-0">
        <img
          src={item.thumb_url}
          alt=""
          loading="lazy"
          className="h-full w-full object-cover group-hover:scale-[1.02] transition-transform"
        />
      </div>
      <div className="mt-1 truncate text-[11px] font-medium text-fg">{title}</div>
      <div className="flex items-center justify-between gap-1 text-[10px] text-fg-2">
        <span className="truncate">{sub}</span>
        <span className="tabular-nums text-accent">{formatPct(item.similarity)}</span>
      </div>
      <div className="text-[9px] uppercase tracking-wider text-fg-2">
        {cropInfo}
      </div>
    </button>
  );
}

/**
 * Similar crops + whole photos for one YOLO box. Box and photo vectors are
 * both teacher BioCLIP embeddings; the whole photo is the "virtual box"
 * fallback for photos where YOLO found nothing.
 */
export function BboxNeighbors({ detectionId, onOpenAsset, onClose }: Props) {
  const [data, setData] = useState<BboxNeighborsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const sectionRef = useRef<HTMLElement | null>(null);

  // The Similar button lives in the debug list at the bottom of the modal;
  // the panel is inserted near the top. Bring it into view when it opens.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      sectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [detectionId]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    getBboxNeighbors(detectionId, 18, true)
      .then(result => {
        if (!cancelled) setData(result);
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [detectionId]);

  const boxes = data?.neighbors.filter(n => n.kind === 'box') ?? [];
  const photos = data?.neighbors.filter(n => n.kind === 'photo') ?? [];

  return (
    <section
      ref={sectionRef}
      className="rounded-lg border border-line2 bg-bg-1 p-2.5"
      aria-label="Similar boxes"
    >
      <header className="flex items-start gap-2">
        {data?.query && (
          <div className="h-14 w-14 flex-shrink-0 overflow-hidden rounded border border-line2 bg-bg-0">
            <img
              src={data.query.thumb_url}
              alt=""
              className="h-full w-full object-cover"
            />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="text-[12px] font-semibold text-fg">Similar boxes & photos</div>
          {data?.query && (
            <div className="truncate text-[11px] text-fg-1">
              {data.query.species || data.query.label} · YOLO {formatPct(data.query.score)}
              {data.query.species_score != null ? ` · BioCLIP ${formatPct(data.query.species_score)}` : ''}
            </div>
          )}
          <div className="text-[10px] text-fg-2">
            teacher space · one card per source image · query image excluded
          </div>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="flex-shrink-0 px-1.5 text-fg-2 hover:text-fg"
            aria-label="Close similar boxes"
          >
            ×
          </button>
        )}
      </header>

      {loading && <p className="mt-2 text-[11px] text-fg-2">Finding similar crops…</p>}
      {error && <p className="mt-2 text-[11px] text-red-300">{error}</p>}

      {!loading && !error && data && (
        <>
          <div className="mt-2 text-[10px] uppercase tracking-wider text-fg-2">
            Crops ({boxes.length} images)
          </div>
          {boxes.length === 0 ? (
            <p className="text-[11px] text-fg-2">No similar crops.</p>
          ) : (
            <div className="mt-1 grid grid-cols-3 gap-1.5">
              {boxes.map(item => (
                <NeighborCard
                  key={`box-${item.detection_id}`}
                  item={item}
                  onOpenAsset={onOpenAsset}
                />
              ))}
            </div>
          )}

          <div className="mt-2 text-[10px] uppercase tracking-wider text-fg-2">
            Whole photos ({photos.length})
          </div>
          {photos.length === 0 ? (
            <p className="text-[11px] text-fg-2">No similar whole photos.</p>
          ) : (
            <div className="mt-1 grid grid-cols-3 gap-1.5">
              {photos.map(item => (
                <NeighborCard
                  key={`photo-${item.asset_id}`}
                  item={item}
                  onOpenAsset={onOpenAsset}
                />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
