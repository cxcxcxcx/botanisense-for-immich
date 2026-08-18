import { useEffect, useState } from 'react';
import { getAsset, reclassifyBbox, reclassify } from '../api/endpoints';
import type { AssetDetail as AssetDetailType } from '../api/types';
import { thumbUrl } from '../lib/thumb';
import { Predictions } from './Predictions';
import { LibraryNeighbors } from './LibraryNeighbors';
import { DetectionBlock } from './DetectionBlock';
import { BboxImage } from './BboxImage';
import { useToasts } from './ToastProvider';
import { plantaeColorClass } from '../lib/plantae';
import { formatPct } from '../lib/format';

interface Props {
  assetId: string;
}

export function AssetDetail({ assetId }: Props) {
  const [data, setData] = useState<AssetDetailType | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [copied, setCopied] = useState(false);
  // If the 'fullsize' thumb fails to load, fall back to the smaller
  // 'preview' size. Reset on asset change so each asset tries fullsize
  // first.
  const [usePreview, setUsePreview] = useState(false);
  const push = useToasts();

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    setCopied(false);
    setUsePreview(false);
    getAsset(assetId)
      .then(d => { if (!cancelled) setData(d); })
      .catch(e => { if (!cancelled) setError(e); });
    return () => { cancelled = true; };
  }, [assetId]);

  const handleReclassifyAll = async () => {
    try {
      await reclassify([assetId]);
      push({ kind: 'success', message: 'Reclassify queued' }, 2000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    }
  };

  const handleReclassifyBbox = async (bboxIndex: number) => {
    try {
      await reclassifyBbox({ asset_id: assetId, bbox_index: bboxIndex });
      push({ kind: 'info', message: 'Per-crop reclassify is a no-op stub (use Reclassify with KNN)' }, 3000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    }
  };

  if (error) return <div className="text-red-300">Failed to load asset: {error.message}</div>;
  if (!data) return <p aria-busy="true">Loading…</p>;

  const fullTop1 = data.top_predictions[0];
  const plantaeScore = data.taxon_scores?.Plantae;
  const plantaeTier = plantaeColorClass(plantaeScore);

  return (
    <div className="text-[13px]">
      {/* Image + YOLO boxes are drawn on a single <canvas> by
          BboxImage (contain mode), so there is no separate overlay to
          align — the boxes and the photo share one coordinate system
          and are guaranteed to line up. The canvas lays out like an
          <img>: max-h 220px, max-w 100%, aspect ratio preserved. The
          text-center wrapper keeps it horizontally centered. A failed
          fullsize load falls back to the preview size. */}
      <div className="text-center mb-2">
        <figure className="m-0 inline-block bg-[#0a0a0a] rounded-md overflow-hidden border border-[#222] max-w-full align-top">
          <BboxImage
            src={thumbUrl(data.asset_id, usePreview ? 'preview' : 'fullsize')}
            detections={data.detections}
            fit="contain"
            maxHeight={220}
            ariaLabel={data.original_file_name}
            onError={() => setUsePreview(true)}
          />
        </figure>
      </div>

      <p className="mb-0.5">
        <strong>{data.original_file_name}</strong>
        {data.cluster_id != null && <small className="text-fg-1"> · cluster {data.cluster_id}</small>}
      </p>

      {/* Asset ID + copy button. Subtle monospace line so users can
          grab the UUID (for sharing / debugging / pasting into
          /api/asset/{id}) without scraping it from the URL. The
          button label flips to "Copied" for 1.5s after a successful
          clipboard write so the user gets visual feedback. */}
      <p className="-mt-1 mb-2 text-[11px] text-fg-1 flex items-center gap-2 flex-wrap">
        <code
          className="font-mono text-fg-1 select-all"
          title={data.asset_id}
        >{data.asset_id}</code>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded border border-[#345] bg-transparent text-[10px] text-fg-1 hover:bg-[#234] hover:border-accent"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(data.asset_id);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            } catch (e) {
              push({ kind: 'error', message: 'Copy failed: ' + (e instanceof Error ? e.message : String(e)) }, 3000);
            }
          }}
        >{copied ? 'Copied ✓' : 'Copy ID'}</button>
        {/* Person chip — mirrors the grid filter state. Three states:
            true (👤, accent), false (no person, dim), null (unscanned,
            italic 'unscanned'). Drives from data.has_person which is
            set by the pipeline's "person" stage. */}
        {data.has_person === true && (
          <span className="px-1.5 py-0.5 rounded border border-accent text-accent text-[10px] font-medium" title="YOLO detected a person in this image">
            👤 person
          </span>
        )}
        {data.has_person === false && (
          <span className="px-1.5 py-0.5 rounded border border-[#345] text-fg-1 text-[10px]" title="Scanned, no person detected">
            no person
          </span>
        )}
        {data.has_person === null && (
          <span className="px-1.5 py-0.5 rounded border border-[#234] text-fg-1 text-[10px] italic" title="Person scan hasn't run on this asset yet">
            unscanned
          </span>
        )}
      </p>

      {(data.captured_at || (data.latitude != null && data.longitude != null)) && (
        <p className="-mt-1 mb-2 text-[11px] text-fg-1 flex flex-wrap gap-x-4 gap-y-1">
          {data.captured_at && (
            <span title={data.captured_at}>
              🕒 {new Date(data.captured_at).toLocaleString()}
            </span>
          )}
          {data.latitude != null && data.longitude != null && (
            <a
              href={`geo:${data.latitude},${data.longitude}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#7ad] no-underline hover:underline"
              title={`${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)}`}
            >
              📍 {data.latitude.toFixed(4)}, {data.longitude.toFixed(4)}
            </a>
          )}
        </p>
      )}

      <nav className="flex flex-wrap items-center gap-x-1.5 gap-y-1 py-1 mb-0.5 text-[12px] leading-[1.3]" aria-label="taxonomy hierarchy">
        {fullTop1?.kingdom && <span className="rank-kingdom">{fullTop1.kingdom}</span>}
        {fullTop1?.family && (
          <>
            <span className="rank-sep">›</span>
            <span className="rank-family">{fullTop1.family}</span>
          </>
        )}
        {fullTop1?.genus && (
          <>
            <span className="rank-sep">›</span>
            <span className="rank-genus">{fullTop1.genus}</span>
          </>
        )}
        {fullTop1?.species && (
          <>
            <span className="rank-sep">›</span>
            <span className="rank-species">{fullTop1.species}</span>
          </>
        )}
        {plantaeScore != null && (
          <span className={`inline-block ml-auto px-2 py-0.5 rounded font-bold text-[11px] font-mono ${plantaeTier}`}>
            Plantae {(plantaeScore * 100).toFixed(1)}%
          </span>
        )}
      </nav>

      <LibraryNeighbors items={data.library_neighbors} />

      <h5 className="m-4 mb-1 text-[0.95rem] text-[#ad9]">Top-{data.top_predictions.length} predictions</h5>
      <div className="pt-0 pb-3 px-3">
        <Predictions items={(data.top_predictions ?? []).slice(0, 5)} />
      </div>

      {data.knn?.top_species ? (
        <>
          <h5 className="m-4 mb-1 text-[0.95rem] text-[#ad9]">KNN (re-classified)</h5>
          <div className="px-3 pb-3 border-t border-dashed border-[#333] mt-1">
            <div className="flex justify-between mb-1 text-[12px]">
              <span>
                {data.knn.top_species}
                {data.knn.common_name ? ` (${data.knn.common_name})` : ''} · {formatPct(data.knn.score)}
              </span>
              <small className="text-fg-1">{data.knn.kingdom}{data.knn.family ? ` · ${data.knn.family}` : ''}</small>
            </div>
            {(data.knn.neighbors ?? []).map((n, i) => (
              <div key={`${n.species}-${i}`} className="flex justify-between gap-2.5 py-1 border-t border-[#222] text-[13px] first:border-t-0">
                <div className="flex-1">
                  {n.species}
                  <div className="text-fg-1 text-[11px]">{n.common_name ?? ''}</div>
                </div>
                <div className="text-[#ad9] tabular-nums min-w-[56px] text-right">{formatPct(n.sim, 1)}</div>
              </div>
            ))}
          </div>
        </>
      ) : null}
      <button type="button" className="bg-[#234] text-fg border border-[#456] px-2.5 py-1 rounded text-[11px] cursor-pointer hover:bg-[#345] mt-1" onClick={handleReclassifyAll}>
        Re-run KNN
      </button>

      {(data.detections ?? []).filter(d => d.score >= 0.10).length > 0 && (
        <>
          <h5 className="m-4 mb-1 text-[0.95rem] text-[#ad9]">YOLO detections (per-crop)</h5>
          {data.detections
            .filter(d => d.score >= 0.10)
            .map((d, i) => (
              <div key={d.id}>
                <DetectionBlock det={d} fullTop1={fullTop1} />
                <button type="button" className="bg-[#234] text-fg border border-[#456] px-2.5 py-1 rounded text-[11px] cursor-pointer hover:bg-[#345] mt-1" onClick={() => handleReclassifyBbox(i)}>
                  Reclassify crop
                </button>
              </div>
            ))}
        </>
      )}
    </div>
  );
}
