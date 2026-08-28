import { useEffect, useState } from 'react';
import { getAsset, reclassifyBbox, reclassify, getConfig } from '../api/endpoints';
import type { AssetDetail as AssetDetailType, PlantAttributionRow } from '../api/types';
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

// Per-row "source" chips (yolo crop / mixed) were removed — the
// attribution pathway is visible from the row itself (highlighted
// bbox on the image + the row's species label is the same as the
// underlying top-1 prediction, so an extra "yolo crop" / "mixed"
// pill was duplicating context that's already apparent).

// Build a Google search URL for a free-form taxonomy term. The query
// keeps the actual name verbatim — quoted ("...") so Google searches
// the exact phrase, and prepended with "plant" so a common English
// word like "Rose" doesn't return recipes / nursery homepages.
function googleSearchUrl(term: string, hint?: string): string {
  const q = hint
    ? `"${term}" ${hint}`
    : `"${term}"`;
  return `https://www.google.com/search?q=${encodeURIComponent(q)}`;
}

// Official Google "G" logo as inline SVG. Sized via the `className`.
// Source: Google's published brand assets (public domain). Stays
// inline so the icon doesn't break if the bundle is loaded offline.
function GoogleIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path fill="#4285F4" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/>
      <path fill="#34A853" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/>
      <path fill="#FBBC05" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/>
      <path fill="#EA4335" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571.001-.001.002-.001.003-.002l6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/>
    </svg>
  );
}

export function AssetDetail({ assetId }: Props) {
  const [data, setData] = useState<AssetDetailType | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [copied, setCopied] = useState(false);
  // If the 'fullsize' thumb fails to load, fall back to the smaller
  // 'preview' size. Reset on asset change so each asset tries fullsize
  // first.
  const [usePreview, setUsePreview] = useState(false);
  // Public Immich URL for the "Open in Immich ↗" deep-link. Fetched
  // once on mount; null hides the button (e.g. behind a misconfigured
  // reverse proxy).
  const [immichPublicUrl, setImmichPublicUrl] = useState<string | null>(null);
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

  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then(c => { if (!cancelled) setImmichPublicUrl(c.immich_public_url); })
      .catch(() => { /* silently no link */ });
    return () => { cancelled = true; };
  }, []);

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
  const attribution = data.plant_attribution;
  const attributionRows = attribution?.rows ?? [];
  const highlightIds = attribution?.highlight_detection_ids ?? [];

  return (
    <div className="text-[13px]">
      {/* Image + YOLO boxes are drawn on a single <canvas> by
          BboxImage (contain mode), so there is no separate overlay to
          align — the boxes and the photo share one coordinate system
          and are guaranteed to line up. The canvas lays out like an
          <img>: max-h 220px, max-w 100%, aspect ratio preserved. The
          text-center wrapper keeps it horizontally centered. A failed
          fullsize load falls back to the preview size.
          highlightDetectionIds pushes the attribution-accepted boxes
          onto the canvas — even low-YOLO-score boxes are drawn when
          the whole-image plantae score confirms the bbox. */}
      <div className="text-center mb-2">
        <figure className="m-0 inline-block bg-[#0a0a0a] rounded-md overflow-hidden border border-[#222] max-w-full align-top">
          <BboxImage
            src={thumbUrl(data.asset_id, usePreview ? 'preview' : 'fullsize')}
            detections={data.detections}
            highlightDetectionIds={highlightIds}
            fit="contain"
            maxHeight={220}
            ariaLabel={data.original_file_name}
            onError={() => setUsePreview(true)}
          />
        </figure>
      </div>

      <div className="mb-0.5 flex items-center gap-2 flex-wrap">
        <strong>{data.original_file_name}</strong>
        {/* Prominent "Open in Immich" CTA — accent-coloured, placed
            right after the filename so the user can jump to the
            source photo without scrolling. Hidden when /api/config
            hasn't returned an immich_public_url (no reverse-proxy
            override configured yet). */}
        {immichPublicUrl && (
          <a
            href={`${immichPublicUrl.replace(/\/$/, '')}/photos/${data.asset_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-2.5 py-0.5 rounded bg-accent-dark border border-accent text-white text-[11px] font-medium no-underline hover:bg-accent-hover"
            title="Open this asset in the Immich web UI"
          >
            Open in Immich ↗
          </a>
        )}
      </div>

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
            <a
              href={googleSearchUrl(fullTop1.family, 'plant family')}
              target="_blank"
              rel="noopener noreferrer"
              title={`Search Google for family ${fullTop1.family}`}
              className="rank-family hover:text-accent hover:underline"
            >{fullTop1.family}</a>
          </>
        )}
        {fullTop1?.genus && (
          <>
            <span className="rank-sep">›</span>
            <a
              href={googleSearchUrl(fullTop1.genus, 'plant genus')}
              target="_blank"
              rel="noopener noreferrer"
              title={`Search Google for genus ${fullTop1.genus}`}
              className="rank-genus hover:text-accent hover:underline"
            >{fullTop1.genus}</a>
          </>
        )}
        {fullTop1?.species && (
          <>
            <span className="rank-sep">›</span>
            <a
              href={googleSearchUrl(
                fullTop1.genus ? `${fullTop1.genus} ${fullTop1.species}` : fullTop1.species,
                'plant'
              )}
              target="_blank"
              rel="noopener noreferrer"
              title={`Search Google for ${fullTop1.species}`}
              className="rank-species hover:text-accent hover:underline"
            >{fullTop1.species}</a>
          </>
        )}
        {plantaeScore != null && (
          <span className={`inline-block ml-auto px-2 py-0.5 rounded font-bold text-[11px] font-mono ${plantaeTier}`}>
            Plantae {(plantaeScore * 100).toFixed(1)}%
          </span>
        )}
      </nav>

      {/* "Plants in this image" — the headline answer. Combines the
          whole-image BioCLIP top-K with YOLO bbox-bioclip into a
          single deduped species list. Rendered as the FIRST content
          block under the image so the user gets the answer before
          the full top-K and YOLO debug sections below. */}
      <section className="border border-[#234] rounded-md bg-[#0e1410] p-3 my-3" aria-label="Plants in this image">
        <div className="flex items-baseline justify-between mb-2">
          <h5 className="m-0 text-[0.95rem] text-[#ad9]">Plants in this image</h5>
          {/* Right-side meta: species count only. The "source" pill
              (whole image / yolo / mixed) was removed — the bbox
              highlight on the image above already tells the user
              which pathway produced each row. */}
          {attribution && attribution.source !== 'none' && (
            <span className="text-[10px] text-fg-1 font-mono">
              {attributionRows.length} species
            </span>
          )}
        </div>
        {attributionRows.length === 0 ? (
          <p className="m-0 text-[12px] text-fg-1 italic">
            No plants confidently attributed to this image.
            {attribution?.plantae_score != null && (
              <> Plantae kingdom score {(attribution.plantae_score * 100).toFixed(1)}%.</>
            )}
          </p>
        ) : (
          <ul className="m-0 p-0 list-none">
            {attributionRows.map((r, i) => {
              // A binomial species name already starts with its genus
              // ("Helianthus annuus" → "Helianthus"). Detect this
              // case-insensitively on the first word so we can avoid
              // duplicating it in both the Google query and the
              // sub-line pill.
              const impliedGenus = !!(
                r.genus && r.species &&
                r.species.toLowerCase().split(/\s+/)[0] === r.genus.toLowerCase()
              );
              const showGenusPill = !!r.genus && !impliedGenus;
              // Species search: prepend the genus only when the
              // species doesn't already start with it. Without this
              // guard, "Helianthus annuus" + genus "Helianthus"
              // produces a "Helianthus Helianthus annuus" query,
              // which Google can't find anything for. The "plant"
              // hint keeps the search botanical for vague single-word
              // species.
              const speciesQuery = r.genus && !impliedGenus
                ? `${r.genus} ${r.species}`
                : r.species;
              const speciesSearch = googleSearchUrl(speciesQuery, 'plant');
              return (
                <li
                  key={`${r.species}-${i}`}
                  className="flex items-baseline justify-between gap-2 py-1 border-t border-[#222] first:border-t-0 text-[13px]"
                >
                  <div className="flex-1 min-w-0">
                    <span className="italic font-semibold">{r.species}</span>
                    {r.common_name && (
                      <span className="text-fg-1 text-[11px] ml-1.5">({r.common_name})</span>
                    )}
                    {/* Google search link for the species. The icon
                        is the official multicolored G so users get
                        the same visual cue as in other Google
                        products. target=_blank + rel=noopener keeps
                        the modal open. */}
                    <a
                      href={speciesSearch}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`Search Google for "${r.species}"`}
                      aria-label={`Search Google for ${r.species}`}
                      className="ml-1.5 inline-flex items-center align-baseline opacity-70 hover:opacity-100"
                    >
                      <GoogleIcon className="w-[12px] h-[12px]" />
                    </a>
                    <span className="block text-[11px] text-fg-1 truncate">
                      {/* Skip the genus pill when the binomial already
                          shows it — "Helianthus annuus" doesn't need
                          "genus Helianthus" beneath it. */}
                      {showGenusPill && (
                        <a
                          href={googleSearchUrl(r.genus as string, 'plant genus')}
                          target="_blank"
                          rel="noopener noreferrer"
                          title={`Search Google for genus ${r.genus}`}
                          className="hover:text-accent hover:underline"
                        >genus {r.genus}</a>
                      )}
                      {showGenusPill && r.family && (
                        <span className="mx-1">·</span>
                      )}
                      {r.family && (
                        <a
                          href={googleSearchUrl(r.family, 'plant family')}
                          target="_blank"
                          rel="noopener noreferrer"
                          title={`Search Google for family ${r.family}`}
                          className="hover:text-accent hover:underline"
                        >family {r.family}</a>
                      )}
                      {!showGenusPill && !r.family && (r.kingdom ?? '')}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    <span className="text-[#ad9] font-mono tabular-nums text-[12px] min-w-[42px] text-right">
                      {formatPct(r.score)}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <LibraryNeighbors items={data.library_neighbors} />

      {data.knn?.top_species ? (
        <details className="my-3">
          <summary className="cursor-pointer text-[0.85rem] text-fg-1 hover:text-fg">KNN (re-classified)</summary>
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
        </details>
      ) : null}
      <button type="button" className="bg-[#234] text-fg border border-[#456] px-2.5 py-1 rounded text-[11px] cursor-pointer hover:bg-[#345] mt-1" onClick={handleReclassifyAll}>
        Re-run KNN
      </button>

      {/* Debug panels — top-K BioCLIP and per-crop YOLO details. The
          "Plants in this image" block above already answers the main
          question; these are for users who want to see the raw
          ranking under the hood. Collapsed by default so the modal
          doesn't open with a wall of debug numbers. */}
      <details className="my-3">
        <summary className="cursor-pointer text-[0.85rem] text-fg-1 hover:text-fg">
          Top-{data.top_predictions.length} predictions (debug)
        </summary>
        <div className="pt-0 pb-3 px-3">
          <Predictions items={(data.top_predictions ?? []).slice(0, 5)} />
        </div>
      </details>

      {(data.detections ?? []).filter(d => d.score >= 0.20).length > 0 && (
        <details className="my-3">
          <summary className="cursor-pointer text-[0.85rem] text-fg-1 hover:text-fg">
            YOLO detections (debug, per-crop)
          </summary>
          {(data.detections ?? [])
            .filter(d => d.score >= 0.20)
            .map((d, i) => (
              <div key={d.id}>
                <DetectionBlock det={d} fullTop1={fullTop1} />
                <button type="button" className="bg-[#234] text-fg border border-[#456] px-2.5 py-1 rounded text-[11px] cursor-pointer hover:bg-[#345] mt-1" onClick={() => handleReclassifyBbox(i)}>
                  Reclassify crop
                </button>
              </div>
            ))}
        </details>
      )}
    </div>
  );
}
