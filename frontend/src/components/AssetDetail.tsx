import { useEffect, useState } from 'react';
import { getAsset, reclassifyBbox, reclassify, getConfig, getKnnStatus } from '../api/endpoints';
import type { KnnStatus } from '../api/types';
import type { AssetDetail as AssetDetailType } from '../api/types';
import { thumbUrl } from '../lib/thumb';
import { Predictions } from './Predictions';
import { LibraryNeighbors } from './LibraryNeighbors';
import { KnnRefNeighbors } from './KnnRefNeighbors';
import { DetectionBlock } from './DetectionBlock';
import { BboxImage } from './BboxImage';
import { useToasts } from './ToastProvider';
import { formatPct } from '../lib/format';

interface Props {
  assetId: string;
}

// ---------- icon set (inline SVGs; no lucide-react dep) ----------
// Match lucide-react's stroke=2, rounded caps, 24×24 viewBox. Sized
// via the `className` prop so callers control the visual weight.

function IconX({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function IconExternalLink({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function IconCopy({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function IconCheck({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconClock({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function IconChevronRight({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

// Build a Google search URL for a free-form taxonomy term. The query
// keeps the actual name verbatim — quoted ("...") so Google searches
// the exact phrase, and prepended with "plant" so a common English
// word like "Rose" doesn't return recipes / nursery homepages.
function googleSearchUrl(term: string, hint?: string): string {
  const q = hint ? `"${term}" ${hint}` : `"${term}"`;
  return `https://www.google.com/search?q=${encodeURIComponent(q)}`;
}

// Official Google "G" logo as inline SVG. Sized via the `className`.
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

// Small colour-coded chip used in the taxonomy breadcrumb. Pulled
// from the example: each rank gets its own subtle bg/fg pair so the
// eye can scan the hierarchy at a glance.
function TaxChip({
  rank,
  label,
}: {
  rank: 'kingdom' | 'family' | 'genus';
  label: string;
}) {
  const cls = {
    kingdom: 'tax-chip-kingdom',
    family:  'tax-chip-family',
    genus:   'tax-chip-genus',
  }[rank];
  return (
    <span className={`px-2 py-0.5 rounded ${cls} text-[12px] font-medium`}>
      {label}
    </span>
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
  // Public Immich URL for the "Open in Immich ↗" deep-link.
  const [immichPublicUrl, setImmichPublicUrl] = useState<string | null>(null);
  // KNN status — fetched once on mount. When the reference index is
  // built, "Re-run KNN on this asset" appears in the modal so the
  // user can opt into per-asset KNN reclassification. Hidden when
  // KNN isn't configured (the index build is hours-long; a button
  // that always 503s is worse than no button).
  const [knnStatus, setKnnStatus] = useState<KnnStatus | null>(null);
  // Disables the re-run button while a reclassify is in flight. The
  // call is synchronous server-side (FAISS search + parquet take), so
  // without this a double-click queues two identical searches.
  const [knnBusy, setKnnBusy] = useState(false);
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

  useEffect(() => {
    let cancelled = false;
    getKnnStatus()
      .then(s => { if (!cancelled) setKnnStatus(s); })
      .catch(() => { /* silently null — button stays hidden */ });
    return () => { cancelled = true; };
  }, []);

  const handleReclassifyAll = async () => {
    if (knnBusy) return;
    setKnnBusy(true);
    try {
      // force: an explicit click must always recompute. Without it the
      // backend skips any asset that already has a row for
      // (index_version, k) and the click is a silent no-op.
      const r = await reclassify([assetId], undefined, true);
      // Use the row the endpoint already returned instead of refetching
      // the whole asset — /api/reclassify hands back the fresh
      // knn_predictions rows keyed by asset_id for exactly this.
      const row = r.results?.[assetId] ?? null;
      setData(d => (d ? { ...d, knn: row } : d));
      push({
        kind: 'success',
        message: row?.top_species
          ? `KNN → ${row.top_species}`
          : 'KNN found no match for this asset',
      }, 3000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    } finally {
      setKnnBusy(false);
    }
  };

  const handleReclassifyBbox = async (detectionId: number) => {
    if (!knnStatus?.index_built) {
      push({ kind: 'info', message: 'Build the KNN index first (Run page → Build KNN index)' }, 3500);
      return;
    }
    try {
      const r = await reclassifyBbox({ asset_id: assetId, detection_id: detectionId });
      push({
        kind: 'success',
        message: `Crop KNN → ${r.result.top_species ?? '(no match)'} (${(r.result.score * 100).toFixed(1)}%)`,
      }, 3500);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    }
  };

  const handleCopy = async () => {
    if (!data) return;
    try {
      await navigator.clipboard.writeText(data.asset_id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      push({ kind: 'error', message: 'Copy failed: ' + (e instanceof Error ? e.message : String(e)) }, 3000);
    }
  };

  if (error) return <div className="text-red-300">Failed to load asset: {error.message}</div>;
  if (!data) return <p aria-busy="true">Loading…</p>;

  const fullTop1 = data.top_predictions[0];
  const plantaeScore = data.taxon_scores?.Plantae;
  const attribution = data.plant_attribution;
  const attributionRows = attribution?.rows ?? [];
  const highlightIds = attribution?.highlight_detection_ids ?? [];

  return (
    // Compact column, centered within the Modal. Matches the example's
    // visual density (max-w-sm, gap-3.5) so a 380px-wide modal feels
    // like a phone-screen card, not a stretched form.
    <div className="text-[13px] max-w-sm w-full mx-auto flex flex-col gap-3.5">
      {/* Photo + YOLO bboxes. Same BboxImage canvas-based overlay as
          before; the box geometry is unchanged, just the framing is
          narrower so it lines up with the rest of the compact layout. */}
      <div className="flex justify-center">
        <figure className="m-0 inline-block bg-bg-0 rounded-xl overflow-hidden border border-line">
          <BboxImage
            src={thumbUrl(data.asset_id, usePreview ? 'preview' : 'fullsize')}
            detections={data.detections}
            highlightDetectionIds={highlightIds}
            fit="contain"
            maxHeight={224}
            ariaLabel={data.original_file_name}
            onError={() => setUsePreview(true)}
          />
        </figure>
      </div>

      {/* Filename + Open in Immich — side-by-side, filename
          truncates, CTA stays visible. Green-tinted accent matches
          the "Plants in this image" card below. */}
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[14px] font-semibold text-fg truncate" title={data.original_file_name}>
          {data.original_file_name}
        </h3>
        {immichPublicUrl && (
          <a
            href={`${immichPublicUrl.replace(/\/$/, '')}/photos/${data.asset_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 flex items-center gap-1 bg-bg-2 border border-line2 text-fg-1 px-2 py-1 rounded-md text-[12px] font-medium hover:border-accent hover:text-fg transition-colors no-underline"
            title="Open this asset in the Immich web UI"
          >
            Open in Immich
            <IconExternalLink />
          </a>
        )}
      </div>

      {/* Asset ID + Copy ID — monospace ID truncates, copy button
          flips to a checkmark for 1.5s after a successful write. */}
      <div className="flex items-center justify-between gap-2">
        <code
          className="text-[11px] font-mono text-fg-1 truncate"
          title={data.asset_id}
        >{data.asset_id}</code>
        <button
          type="button"
          onClick={handleCopy}
          className="flex-shrink-0 flex items-center gap-1 bg-bg-2 border border-line2 text-fg-1 px-2 py-0.5 rounded-md text-[11px] hover:border-accent transition-colors"
          aria-label="Copy asset ID"
        >
          {copied ? (
            <>
              <span className="text-green-500"><IconCheck /></span>
              <span className="text-green-500">Copied</span>
            </>
          ) : (
            <>
              <IconCopy />
              Copy ID
            </>
          )}
        </button>
      </div>

      {/* Capture time (always shown when available). Single line
          with a small clock icon. */}
      {data.captured_at && (
        <div className="flex items-center text-[12px] text-fg-1 gap-1.5">
          <IconClock />
          <span>{new Date(data.captured_at).toLocaleString()}</span>
        </div>
      )}

      {/* Taxonomy breadcrumb — kingdom › family › genus, with a
          separate row below for the species (italic) + Plantae%.
          Each rank gets its own subtle colour so the hierarchy is
          scannable at a glance. */}
      {(fullTop1?.kingdom || fullTop1?.family || fullTop1?.genus) && (
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
            {fullTop1?.kingdom && <TaxChip rank="kingdom" label={fullTop1.kingdom} />}
            {fullTop1?.family && (
              <>
                <IconChevronRight />
                <a
                  href={googleSearchUrl(fullTop1.family, 'plant family')}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={`Search Google for family ${fullTop1.family}`}
                  className="no-underline"
                ><TaxChip rank="family" label={fullTop1.family} /></a>
              </>
            )}
            {fullTop1?.genus && (
              <>
                <IconChevronRight />
                <a
                  href={googleSearchUrl(fullTop1.genus, 'plant genus')}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={`Search Google for genus ${fullTop1.genus}`}
                  className="no-underline"
                ><TaxChip rank="genus" label={fullTop1.genus} /></a>
              </>
            )}
          </div>
          {fullTop1?.species && (
            <div className="flex items-center justify-between text-[13px] px-1">
              <span className="italic font-medium text-fg truncate" title={fullTop1.species}>
                {fullTop1.species}
              </span>
              {plantaeScore != null && (
                <span className="text-fg-1 font-medium tabular-nums text-[12px]">
                  Plantae {(plantaeScore * 100).toFixed(1)}%
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* "Plants in this image" — green-tinted card matching the
          example. Species count + first row visible; the rest of the
          attribution rows live below as a compact list. */}
      <section className="plants-card p-3 flex flex-col gap-2" aria-label="Plants in this image">
        <div className="flex items-center justify-between">
          <h4 className="plants-title text-[13px] font-medium">Plants in this image</h4>
          {attribution && attribution.source !== 'none' && (
            <span className="plants-count text-[11px]">
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
          <ul className="m-0 p-0 list-none flex flex-col gap-1.5">
            {attributionRows.slice(0, 5).map((r, i) => {
              // Skip genus pill when binomial already shows it.
              const impliedGenus = !!(
                r.genus && r.species &&
                r.species.toLowerCase().split(/\s+/)[0] === r.genus.toLowerCase()
              );
              const speciesQuery = r.genus && !impliedGenus
                ? `${r.genus} ${r.species}`
                : r.species;
              return (
                <li key={`${r.species}-${i}`} className="flex items-start justify-between gap-2 text-[13px]">
                  <div className="flex items-center gap-1.5 min-w-0 flex-1">
                    <a
                      href={googleSearchUrl(speciesQuery, 'plant')}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`Search Google for "${r.species}"`}
                      className="text-[14px] font-bold text-fg italic truncate hover:text-accent no-underline"
                    >{r.species}</a>
                    {r.common_name && (
                      <span className="text-[11px] text-fg-1 shrink-0">({r.common_name})</span>
                    )}
                    <a
                      href={googleSearchUrl(speciesQuery, 'plant')}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={`Search Google for "${r.species}"`}
                      aria-label={`Search Google for ${r.species}`}
                      className="flex items-center justify-center w-3.5 h-3.5 rounded-full bg-white text-black text-[9px] font-bold shrink-0 hover:opacity-80 no-underline"
                    >G</a>
                  </div>
                  <span className="plants-score font-medium tabular-nums text-[13px] shrink-0">
                    {formatPct(r.score)}
                  </span>
                </li>
              );
            })}
            {attributionRows.length > 5 && (
              <li className="text-[11px] text-fg-1 italic">
                + {attributionRows.length - 5} more in Top-K below
              </li>
            )}
          </ul>
        )}
      </section>

      {/* Library neighbours — the example renders a 4-col thumb grid
          here. LibraryNeighbors already does that; the gap above is
          enough visual separation. */}
      <LibraryNeighbors items={data.library_neighbors} />

      {/* Per-asset KNN. Only rendered when the reference index is built
          — fetching /api/knn-status once on mount tells us. Shows the
          consensus species plus the reference photos the vote came
          from, so the user can judge the match rather than take the
          species string on faith. For the whole grid there's
          'Reclassify with KNN' on the Run page. */}
      {knnStatus?.index_built && (
        <section className="flex flex-col gap-2">
          <div className="flex items-baseline justify-between gap-2">
            <h3 className="text-[12px] uppercase tracking-wide text-fg-1">
              KNN match
            </h3>
            <button
              type="button"
              className="text-[11px] text-fg-1 underline hover:text-fg disabled:opacity-50 disabled:no-underline"
              onClick={handleReclassifyAll}
              disabled={knnBusy}
              title="Recompute the KNN match for this asset against the reference index"
            >
              {knnBusy ? 'Running…' : 'Re-run KNN on this asset'}
            </button>
          </div>

          {data.knn?.top_species ? (
            <>
              <div className="text-[13px]">
                {data.knn.top_species}
                {data.knn.common_name ? ` (${data.knn.common_name})` : ''}
                <span className="text-fg-1">
                  {' · '}vote {formatPct(data.knn.score)}
                </span>
              </div>
              {(data.knn.kingdom || data.knn.family) && (
                <div className="text-[11px] text-fg-1">
                  {data.knn.kingdom}
                  {data.knn.family ? ` · ${data.knn.family}` : ''}
                  {data.knn.genus ? ` · ${data.knn.genus}` : ''}
                </div>
              )}
              <KnnRefNeighbors neighbors={data.knn.neighbors ?? []} />
            </>
          ) : (
            <p className="text-[12px] text-fg-1">
              {data.knn
                ? 'KNN ran but found no confident match.'
                : 'No KNN result for this asset yet.'}
            </p>
          )}
        </section>
      )}

      {/* Debug panels — collapsed by default; users who want the raw
          ranking open these. */}
      <details className="-mt-1">
        <summary className="cursor-pointer text-[12px] text-fg-1 hover:text-fg">
          Top-{data.top_predictions.length} predictions (debug)
        </summary>
        <div className="pt-1 pb-2 px-1">
          <Predictions items={(data.top_predictions ?? []).slice(0, 5)} />
        </div>
      </details>

      {(data.detections ?? []).filter(d => d.score >= 0.20).length > 0 && (
        <details className="-mt-1">
          <summary className="cursor-pointer text-[12px] text-fg-1 hover:text-fg">
            YOLO detections (debug, per-crop)
          </summary>
          {(data.detections ?? [])
            .filter(d => d.score >= 0.20)
            .map((d, i) => (
              <div key={d.id}>
                <DetectionBlock det={d} fullTop1={fullTop1} />
                <button
                  type="button"
                  className="self-start text-[11px] text-fg-1 underline hover:text-fg"
                  onClick={() => handleReclassifyBbox(d.id)}
                >
                  Reclassify crop
                </button>
              </div>
            ))}
        </details>
      )}
    </div>
  );
}