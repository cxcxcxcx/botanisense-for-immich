// Build a /thumb/{asset_id} URL. The Vite dev server proxies /thumb to
// uvicorn; in prod FastAPI serves it directly. Same code path either way.
//
// Sizes (immich's two sizes — see backend _ALLOWED_SIZES):
//   - 'thumbnail' ~250px  — used by the card grid + near-dup strip.
//     Cards render at ~200-300px on screen, so a 1440px preview was
//     ~20-50× more bytes than needed and the dominant cause of slow
//     grid paint. Stick to thumbnail here unless the caller actually
//     wants full-screen quality.
//   - 'preview'   ~1440px — used by LibraryNeighbors (the modal-style
//     strip inside asset-detail) and any place that needs large.
//   - 'fullsize'  alias for preview — used by the asset-detail modal
//     so the caller can express intent without knowing which Immich
//     sizes actually exist. The backend coerces unknown sizes to
//     preview.

export type ThumbSize = 'preview' | 'thumbnail' | 'fullsize';

export function thumbUrl(aid: string, size?: ThumbSize): string {
  const enc = encodeURIComponent(aid);
  return size ? `/thumb/${enc}?size=${size}` : `/thumb/${enc}`;
}