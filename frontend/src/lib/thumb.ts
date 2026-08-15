// Build a /thumb/{asset_id} URL. The Vite dev server proxies /thumb to
// uvicorn; in prod FastAPI serves it directly. Same code path either way.
//
// Sizes:
//   - 'preview'   ~1440px — used by card grid (default).
//   - 'thumbnail' ~250px  — used by LibraryNeighbors + near-dup strip.
//   - 'fullsize'  alias for preview — used by the modal so the caller
//     can express intent without knowing which Immich sizes actually
//     exist. The backend coerces unknown sizes to preview.

export type ThumbSize = 'preview' | 'thumbnail' | 'fullsize';

export function thumbUrl(aid: string, size?: ThumbSize): string {
  const enc = encodeURIComponent(aid);
  return size ? `/thumb/${enc}?size=${size}` : `/thumb/${enc}`;
}