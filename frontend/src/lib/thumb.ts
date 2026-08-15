// Build a /thumb/{asset_id} URL. The Vite dev server proxies /thumb to
// uvicorn; in prod FastAPI serves it directly. Same code path either way.

export function thumbUrl(aid: string, size?: 'preview' | 'thumbnail' | 'fullsize'): string {
  const enc = encodeURIComponent(aid);
  return size ? `/thumb/${enc}?size=${size}` : `/thumb/${enc}`;
}