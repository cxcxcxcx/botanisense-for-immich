import type { Pagination as PaginationType } from '../api/types';
import { useFilters } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';

/** Build the page list with ellipsis around the current page. */
function buildPageList(current: number, total: number): (number | 'ellipsis')[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i);
  }
  const out: (number | 'ellipsis')[] = [0];
  const window = 1;
  const left = Math.max(1, current - window);
  const right = Math.min(total - 2, current + window);
  if (left > 1) out.push('ellipsis');
  for (let i = left; i <= right; i++) out.push(i);
  if (right < total - 2) out.push('ellipsis');
  out.push(total - 1);
  return out;
}

interface Props {
  data: PaginationType;
}

export function Pagination({ data }: Props) {
  const filters = useFilters();
  const setFilters = useUrlFilters();
  if (data.total_pages <= 1) return null;
  const page = data.page;
  const total = data.total_pages;
  const pages = buildPageList(page, total);
  const goTo = (p: number) => setFilters({ page: p });
  return (
    <nav className="pagination-bar" aria-label="Pagination">
      <span>{data.total.toLocaleString()} photos · page {page + 1} / {total}</span>
      <span className="pages">
        <button type="button" className="pg-btn" disabled={page === 0} onClick={() => goTo(0)}>«</button>
        <button type="button" className="pg-btn" disabled={page === 0} onClick={() => goTo(page - 1)}>‹</button>
        {pages.map((p, i) =>
          p === 'ellipsis'
            ? <span key={`e${i}`} className="ellipsis">…</span>
            : <button
                key={p}
                type="button"
                className={`pg-btn ${p === page ? 'on' : ''}`}
                onClick={() => goTo(p)}
              >
                {p + 1}
              </button>
        )}
        <button type="button" className="pg-btn" disabled={page >= total - 1} onClick={() => goTo(page + 1)}>›</button>
        <button type="button" className="pg-btn" disabled={page >= total - 1} onClick={() => goTo(total - 1)}>»</button>
      </span>
    </nav>
  );
}