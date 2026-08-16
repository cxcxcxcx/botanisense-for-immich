import { useEffect, useState } from 'react';
import { useApp, useFilters, type GroupBy } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { useDebounce } from '../hooks/useDebounce';
import { FilterCards } from './FilterCards';

const GROUP_BY_CARDS: { value: GroupBy; label: string }[] = [
  { value: 'cluster', label: 'cluster' },
  { value: 'species', label: 'species' },
  { value: 'genus', label: 'genus' },
  { value: 'family', label: 'family' },
];

// Toolbar owns the cluster/taxon list-view chrome. Photo-sort and
// pagination live with the photo grid (Grid.tsx).
export function Toolbar() {
  const filters = useFilters();
  const { state } = useApp();
  const setFilters = useUrlFilters();
  const debouncedQ = useDebounce(filters.q, 300);
  const [q, setQ] = useState(filters.q);

  useEffect(() => {
    if (debouncedQ !== filters.q) {
      setFilters({ q: debouncedQ, page: 0 });
    }
  }, [debouncedQ, filters.q, setFilters]);

  const collapsed = state.filtersCollapsed;

  return (
    <form className="flex flex-col gap-2 py-1 px-4 border-b border-line2" onSubmit={e => e.preventDefault()}>
      {!collapsed && (
        <fieldset role="group" className="flex flex-wrap items-center gap-2 border-0 p-0 m-0">
          <div className="inline-flex gap-[3px] p-[3px] bg-bg-1 border border-line2 rounded-lg box-border items-center" role="radiogroup" aria-label="Group by">
            {GROUP_BY_CARDS.map(g => (
              <button
                key={g.value}
                type="button"
                role="radio"
                aria-checked={filters.group_by === g.value}
                className={`bg-transparent border-0 ${
                  filters.group_by === g.value
                    ? 'bg-accent-dark text-white'
                    : 'text-fg-0 hover:text-fg'
                } rounded h-[22px] leading-[22px] inline-flex items-center justify-center px-3 text-[0.85rem] cursor-pointer lowercase m-0`}
                onClick={() => setFilters({ group_by: g.value, page: 0 })}
                title={`Bucket photos by ${g.label}`}
              >{g.label}</button>
            ))}
          </div>
        </fieldset>
      )}
      <FilterCards />
      {!collapsed && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <input
            type="search"
            className="bg-bg-2 text-fg border border-[#333] px-2.5 py-[5px] rounded text-[0.85rem] min-w-[200px] max-w-[320px]"
            placeholder="filter cards by name…"
            autoComplete="off"
            value={q}
            onChange={e => setQ(e.target.value)}
            title="Hide cards whose label doesn't match (sub-filter; doesn't change the active filter)"
          />
          <small className="text-fg-1 text-[0.75rem] flex-1 basis-[200px]">
            Filters the visible cards by name — your active {filters.group_by} filter is unchanged.
            Click a card's ✕ or the <em className="text-fg-0 not-italic font-medium">back</em> link in the photo grid to clear it.
          </small>
        </div>
      )}
    </form>
  );
}