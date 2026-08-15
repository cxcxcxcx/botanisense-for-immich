import { useEffect, useState } from 'react';
import { useApp, useFilters, type GroupBy, type SortMode } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { useDebounce } from '../hooks/useDebounce';
import { FilterCards } from './FilterCards';

const GROUP_BY_CARDS: { value: GroupBy; label: string }[] = [
  { value: 'cluster', label: 'cluster' },
  { value: 'species', label: 'species' },
  { value: 'genus', label: 'genus' },
  { value: 'family', label: 'family' },
];

// Sort options for the photo grid. The label is the user-facing phrase;
// the hint is shown beneath it (or in a tooltip on narrow screens) so
// users understand what each sort actually does without trying it.
const SORT_CARDS: { value: SortMode; label: string; hint: string }[] = [
  { value: 'rank',    label: 'default',        hint: 'order returned by the server' },
  { value: 'plantae', label: 'plants first',   hint: 'highest Plantae confidence on top' },
  { value: 'score',   label: 'most confident', hint: 'highest BioCLIP top-1 score on top' },
];

export function Toolbar() {
  const filters = useFilters();
  const { state } = useApp();
  const setFilters = useUrlFilters();
  const debouncedQ = useDebounce(filters.q, 300);
  const [q, setQ] = useState(filters.q);

  // Push the debounced value into URL state. We use an effect (not a
  // render-time setState) to avoid the "setState during render" warning
  // and to batch the dispatch with any other pending state.
  useEffect(() => {
    if (debouncedQ !== filters.q) {
      setFilters({ q: debouncedQ, page: 0 });
    }
  }, [debouncedQ, filters.q, setFilters]);

  // When a filter is active and FilterCards is showing only the
  // collapsed active-bar, hide the group-by card row + sort row too —
  // the active-bar already carries the cluster/taxon identity, so
  // showing the chrome at the same time reads as duplicate UI.
  const collapsed = state.filtersCollapsed;

  return (
    <form className="toolbar" onSubmit={e => e.preventDefault()}>
      {!collapsed && (
        <fieldset role="group" className="toolbar-fieldset">
          <div className="toolbar-control">
            <div className="toolbar-control-label">group by</div>
            <div className="group-by-cards" role="radiogroup" aria-label="Group by">
              {GROUP_BY_CARDS.map(g => (
                <button
                  key={g.value}
                  type="button"
                  role="radio"
                  aria-checked={filters.group_by === g.value}
                  className={`group-by-card ${filters.group_by === g.value ? 'active' : ''}`}
                  onClick={() => setFilters({ group_by: g.value, page: 0 })}
                  title={`Bucket photos by ${g.label}`}
                >
                  {g.label}
                </button>
              ))}
            </div>
          </div>
          <div className="toolbar-control">
            <div className="toolbar-control-label">sort photos by</div>
            <div className="sort-cards" role="radiogroup" aria-label="Sort photos by">
              {SORT_CARDS.map(s => (
                <button
                  key={s.value}
                  type="button"
                  role="radio"
                  aria-checked={filters.sort === s.value}
                  className={`sort-card ${filters.sort === s.value ? 'active' : ''}`}
                  onClick={() => setFilters({ sort: s.value })}
                  title={s.hint}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </fieldset>
      )}
      <FilterCards />
      {!collapsed && (
        <div className="filter-search-row">
          <input
            type="search"
            className="filter-search-input"
            placeholder="filter cards by name…"
            autoComplete="off"
            value={q}
            onChange={e => setQ(e.target.value)}
            title="Hide cards whose label doesn't match (sub-filter; doesn't change the active filter)"
          />
          <small className="filter-search-hint">
            Filters the visible cards by name — your active {filters.group_by} filter is unchanged.
            Use the ✕ on the active card or the <em>back to filters</em> button to clear it.
          </small>
        </div>
      )}
    </form>
  );
}
