import { useEffect, useState } from 'react';
import { useFilters, type GroupBy, type SortMode } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { useDebounce } from '../hooks/useDebounce';
import { FilterCards } from './FilterCards';

const GROUP_BY_CARDS: { value: GroupBy; label: string }[] = [
  { value: 'cluster', label: 'cluster' },
  { value: 'species', label: 'species' },
  { value: 'genus', label: 'genus' },
  { value: 'family', label: 'family' },
];

export function Toolbar() {
  const filters = useFilters();
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

  return (
    <form className="toolbar" onSubmit={e => e.preventDefault()}>
      <fieldset role="group" className="toolbar-fieldset">
        <div className="group-by-cards" role="radiogroup" aria-label="Group by">
          {GROUP_BY_CARDS.map(g => (
            <button
              key={g.value}
              type="button"
              role="radio"
              aria-checked={filters.group_by === g.value}
              className={`group-by-card ${filters.group_by === g.value ? 'active' : ''}`}
              onClick={() => setFilters({ group_by: g.value, page: 0 })}
            >
              {g.label}
            </button>
          ))}
        </div>
        <label className="toolbar-sort">sort
          <select
            value={filters.sort}
            onChange={e => setFilters({ sort: e.target.value as SortMode })}
          >
            <option value="rank">search rank</option>
            <option value="plantae">plantae confidence</option>
            <option value="score">top-1 score</option>
          </select>
        </label>
      </fieldset>
      <FilterCards />
      <input
        type="search"
        className="filter-search-input"
        placeholder="search filters…"
        autoComplete="off"
        value={q}
        onChange={e => setQ(e.target.value)}
        title="Filter the visible cards by name"
      />
    </form>
  );
}