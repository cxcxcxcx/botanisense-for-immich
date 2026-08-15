import { useEffect, useState } from 'react';
import { useFilters, type GroupBy, type SortMode } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { useDebounce } from '../hooks/useDebounce';
import { FilterCards } from './FilterCards';

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
        <label>group by
          <select
            value={filters.group_by}
            onChange={e => setFilters({ group_by: e.target.value as GroupBy, page: 0 })}
          >
            <option value="species">species</option>
            <option value="genus">genus</option>
            <option value="family">family</option>
            <option value="cluster">cluster (HDBSCAN)</option>
          </select>
        </label>
        <label>sort
          <select
            value={filters.sort}
            onChange={e => setFilters({ sort: e.target.value as SortMode })}
          >
            <option value="rank">search rank</option>
            <option value="plantae">plantae confidence</option>
            <option value="score">top-1 score</option>
          </select>
        </label>
        <input
          type="search"
          placeholder="search filters…"
          autoComplete="off"
          value={q}
          onChange={e => setQ(e.target.value)}
        />
      </fieldset>
      <FilterCards />
    </form>
  );
}