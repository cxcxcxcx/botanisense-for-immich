// Browse-list container — owns the FilterCards (chip groups + cards
// grid). All filter UI lives in FilterCards.tsx now; this file is a
// thin wrapper that just renders FilterCards so we keep the App.tsx
// composition surface unchanged.

import { useApp } from '../state/AppContext';
import { FilterCards } from './FilterCards';

export function Toolbar() {
  const { state } = useApp();
  const collapsed = state.filtersCollapsed;

  // collapsed-by-active is handled inside FilterCards (it returns
  // null when there's a sticky active card AND forceShow is false).
  // We still respect it here so the surrounding chrome disappears too.
  if (collapsed) return null;

  return (
    <section className="flex flex-col gap-2 py-2 sm:px-4 border-b border-line2">
      <FilterCards />
    </section>
  );
}