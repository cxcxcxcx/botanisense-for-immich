// FilterModal — bottom-sheet on mobile, centered card on `sm:` and up.
// Triggered by the Filter button on Level 1 (browse list). Owns the
// Plantae tier chips and the Person presence chips. Closing the modal
// with Apply commits the in-memory state back to the URL filter store.

import { useEffect, useState } from 'react';
import { useFilters } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import type { Filters } from '../state/AppContext';

interface Props {
  open: boolean;
  onClose: () => void;
}

const PLANTAE_OPTS: Array<{ v: Filters['plantae']; label: string; dotClass: string; title: string }> = [
  { v: undefined,    label: 'all',        dotClass: 'bg-fg-2',       title: 'Show plants + uncertain (Plantae ≥ 0.35). Flip "Show non-plants" below to include non-plants.' },
  { v: 'plant',      label: 'plants',     dotClass: 'tier-dot tier-dot-green',  title: 'Plantae ≥ 0.60' },
  { v: 'uncertain',  label: 'uncertain',  dotClass: 'tier-dot tier-dot-yellow', title: 'Plantae 0.35–0.60' },
  { v: 'non_plant',  label: 'non-plants', dotClass: 'tier-dot tier-dot-red',    title: 'Plantae < 0.35 (or unknown)' },
];

const PERSON_OPTS: Array<{ v: Filters['has_person']; label: string; title: string }> = [
  { v: undefined, label: 'any', title: 'Show all photos (scanned + unscanned)' },
  { v: 1,          label: 'has', title: 'Show only photos with a person detected' },
  { v: 0,          label: 'no',  title: 'Show only photos scanned and confirmed without a person' },
];

export function FilterModal({ open, onClose }: Props) {
  const filters = useFilters();
  const setFilters = useUrlFilters();

  // Local draft state — only committed on Apply. Lets the user tweak
  // multiple dimensions and review before committing, and makes
  // "Clear all" trivially revertible.
  const [plantae, setPlantae] = useState<Filters['plantae']>(filters.plantae);
  const [hasPerson, setHasPerson] = useState<Filters['has_person']>(filters.has_person);
  const [includeNonPlants, setIncludeNonPlants] = useState<boolean>(
    Boolean(filters.include_non_plants),
  );

  // Sync from outside whenever the modal opens (in case the URL
  // changed via back/forward or another component).
  useEffect(() => {
    if (open) {
      setPlantae(filters.plantae);
      setHasPerson(filters.has_person);
      setIncludeNonPlants(Boolean(filters.include_non_plants));
    }
  }, [open, filters.plantae, filters.has_person, filters.include_non_plants]);

  if (!open) return null;

  const handleClearAll = () => {
    setPlantae(undefined);
    setHasPerson(undefined);
    setIncludeNonPlants(false);
  };

  const handleApply = () => {
    setFilters({
      plantae,
      has_person: hasPerson,
      include_non_plants: includeNonPlants || undefined,
      page: 0,
    });
    onClose();
  };

  return (
    <div className="filter-modal-overlay" role="dialog" aria-modal="true" aria-label="Filters">
      <div
        className="filter-modal-backdrop"
        onClick={onClose}
        aria-hidden="true"
      />
      <div className="filter-modal-card">
        <div className="filter-modal-header">
          <h3 className="filter-modal-title">
            <FilterIcon />
            Advanced Filters
          </h3>
          <button
            type="button"
            onClick={onClose}
            className="filter-modal-close"
            aria-label="Close filters"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="filter-modal-body">
          <div className="filter-modal-section">
            <label className="filter-modal-section-label">Plants category</label>
            <div className="filter-modal-chips">
              {PLANTAE_OPTS.map(opt => (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => setPlantae(opt.v)}
                  className={plantae === opt.v ? 'modal-chip modal-chip-active' : 'modal-chip modal-chip-inactive'}
                  title={opt.title}
                >
                  <span className={opt.dotClass} aria-hidden="true" style={{ width: 6, height: 6, borderRadius: 9999 }} />
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-modal-section">
            <label className="filter-modal-section-label">Person presence</label>
            <div className="filter-modal-chips">
              {PERSON_OPTS.map(opt => (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => setHasPerson(opt.v)}
                  className={hasPerson === opt.v ? 'modal-chip modal-chip-active' : 'modal-chip modal-chip-inactive'}
                  title={opt.title}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-modal-section">
            <label className="filter-modal-section-label">Non-plant visibility</label>
            <label className="modal-toggle-row" title={includeNonPlants ? 'Non-plants ARE included — grid & counts reflect every photo.' : 'Non-plants are hidden — grid & counts only show Plantae ≥ 0.35 (the project-wide policy). Flip on to include every photo in the all-view.'}>
              <input
                type="checkbox"
                checked={includeNonPlants}
                onChange={(e) => setIncludeNonPlants(e.target.checked)}
                className="modal-toggle-checkbox"
              />
              <span className="modal-toggle-text">
                Show non-plants in All view
              </span>
            </label>
          </div>
        </div>

        <div className="filter-modal-footer">
          <button
            type="button"
            onClick={handleClearAll}
            className="filter-modal-clear"
          >
            Clear all
          </button>
          <button
            type="button"
            onClick={handleApply}
            className="filter-modal-apply"
          >
            Apply Filters
          </button>
        </div>
      </div>
    </div>
  );
}

function FilterIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent" aria-hidden="true">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}