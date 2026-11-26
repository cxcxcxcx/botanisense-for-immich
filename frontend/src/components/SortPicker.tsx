// SortPicker — sort control for the Level-2 back-bars. Styled like the
// map/filter triggers: a square icon button on phones, and from `sm:` up
// the same button widened with the current sort label + chevron. The
// native <select> sits invisibly on top, so a click still opens the OS
// picker without the browser's own select chrome showing.

interface Props<T extends string> {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}

export function SortPicker<T extends string>({ value, options, onChange }: Props<T>) {
  const current = options.find(o => o.value === value)?.label ?? '';
  return (
    <span className="filter-trigger sm:w-auto sm:px-3 sm:gap-1.5">
      <SortIcon />
      <span className="hidden sm:inline text-xs font-medium whitespace-nowrap">{current}</span>
      <ChevronIcon />
      <select
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        aria-label="Sort photos"
        title={`Sort: ${current}`}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </span>
  );
}

function SortIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 4v16M3 16l4 4 4-4" />
      <path d="M17 20V4M13 8l4-4 4 4" />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg className="hidden sm:block opacity-60" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}
