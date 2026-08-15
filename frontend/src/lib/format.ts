export const formatPct = (x: number, digits = 2): string =>
  `${(x * 100).toFixed(digits)}%`;

export function fmtDuration(s: number): string {
  if (!Number.isFinite(s) || s < 0) return '?';
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s - m * 60);
  return `${m}m${sec.toString().padStart(2, '0')}s`;
}

export function cls(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}