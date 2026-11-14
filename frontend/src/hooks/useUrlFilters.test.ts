import { describe, expect, it } from 'vitest';

import { buildUrl, parseUrlFilters } from './useUrlFilters';

// Map click-through plumbing: a place is a four-part URL contract
// (`near_lat`/`near_lng`/`near_radius` + `place`/`place_gran`/`place_min`).
// A partial triple must never become a silent geographic filter.
describe('map place URL filters', () => {
  it('round-trips a map place', () => {
    const f = parseUrlFilters(
      '?group_by=cluster&near_lat=37.7749&near_lng=-122.4194&near_radius=42' +
        '&place=loc-wide-3&place_gran=wide&place_min=1',
    );

    expect(f.near_lat).toBeCloseTo(37.7749);
    expect(f.near_lng).toBeCloseTo(-122.4194);
    expect(f.near_radius).toBe(42);
    expect(f.place_id).toBe('loc-wide-3');
    expect(f.place_granularity).toBe('wide');
    expect(f.place_min_photos).toBe(1);

    const url = buildUrl(f);
    expect(url).toContain('near_lat=37.7749');
    expect(url).toContain('near_lng=-122.4194');
    expect(url).toContain('near_radius=42');
    expect(url).toContain('place=loc-wide-3');
    expect(url).toContain('place_gran=wide');
    expect(url).toContain('place_min=1');
  });

  it('defaults the radius/min and ignores a place without coordinates', () => {
    const ok = parseUrlFilters('?near_lat=10&near_lng=20&place=loc-1');
    expect(ok.near_radius).toBe(150);
    expect(ok.place_min_photos).toBe(2);
    expect(ok.place_granularity).toBe('fine');

    const noCoords = parseUrlFilters('?place=loc-1&place_gran=wide');
    expect(noCoords.near_lat).toBeUndefined();
    expect(noCoords.place_id).toBeUndefined();
  });

  it('a taxonomy card URL does not carry a stale place', () => {
    const f = parseUrlFilters('?group_by=cluster&cluster_id=7');
    expect(f.place_id).toBeUndefined();
    expect(buildUrl(f)).not.toContain('place=');
  });
});
