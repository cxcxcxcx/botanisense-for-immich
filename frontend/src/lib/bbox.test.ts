import { describe, expect, it } from 'vitest';
import type { Detection } from '../api/types';
import { mergeDetectionsForDisplay } from './bbox';

function det(
  id: number,
  label: string,
  x: number,
  y: number,
  w = 0.1,
  h = 0.1,
  score = 0.8,
): Detection {
  return { id, label, score, x, y, w, h };
}

describe('mergeDetectionsForDisplay', () => {
  it('merges adjacent same-label flower boxes into one union box', () => {
    const out = mergeDetectionsForDisplay([
      det(1, 'flower', 0.1, 0.1, 0.1, 0.1, 0.7),
      det(2, 'flower', 0.22, 0.1, 0.1, 0.1, 0.9),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0]!.id).toBe(2); // best member wins the click target
    expect(out[0]!.group_count).toBe(2);
    expect(out[0]!.member_ids.sort()).toEqual([1, 2]);
    expect(out[0]!.x).toBeCloseTo(0.1);
    expect(out[0]!.w).toBeCloseTo(0.22);
  });

  it('does not merge distant flowers', () => {
    const out = mergeDetectionsForDisplay([
      det(1, 'flower', 0.05, 0.05),
      det(2, 'flower', 0.75, 0.75),
    ]);
    expect(out).toHaveLength(2);
    expect(out.every(d => d.group_count === 1)).toBe(true);
  });

  it('does not merge different labels even when they overlap', () => {
    const out = mergeDetectionsForDisplay([
      det(1, 'flower', 0.1, 0.1),
      det(2, 'plant', 0.1, 0.1),
    ]);
    expect(out).toHaveLength(2);
  });

  it('leaves person/hand/pot boxes individual', () => {
    const out = mergeDetectionsForDisplay([
      det(1, 'person', 0.1, 0.1),
      det(2, 'person', 0.12, 0.1),
      det(3, 'hand', 0.14, 0.1),
      det(4, 'pot', 0.16, 0.1),
    ]);
    expect(out).toHaveLength(4);
  });

  it('leaves adjacent whole-plant boxes individual', () => {
    // A garden bed: neighbouring plants, each boxed once by the detector.
    const out = mergeDetectionsForDisplay([
      det(1, 'plant', 0.1, 0.1),
      det(2, 'plant', 0.22, 0.1),
      det(3, 'plant', 0.1, 0.15),
      det(4, 'tree', 0.5, 0.5),
      det(5, 'tree', 0.55, 0.5),
    ]);
    expect(out).toHaveLength(5);
    expect(out.every(d => d.group_count === 1)).toBe(true);
  });

  it('merges a chain transitively', () => {
    const out = mergeDetectionsForDisplay([
      det(1, 'flower', 0.1, 0.1),
      det(2, 'flower', 0.22, 0.1),
      det(3, 'flower', 0.34, 0.1),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0]!.group_count).toBe(3);
    expect(out[0]!.w).toBeCloseTo(0.34);
  });
});
