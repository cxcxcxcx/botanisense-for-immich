import type { Detection } from '../api/types';

/** A detection group ready to draw. `id`/`score` come from the best
 * member, so clicking or classifying still targets a real detection. */
export interface DisplayDetection extends Detection {
  member_ids: number[];
  group_count: number;
}

/** Plant-PART labels that may be merged when their boxes are adjacent:
 * several flower heads or leaves are usually one plant. Whole-plant
 * labels (plant/tree/succulent/…) are never merged — the trained
 * detectors box one plant per box, and the transitive adjacency rule
 * chained a whole succulent garden into a single "plant ×15" box.
 * Person/hand/face/pot stay individual too. */
const MERGE_LABELS = new Set([
  'flower',
  'bloom',
  'petal',
  'leaf',
  'stem',
  'branch',
]);

/** How much empty space (relative to the largest member box) can sit
 * between two same-label boxes before they are considered one cluster.
 * 0.25 means a quarter of the box size; this is deliberately generous
 * for bouquets where YOLO boxes each flower head separately. */
const NEAR_PAD_RATIO = 0.25;

function boxesAreNear(a: Detection, b: Detection): boolean {
  const pad = NEAR_PAD_RATIO * Math.max(a.w, a.h, b.w, b.h);
  return (
    a.x - pad < b.x + b.w &&
    a.x + a.w + pad > b.x &&
    a.y - pad < b.y + b.h &&
    a.y + a.h + pad > b.y
  );
}

/**
 * Collapse same-label, adjacent detections into one visual box.
 *
 * This is intentionally a *display* merge, not a database rewrite: the
 * per-crop detections remain available in the debug list and in the box
 * similarity index. The union box's score/id are taken from the best
 * member so the existing click → Similar flow keeps working.
 */
export function mergeDetectionsForDisplay(
  detections: Detection[],
): DisplayDetection[] {
  const n = detections.length;
  if (n === 0) return [];

  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => {
    let root = i;
    while (parent[root]! !== root) root = parent[root]!;
    while (parent[i]! !== i) {
      const next = parent[i]!;
      parent[i] = root;
      i = next;
    }
    return root;
  };
  const union = (i: number, j: number): void => {
    const ri = find(i);
    const rj = find(j);
    if (ri !== rj) parent[rj] = ri;
  };

  for (let i = 0; i < n; i += 1) {
    const a = detections[i]!;
    if (!MERGE_LABELS.has(a.label)) continue;
    for (let j = i + 1; j < n; j += 1) {
      const b = detections[j]!;
      if (a.label !== b.label) continue;
      if (boxesAreNear(a, b)) union(i, j);
    }
  }

  const groups = new Map<number, number[]>();
  for (let i = 0; i < n; i += 1) {
    const root = find(i);
    const group = groups.get(root);
    if (group) group.push(i);
    else groups.set(root, [i]);
  }

  const merged: DisplayDetection[] = [];
  for (const members of groups.values()) {
    let bestIndex = members[0]!;
    for (const index of members) {
      if (detections[index]!.score > detections[bestIndex]!.score) {
        bestIndex = index;
      }
    }
    const best = detections[bestIndex]!;
    const x0 = Math.min(...members.map(i => detections[i]!.x));
    const y0 = Math.min(...members.map(i => detections[i]!.y));
    const x1 = Math.max(...members.map(i => detections[i]!.x + detections[i]!.w));
    const y1 = Math.max(...members.map(i => detections[i]!.y + detections[i]!.h));
    merged.push({
      ...best,
      x: x0,
      y: y0,
      w: x1 - x0,
      h: y1 - y0,
      member_ids: members.map(i => detections[i]!.id),
      group_count: members.length,
    });
  }

  // Draw large regions first so a small tight box stays visible on top.
  merged.sort(
    (a, b) => b.w * b.h - a.w * a.h || b.score - a.score,
  );
  return merged;
}
