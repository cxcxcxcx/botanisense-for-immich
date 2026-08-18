// Filter the global /api/events stream down to one specific job. The
// server pushes every active job in the snapshot; the consumer (e.g.
// JobPanel, KnnBuildModal) wants just the slice that matches its
// jobId. Replaces the per-job REST poll loops each modal used to run.

import { useEventStream } from './useEventStream';
import type { EventSnapshot, JobSnapshot } from '../api/types';

export type JobRegistryKey =
  | 'plants_jobs'
  | 'cluster_album_jobs'
  | 'kingdom_jobs'
  | 'clustering_jobs'
  | 'knn_build_jobs'
  | 'backfill_exif_jobs';

export function useJobStream<T extends JobSnapshot = JobSnapshot>(
  registryKey: JobRegistryKey,
  jobId: string | null,
): { job: T | null; ready: boolean } {
  const { data } = useEventStream<EventSnapshot>('/api/events');
  if (!jobId) return { job: null, ready: data !== null };
  const registry = data?.[registryKey];
  const job = (registry?.[jobId] as T | undefined) ?? null;
  return { job, ready: data !== null };
}