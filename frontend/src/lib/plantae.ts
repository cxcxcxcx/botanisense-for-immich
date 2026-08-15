export type PlantaeTier = 'green' | 'yellow' | 'red';

/** Bucket the Plantae confidence score into one of three display tiers.
 *  Matches the legacy CSS: `.plantae.green/yellow/red`. */
export function plantaeTier(score: number | null | undefined,
                           green = 0.60, yellow = 0.35): PlantaeTier {
  if (score == null) return 'red';
  if (score >= green) return 'green';
  if (score >= yellow) return 'yellow';
  return 'red';
}

export const plantaeColorClass = plantaeTier;