// Plantae kingdom-confidence coloring and the legend bar in the toolbar.
export function plantaeClass(score, green, yellow) {
  if (score == null) return 'red';
  if (score >= green) return 'green';
  if (score >= yellow) return 'yellow';
  return 'red';
}

export function renderPlantaeLegend(data) {
  const legend = document.getElementById('plantae-legend');
  const g = (data.plantae_green ?? 0.60).toFixed(2);
  const y = (data.plantae_yellow ?? 0.35).toFixed(2);
  legend.innerHTML =
    `Plantae ` +
    `<span class="plantae green">≥ ${g}</span>` +
    `<span class="plantae yellow">≥ ${y}</span>` +
    `<span class="plantae red">&lt; ${y}</span>` +
    ` (BioCLIP kingdom-rank confidence)`;
}
