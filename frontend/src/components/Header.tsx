import { KnnPill } from './KnnPill';
import { StatusPanel } from './StatusPanel';
import { ParamsForm } from './ParamsForm';
import { LogTail } from './LogTail';
import { LegendBar } from './LegendBar';

export function Header() {
  return (
    <header className="app-header">
      <hgroup>
        <h1>Immich × BioCLIP</h1>
        <KnnPill />
      </hgroup>
      <StatusPanel />
      <ParamsForm />
      <LogTail />
      <LegendBar />
    </header>
  );
}