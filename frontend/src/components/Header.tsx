import { KnnPill } from './KnnPill';
import { StatusPanel } from './StatusPanel';
import { LegendBar } from './LegendBar';
import { useApp } from '../state/AppContext';

export function Header() {
  const { dispatch } = useApp();
  return (
    <header className="app-header">
      <hgroup>
        <h1>Immich × BioCLIP</h1>
        <KnnPill />
        <div className="header-buttons">
          <button
            type="button"
            className="secondary outline header-btn"
            onClick={() => dispatch({ type: 'OPEN_PARAMS_MODAL' })}
            title="Pipeline parameters"
          >
            ⚙ params
          </button>
          <button
            type="button"
            className="secondary outline header-btn"
            onClick={() => dispatch({ type: 'OPEN_LOGS_MODAL' })}
            title="Server logs"
          >
            📋 logs
          </button>
        </div>
      </hgroup>
      <StatusPanel />
      <LegendBar />
    </header>
  );
}