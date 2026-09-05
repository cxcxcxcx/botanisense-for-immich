import { Modal } from './Modal';
import { ConnectForm } from './ConnectForm';
import { useApp } from '../state/AppContext';
import type { Health, PublicConfig } from '../api/types';

// "Fix connection" dialog — the edit path for an already-configured
// install. Opened from HealthBanner's Immich blockers and ImmichSync's
// connection row; the Welcome wizard embeds ConnectForm directly since
// it IS the setup flow. Stays open on a failed check so the user can
// correct the values; closes on success.
export function ConnectionModal() {
  const { state, dispatch } = useApp();

  const onSaved = (_config: PublicConfig, health: Health) => {
    if (health.immich?.ok && health.api_key?.ok) {
      dispatch({ type: 'CLOSE_CONNECTION_MODAL' });
    }
  };

  return (
    <Modal
      open={state.connectionModalOpen}
      onClose={() => dispatch({ type: 'CLOSE_CONNECTION_MODAL' })}
      title="Immich connection"
    >
      <p className="text-fg-2 text-[12px] m-0 mb-3">
        Where the server reaches your Immich, and the API key it uses.
        Create a key in Immich → Administration → Settings → API Keys.
      </p>
      <ConnectForm onSaved={onSaved} />
    </Modal>
  );
}
