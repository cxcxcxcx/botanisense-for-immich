// Reusable confirm dialog. Replaces native `window.confirm()` so the
// destructive-action flow matches the rest of the app's modal styling
// (keyboard, focus trap, theming). Drives from the same `<Modal>`
// wrapper, so the existing backdrop-click-to-close and Esc-to-close
// behaviour carry over for free.

import { Modal } from './Modal';

interface Props {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  // Plain-text body is fine for the few existing call sites (cluster
  // album delete). If we later need rich content, swap this for
  // ReactNode and pass JSX from the caller.
  body: string;
  confirmLabel?: string;
  cancelLabel?: string;
  // When true, the confirm button uses the danger colour and reads
  // "Remove" instead of "Confirm" by default.
  danger?: boolean;
}

export function ConfirmModal({
  open,
  onClose,
  onConfirm,
  title,
  body,
  confirmLabel,
  cancelLabel = 'Cancel',
  danger = false,
}: Props) {
  const confirmText = confirmLabel ?? (danger ? 'Remove' : 'Confirm');
  const confirmCls = danger
    ? 'px-2.5 py-1 rounded bg-[#7a2a2a] border border-[#a44] text-white text-sm hover:bg-[#a44]'
    : 'px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover';
  return (
    <Modal open={open} onClose={onClose} title={title}>
      <div className="flex flex-col gap-3 text-[0.85rem]">
        <p className="m-0">{body}</p>
        <footer className="flex gap-2 mt-2 justify-end">
          <button
            type="button"
            className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]"
            onClick={onClose}
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            className={confirmCls}
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            {confirmText}
          </button>
        </footer>
      </div>
    </Modal>
  );
}