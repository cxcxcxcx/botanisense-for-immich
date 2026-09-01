import { useEffect, useRef, type ReactNode } from 'react';

interface Props {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
}

export function Modal({ open, onClose, title, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="max-w-[min(92vw,900px)] max-h-[92vh] bg-bg-1 text-fg border border-line2 rounded-lg p-0"
      onClose={onClose}
      onClick={e => {
        if (e.target === ref.current) onClose();
      }}
    >
      <article className="m-0">
        {/* Header padding bumped from `mb-2 pb-2` to `mb-4 py-4 px-6`
            so it visually matches the body's `px-6 py-4`. Subtle but
            gives the title bar breathing room next to the dense
            asset-detail content below. */}
        <header className="flex items-center justify-between mb-4 border-b border-line py-4 px-6">
          <h3 className="m-0 flex-1">{title || 'Asset detail'}</h3>
          <button
            className="text-fg-1 hover:text-fg text-lg leading-none px-2"
            aria-label="Close"
            onClick={onClose}
          >×</button>
        </header>
        <div className="overflow-auto max-h-[70vh] px-6 py-4">{children}</div>
      </article>
    </dialog>
  );
}