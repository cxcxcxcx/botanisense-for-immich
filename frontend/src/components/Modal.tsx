import { useEffect, useRef, useState, type ReactNode } from 'react';

interface Props {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  /** Wider dialog for content like the scoped map. */
  wide?: boolean;
}

export function Modal({ open, onClose, title, children, wide = false }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  // Children are mounted one commit *after* showModal(). Anything that
  // measures its container on mount (Leaflet, in the scoped map dialog)
  // otherwise initialises against a display:none dialog at 0×0 and draws
  // zero-radius markers that never recover.
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
    setReady(open);
  }, [open]);

  return (
    <dialog
      ref={ref}
      className={`${wide ? 'w-[min(96vw,1200px)] max-w-[min(96vw,1200px)]' : 'max-w-[min(92vw,900px)]'} max-h-[92vh] bg-bg-1 text-fg border border-line2 rounded-lg p-0`}
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
        <div className={`overflow-auto max-h-[70vh] ${wide ? 'px-2 py-2' : 'px-6 py-4'}`}>
          {ready ? children : null}
        </div>
      </article>
    </dialog>
  );
}