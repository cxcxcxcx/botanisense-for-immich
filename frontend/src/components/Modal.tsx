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
      className="detail-dialog"
      onClose={onClose}
      onClick={e => {
        // Click on backdrop (the dialog itself, not its children) closes.
        if (e.target === ref.current) onClose();
      }}
    >
      <article>
        <header>
          <button className="close" rel="prev" aria-label="Close" onClick={onClose}>✕</button>
          <h3>{title || 'Asset detail'}</h3>
        </header>
        <div className="modal-body">{children}</div>
      </article>
    </dialog>
  );
}