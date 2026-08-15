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
          {/* Pico.css paints the X via `background-image: var(--pico-icon-close)`
             on any .close / [rel=prev] inside <dialog><article>. The button
             must therefore have empty content; a literal "✕" would stack on
             top of the SVG and render the close glyph twice. */}
          <button className="close" rel="prev" aria-label="Close" onClick={onClose} />
          <h3>{title || 'Asset detail'}</h3>
        </header>
        <div className="modal-body">{children}</div>
      </article>
    </dialog>
  );
}