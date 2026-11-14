import { Modal } from './Modal';

interface Props {
  open: boolean;
  onClose: () => void;
}

const SHORTCUTS: Array<{ key: string; label: string; desc: string }> = [
  { key: 'b', label: 'b', desc: 'Browse view' },
  { key: 'm', label: 'm', desc: 'Map view' },
  { key: 'r', label: 'r', desc: 'Run / Sync view' },
  { key: '?', label: '?', desc: 'Show this shortcuts help' },
  { key: 'Escape', label: 'Esc', desc: 'Close any open modal' },
];

// Tiny modal listing every keyboard shortcut in the app. Opened by
// pressing `?` (wired in App.tsx via useKeyboardShortcuts) or by the
// header help button (future).

export function ShortcutsHelpModal({ open, onClose }: Props) {
  return (
    <Modal open={open} onClose={onClose} title="Keyboard shortcuts">
      <table className="w-full text-[0.85rem]">
        <thead>
          <tr className="text-fg-1">
            <th className="text-left font-normal pb-2 w-[80px]">Key</th>
            <th className="text-left font-normal pb-2">Action</th>
          </tr>
        </thead>
        <tbody>
          {SHORTCUTS.map(s => (
            <tr key={s.key} className="border-t border-[#222]">
              <td className="py-2 pr-3">
                <kbd className="font-mono text-[12px] px-1.5 py-0.5 rounded border border-[#456] bg-bg-2 text-fg">
                  {s.label}
                </kbd>
              </td>
              <td className="py-2 text-fg-1">{s.desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="m-0 mt-3 text-[11px] text-fg-2">
        Shortcuts are disabled while typing in an input or while a
        modal is open.
      </p>
    </Modal>
  );
}