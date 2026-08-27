import { Modal } from './Modal';

interface Props {
  open: boolean;
  onClose: () => void;
}

// Modal that explains why a feature is flagged beta. Used by the
// cluster-sync beta tags on the Run page and the cluster-album bar.
//
// Currently one shared copy block (cluster sync). If more beta
// features land, take a `kind` prop and switch on it.

export function BetaInfoModal({ open, onClose }: Props) {
  return (
    <Modal open={open} onClose={onClose} title="Beta: cluster sync">
      <div className="flex flex-col gap-3 text-[0.85rem] leading-relaxed">
        <p className="m-0">
          Cluster-sync is marked <strong>beta</strong> for two reasons:
        </p>
        <ul className="m-0 pl-5 list-disc">
          <li>
            <strong>Album titles can go stale after a cluster rename.</strong>
            {' '}We create a new album when a cluster first syncs, but we don't
            currently <code className="font-mono">PATCH</code> the title when
            only the cluster name changes. Re-sync to refresh.
          </li>
          <li>
            <strong>Removing a mapping always deletes the Immich album.</strong>
            {' '}There's no "leave orphan" option — bulk-remove is destructive
            on the Immich side too. You can re-sync to recreate.
          </li>
        </ul>
        <p className="m-0">
          Workarounds in the meantime: re-sync after cluster renames, and
          always double-check the confirmation modal before bulk remove.
        </p>
        <p className="m-0 text-fg-1 text-[12px]">
          Track open issues or report a regression:{' '}
          <a
            href="https://github.com/cxcxcxcx/botanisense-for-immich/issues?q=is%3Aopen+label%3Acluster-sync-beta"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#7ad] hover:underline"
          >
            github.com/.../issues?q=label:cluster-sync-beta ↗
          </a>
        </p>
      </div>
    </Modal>
  );
}