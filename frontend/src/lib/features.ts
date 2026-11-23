// Build-time feature switches. Vite inlines `import.meta.env.VITE_*` as
// literals, so a disabled feature's branches are dead code and its
// lazily-imported modules (and their deps) are dropped from the bundle.

// BotaniSense account integration: sign-in, the Diary tab and "Send to
// BotaniSense". Off by default — the cloud side isn't ready for the
// open-source release. Enable with VITE_BOTANISENSE_ENABLED=1 at build
// time (Docker: `--build-arg VITE_BOTANISENSE_ENABLED=1`).
export const BOTANISENSE_ENABLED =
  import.meta.env.VITE_BOTANISENSE_ENABLED === '1' ||
  import.meta.env.VITE_BOTANISENSE_ENABLED === 'true';
