// Outbound links shown in the header's About menu and the Welcome footer.
import pkg from '../../package.json';

export const APP_VERSION: string = pkg.version;

export const REPO_URL = 'https://github.com/cxcxcxcx/botanisense-for-immich';
export const BOTANISENSE_APP_URL = 'https://botanisense.app';

// GitHub issue forms pre-fill a field from a query param named after its
// `id` (.github/ISSUE_TEMPLATE/bug.yml has `id: env`), so bug reports
// arrive with the version already filled in.
export function bugReportUrl(detail?: string): string {
  const env = `BotaniSense for Immich v${APP_VERSION}${detail ? ` (${detail})` : ''}`;
  return `${REPO_URL}/issues/new?template=bug.yml&env=${encodeURIComponent(env)}`;
}

export const FEEDBACK_URL = `${REPO_URL}/issues/new/choose`;
