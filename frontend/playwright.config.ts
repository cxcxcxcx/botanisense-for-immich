import { defineConfig } from '@playwright/test';

// E2E against the REAL app (uvicorn serving the built SPA) and a tiny
// fake Immich. No model, no photo library, no Immich instance needed:
// the suite covers first boot, onboarding, connection save, health,
// routing and theme persistence — the failures a unit test can't see.
//
// By default Playwright starts both servers itself. To test an already
// running artifact (e.g. the release image), set E2E_BASE_URL and only
// the fake Immich is started:
//
//   docker run --rm -d --name bs-e2e -p 8799:8765 --tmpfs /app/data \
//     --add-host=host.docker.internal:host-gateway botanisense:rc
//   E2E_BASE_URL=http://127.0.0.1:8799 \
//   E2E_FAKE_IMMICH_URL=http://host.docker.internal:2284 \
//     npm run e2e
//
// (Use a FRESH container/DB: the first test asserts "unconfigured".)

const APP_PORT = Number(process.env.E2E_APP_PORT ?? 8799);
const IMMICH_PORT = Number(process.env.E2E_IMMICH_PORT ?? 2284);
const externalBaseURL = process.env.E2E_BASE_URL?.trim() || undefined;

type WebServer = {
  command: string;
  url: string;
  cwd?: string;
  reuseExistingServer?: boolean;
  timeout?: number;
};

const webServers: WebServer[] = [
  {
    command: `python3 fake_immich.py --host 0.0.0.0 --port ${IMMICH_PORT}`,
    cwd: './e2e',
    url: `http://127.0.0.1:${IMMICH_PORT}/api/server/ping`,
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
];

if (!externalBaseURL) {
  // Fresh DB per run; empty IMMICH_* shadow the developer's .env so the
  // first-run wizard is actually visible (dotenv never overrides a
  // variable that is already set, even to "").
  webServers.unshift({
    command:
      'npm run build && cd .. && ' +
      'mkdir -p "${TMPDIR:-/tmp}/opencode" && ' +
      'DATA=$(mktemp -d "${TMPDIR:-/tmp}/opencode/botanisense-e2e.XXXXXX") && ' +
      'exec env IMMICH_BASE_URL= IMMICH_API_KEY= DB_PATH="$DATA/state.db" ' +
      `uv run uvicorn app:app --host 127.0.0.1 --port ${APP_PORT}`,
    cwd: '.',
    url: `http://127.0.0.1:${APP_PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 180_000,
  });
}

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI
    ? [['github'], ['html', { open: 'never' }]]
    : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: externalBaseURL ?? `http://127.0.0.1:${APP_PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: webServers,
});
