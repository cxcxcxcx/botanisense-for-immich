import { expect, test, type Page } from '@playwright/test';

// One stateful flow, in order: boot unconfigured → reject a bad URL →
// save a working connection → verify health → navigate/theme. Serial
// mode + one worker keeps the shared backend database deterministic;
// a failure skips the rest instead of cascading.
test.describe.configure({ mode: 'serial' });

const IMMICH_URL =
  process.env.E2E_FAKE_IMMICH_URL?.trim() || 'http://127.0.0.1:2284';
const API_KEY = 'e2e-api-key';

// ConnectForm prefills the URL from /api/config on mount. Filling
// before that response lands lets the prefill overwrite the input, so
// wait for it explicitly.
async function openFresh(page: Page): Promise<void> {
  await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/config') && r.ok()),
    page.goto('/'),
  ]);
}

test.describe('first-run journey', () => {
  test('boots unconfigured and shows the setup wizard', async ({
    page,
    request,
  }) => {
    const pageErrors: string[] = [];
    page.on('pageerror', e => pageErrors.push(e.message));

    await page.goto('/');
    await expect(
      page.getByRole('heading', { name: 'Welcome to BotaniSense' }),
    ).toBeVisible();
    await expect(page.getByText('Step 1 / 2')).toBeVisible();
    await expect(
      page.getByRole('button', { name: 'Save & test connection' }),
    ).toBeVisible();

    const health = await (await request.get('/api/health')).json();
    expect(health.db.ok).toBe(true);
    expect(health.immich.ok).toBe(false);
    expect(health.engine.id).toBe('teacher');
    expect(pageErrors).toEqual([]);
  });

  test('rejects an invalid Immich URL', async ({ page }) => {
    await openFresh(page);
    // The HealthBanner also renders a (hidden) ConnectForm, so scope the
    // CSS locators to the wizard that is actually on screen.
    const wizard = page.getByRole('region', { name: 'Setup wizard' });
    await wizard.locator('input[autocomplete="url"]').fill('not-a-url');
    await wizard.locator('input[type="password"]').fill(API_KEY);
    await wizard
      .getByRole('button', { name: 'Save & test connection' })
      .click();

    await expect(
      wizard.getByText(/Enter a plain http\(s\) URL/),
    ).toBeVisible();
    await expect(page.getByText('Step 1 / 2')).toBeVisible();
  });

  test('saves a working connection and advances the wizard', async ({
    page,
  }) => {
    await openFresh(page);
    const wizard = page.getByRole('region', { name: 'Setup wizard' });
    await wizard.locator('input[autocomplete="url"]').fill(IMMICH_URL);
    await wizard.locator('input[type="password"]').fill(API_KEY);
    await wizard
      .getByRole('button', { name: 'Save & test connection' })
      .click();

    await expect(wizard.getByText(/Connected/)).toBeVisible({
      timeout: 10_000,
    });
    await wizard
      .getByRole('button', { name: 'Next: start first run' })
      .click();
    await expect(page.getByText('Step 2 / 2')).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Start first run' }),
    ).toBeVisible();
  });

  test('reports the saved connection through /api/health', async ({
    request,
  }) => {
    const health = await (await request.get('/api/health')).json();
    expect(health.immich.ok).toBe(true);
    expect(health.immich.version).toBe('e2e-1.0.0');
    expect(health.api_key.ok).toBe(true);
    expect(health.db.ok).toBe(true);
  });

  test('navigates to the run page and persists the theme', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('tab', { name: 'Run / Sync' }).click();

    await expect(
      page.getByRole('region', { name: 'Pipeline', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('button', { name: 'Run Pipeline' }),
    ).toBeVisible();

    const html = page.locator('html');
    const before = await html.getAttribute('data-theme');
    const want = before === 'dark' ? 'light' : 'dark';
    await page
      .getByRole('button', { name: /Switch to (light|dark) theme/ })
      .click();
    await expect(html).toHaveAttribute('data-theme', want);

    // The inline bootstrap in index.html must restore it on reload.
    await page.reload();
    await expect(html).toHaveAttribute('data-theme', want);
  });

  test('map tab renders with an empty library', async ({ page }) => {
    const pageErrors: string[] = [];
    page.on('pageerror', e => pageErrors.push(e.message));

    await page.goto('/');
    await page.getByRole('tab', { name: 'Map' }).click();

    await expect(
      page.getByRole('region', { name: 'Photo map' }),
    ).toBeVisible();
    // The empty DB has no geotagged assets, so the map must explain that
    // instead of rendering a blank tile sheet.
    await expect(page.getByText('No geotagged photos')).toBeVisible({
      timeout: 10_000,
    });
    expect(pageErrors).toEqual([]);
  });
});
