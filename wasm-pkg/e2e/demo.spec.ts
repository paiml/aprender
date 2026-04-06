import { test, expect } from '@playwright/test';

test.describe('trueno-db WASM Demo', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for WASM to initialize (compute badge updates after init)
    await page.waitForFunction(() => {
      const badge = document.getElementById('compute-badge');
      return badge && !badge.textContent?.includes('Detecting');
    }, { timeout: 15000 });
  });

  test('loads and displays compute tier', async ({ page }) => {
    const badge = page.locator('#compute-badge');
    await expect(badge).toBeVisible();
    const text = await badge.textContent();
    // Should show one of: WebGPU, SIMD128, or Scalar
    expect(text).toMatch(/(WebGPU|SIMD128|Scalar)/i);
  });

  test('detects compute tier correctly', async ({ page }) => {
    const badge = page.locator('#compute-badge');
    await expect(badge).toBeVisible();

    const text = await badge.textContent();
    // Should show one of: WEBGPU, SIMD128, or SCALAR with "Compute"
    expect(text).toMatch(/(WebGPU|SIMD128|Scalar) Compute/i);
  });

  test('shows compute badge with correct styling', async ({ page }) => {
    const badge = page.locator('#compute-badge');
    await expect(badge).toBeVisible();

    // Badge should have one of the tier classes
    const className = await badge.getAttribute('class');
    expect(className).toMatch(/(webgpu|simd128|scalar)/);
  });

  test('sql textarea is editable', async ({ page }) => {
    const sql = page.locator('#sql');
    await expect(sql).toBeVisible();

    // Clear and type new query
    await sql.fill('SELECT playerID, yearID FROM batting WHERE HR > 50');
    const value = await sql.inputValue();
    expect(value).toContain('SELECT playerID, yearID FROM batting');
  });

  test('execute query button works after loading data', async ({ page }) => {
    // First load data
    const loadBtn = page.getByRole('button', { name: 'Load Data' });
    await loadBtn.click();

    // Wait for load to complete
    await page.waitForFunction(() => {
      const btn = document.getElementById('load-btn');
      return btn?.textContent?.includes('Loaded');
    }, { timeout: 10000 });

    // Execute query
    const executeBtn = page.getByRole('button', { name: 'Execute Query' });
    await executeBtn.click();
    await page.waitForTimeout(500);

    const results = page.locator('#results');
    const text = await results.textContent();
    // Should show results with row data
    expect(text).toBeTruthy();
    expect(text?.length).toBeGreaterThan(0);
    expect(text).not.toContain('Please load data first');
  });

  test('load data button triggers stats display', async ({ page }) => {
    const loadBtn = page.getByRole('button', { name: 'Load Data' });
    await expect(loadBtn).toBeVisible();

    await loadBtn.click();

    // Wait for stats to populate
    await page.waitForFunction(() => {
      const count = document.getElementById('stat-count');
      return count && count.textContent !== '-';
    }, { timeout: 10000 });

    // Check stats are displayed
    const count = await page.locator('#stat-count').textContent();
    expect(count).not.toBe('-');
    expect(parseInt(count || '0')).toBeGreaterThan(0);

    const hrMean = await page.locator('#stat-hr-mean').textContent();
    expect(hrMean).not.toBe('-');
  });

  test('page has correct title', async ({ page }) => {
    await expect(page).toHaveTitle('Trueno-DB Browser Analytics');
  });

  test('scatter plot canvas exists', async ({ page }) => {
    const canvas = page.locator('#scatter-canvas');
    await expect(canvas).toBeVisible();
  });

  test('pre-built query buttons exist', async ({ page }) => {
    await expect(page.getByRole('button', { name: 'Top Home Runs' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Top Batting Avg' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Total HR' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Avg BA' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'All Data' })).toBeVisible();
  });

  test('pre-built query button updates sql textarea', async ({ page }) => {
    const sql = page.locator('#sql');

    // Click "Total HR" button
    await page.getByRole('button', { name: 'Total HR' }).click();

    const value = await sql.inputValue();
    expect(value).toContain('SUM(HR)');
  });

  test('screenshot: initial state', async ({ page }) => {
    // Wait for full initialization
    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot('demo-initial.png', {
      fullPage: true,
      animations: 'disabled',
      maxDiffPixelRatio: 0.10,
    });
  });

  test('screenshot: after query', async ({ page }) => {
    // Load data first
    await page.getByRole('button', { name: 'Load Data' }).click();
    await page.waitForFunction(() => {
      const btn = document.getElementById('load-btn');
      return btn?.textContent?.includes('Loaded');
    }, { timeout: 10000 });

    // Execute query
    await page.getByRole('button', { name: 'Execute Query' }).click();
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('demo-after-query.png', {
      fullPage: true,
      animations: 'disabled',
      maxDiffPixelRatio: 0.10,
    });
  });
});
