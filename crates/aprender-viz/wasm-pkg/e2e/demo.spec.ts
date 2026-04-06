import { test, expect } from '@playwright/test';

test.describe('trueno-viz WASM Demo', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for WASM to initialize
    await page.waitForFunction(() => {
      return document.getElementById('compute-badge')?.textContent !== 'Detecting...';
    }, { timeout: 10000 });
  });

  test('loads and displays compute tier', async ({ page }) => {
    const badge = page.locator('#compute-badge');
    await expect(badge).toBeVisible();

    const text = await badge.textContent();
    // JS-side detection: "WebGPU Compute", "SIMD128 Compute", or "Scalar Compute"
    expect(text).toMatch(/(WebGPU|SIMD128|Scalar) Compute/);
  });

  test('renders scatter plot canvas', async ({ page }) => {
    const canvas = page.locator('#scatter-canvas');
    await expect(canvas).toBeVisible();

    // Check canvas has content (non-zero pixels)
    const hasContent = await page.evaluate(() => {
      const canvas = document.getElementById('scatter-canvas') as HTMLCanvasElement;
      const ctx = canvas.getContext('2d');
      if (!ctx) return false;
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      return data.some(v => v !== 0);
    });
    expect(hasContent).toBe(true);
  });

  test('scatter animation works', async ({ page }) => {
    const btn = page.locator('#scatter-btn');
    await expect(btn).toHaveText('Start Animation');

    await btn.click();
    await expect(btn).toHaveText('Stop Animation');

    // Let it animate for a bit (stats update every 1s)
    await page.waitForTimeout(1500);

    // Check performance stats show throughput
    const perf = page.locator('#scatter-perf');
    const perfText = await perf.textContent();
    expect(perfText).toContain('pts/s');

    await btn.click();
    await expect(btn).toHaveText('Start Animation');
  });

  test('histogram renders', async ({ page }) => {
    const canvas = page.locator('#hist-canvas');
    await expect(canvas).toBeVisible();
  });

  test('histogram streaming works', async ({ page }) => {
    const btn = page.locator('#hist-btn');
    await btn.click();

    await page.waitForTimeout(500);

    const perf = page.locator('#hist-perf');
    const text = await perf.textContent();
    expect(text).toContain('samples');

    // Stop streaming
    await btn.click();
  });

  test('benchmarks run successfully', async ({ page }) => {
    const runBtn = page.getByRole('button', { name: 'Run All Benchmarks' });
    await runBtn.click();

    // Wait for benchmarks to complete
    await page.waitForFunction(() => {
      const rows = document.querySelectorAll('#benchmark-body tr');
      return rows.length > 1 && !rows[0].textContent?.includes('Running');
    }, { timeout: 30000 });

    // Check results are populated
    const rows = page.locator('#benchmark-body tr');
    const count = await rows.count();
    expect(count).toBeGreaterThan(3);

    // Check throughput values are shown
    const values = page.locator('#benchmark-body .value');
    const firstValue = await values.first().textContent();
    expect(firstValue).toMatch(/\d+\.\d+ M\/s/);
  });

  test('point count selector works', async ({ page }) => {
    const select = page.locator('#scatter-count');
    await select.selectOption('50000');

    // Trigger re-render by starting/stopping animation
    const btn = page.locator('#scatter-btn');
    await btn.click();
    await page.waitForTimeout(100);
    await btn.click();

    const perf = page.locator('#scatter-perf');
    const text = await perf.textContent();
    expect(text).toBeTruthy();
  });

  test('stats update in hero section', async ({ page }) => {
    // Start animation to generate stats
    await page.locator('#scatter-btn').click();

    // Wait for stats to update
    await page.waitForTimeout(1500);

    const totalPoints = page.locator('#total-points');
    const text = await totalPoints.textContent();
    expect(text).not.toBe('-');
    expect(text).toContain('M');

    await page.locator('#scatter-btn').click();
  });

  test('physics simulation renders', async ({ page }) => {
    const canvas = page.locator('#physics-canvas');
    await expect(canvas).toBeVisible();

    // Check canvas has balls rendered (non-zero pixels)
    const hasContent = await page.evaluate(() => {
      const canvas = document.getElementById('physics-canvas') as HTMLCanvasElement;
      const ctx = canvas.getContext('2d');
      if (!ctx) return false;
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      return data.some(v => v !== 0);
    });
    expect(hasContent).toBe(true);
  });

  test('physics simulation runs', async ({ page }) => {
    const btn = page.locator('#physics-btn');
    await expect(btn).toHaveText('Start Simulation');

    await btn.click();
    await expect(btn).toHaveText('Stop Simulation');

    // Let it run
    await page.waitForTimeout(500);

    // Check FPS counter is updating
    const fps = page.locator('#physics-fps');
    const fpsText = await fps.textContent();
    expect(fpsText).not.toBe('0 FPS');

    // Check perf stats
    const perf = page.locator('#physics-perf');
    const perfText = await perf.textContent();
    expect(perfText).toContain('balls');
    expect(perfText).toContain('checks');

    await btn.click();
    await expect(btn).toHaveText('Start Simulation');
  });

  test('physics shake button works', async ({ page }) => {
    // Start simulation
    await page.locator('#physics-btn').click();
    await page.waitForTimeout(300);

    // Click shake
    await page.locator('button:has-text("Shake!")').click();

    // Simulation should still be running
    await expect(page.locator('#physics-btn')).toHaveText('Stop Simulation');

    // Stop
    await page.locator('#physics-btn').click();
  });

  test('physics ball count selector works', async ({ page }) => {
    const select = page.locator('#physics-count');
    await select.selectOption('1000');

    // Start simulation to verify count
    await page.locator('#physics-btn').click();
    await page.waitForTimeout(300);

    const perf = page.locator('#physics-perf');
    const text = await perf.textContent();
    expect(text).toContain('1000 balls');

    await page.locator('#physics-btn').click();
  });

  test('screenshot: initial state', async ({ page }) => {
    // Wait for initial render to complete
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('demo-initial.png', {
      fullPage: true,
      animations: 'disabled',
      maxDiffPixelRatio: 0.10, // Allow 10% variance - physics balls are random
    });
  });

  test('screenshot: with animation', async ({ page }) => {
    // Start animation
    await page.locator('#scatter-btn').click();
    await page.waitForTimeout(1000);

    // Stop animation for stable screenshot
    await page.locator('#scatter-btn').click();
    await page.waitForTimeout(200);

    await expect(page).toHaveScreenshot('demo-animated.png', {
      fullPage: true,
      animations: 'disabled',
      maxDiffPixelRatio: 0.10, // Allow 10% variance
    });
  });
});
