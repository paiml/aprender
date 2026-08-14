# Why Probar?

Probar was created as a complete replacement for Playwright in the Jugar ecosystem.

## The Problem with Playwright

1. **JavaScript Dependency**: Playwright requires Node.js and npm
2. **Browser Overhead**: Must download and run Chromium
3. **Black Box Testing**: Can only inspect DOM, not game state
4. **CI Complexity**: Requires browser installation in CI
5. **Violates Zero-JS**: Contradicts Jugar's core constraint

## What Probar Solves

### Zero JavaScript

```
Before (Playwright):
├── package.json
├── node_modules/
├── tests/
│   └── pong.spec.ts  ← TypeScript!
└── playwright.config.ts

After (Probar):
└── tests/
    └── probar_pong.rs  ← Pure Rust!
```

### Direct State Access

**Playwright** treats the game as a black box:
```typescript
// Can only check DOM
await expect(page.locator('#score')).toHaveText('10');
```

**Probar** can inspect game state directly:
```rust
// Direct access to game internals
let score = platform.get_game_state().score;
assert_eq!(score, 10);

// Check entity positions
for entity in platform.query_entities::<Ball>() {
    assert!(entity.position.y < 600.0);
}
```

### Deterministic Testing

**Playwright**: Non-deterministic due to browser timing
```typescript
// Flaky! Timing varies between runs
await page.waitForTimeout(100);
await expect(ball).toBeVisible();
```

**Probar**: Fully deterministic
```rust
// Exact frame control
for _ in 0..100 {
    platform.advance_frame(1.0 / 60.0);
}
let ball_pos = platform.get_ball_position();
assert_eq!(ball_pos, expected_pos);  // Always passes
```

### Simpler CI

**Playwright CI**:
```yaml
- name: Install Node.js
  uses: actions/setup-node@v3
- name: Install dependencies
  run: npm ci
- name: Install Playwright
  run: npx playwright install chromium
- name: Run tests
  run: npm test
```

**Probar CI**:
```yaml
- name: Run tests
  run: cargo test
```

## Feature Comparison

| Feature | Playwright | Probar |
|---------|-----------|--------|
| Language | TypeScript | Pure Rust |
| Browser required | Yes | No |
| Game state access | DOM only | Direct |
| Deterministic | No | Yes |
| CI setup | Complex | Simple |
| Frame control | Approximate | Exact |
| Memory inspection | No | Yes |
| Replay support | No | Yes |
| Fuzzing | No | Yes |

## Migration Example

### Before (Playwright)

```typescript
import { test, expect } from '@playwright/test';

test('ball bounces off walls', async ({ page }) => {
  await page.goto('http://localhost:8080');

  // Wait for game to load
  await page.waitForSelector('#game-canvas');

  // Simulate gameplay
  await page.waitForTimeout(2000);

  // Check score changed (indirect verification)
  const score = await page.locator('#score').textContent();
  expect(parseInt(score)).toBeGreaterThan(0);
});
```

### After (Probar)

```rust
#[test]
fn ball_bounces_off_walls() {
    let mut platform = WebPlatform::new_for_test(WebConfig::default());

    // Advance exactly 120 frames (2 seconds at 60fps)
    for _ in 0..120 {
        platform.advance_frame(1.0 / 60.0);
    }

    // Direct state verification
    let state = platform.get_game_state();
    assert!(state.ball_bounces > 0, "Ball should have bounced");
    assert!(state.score > 0, "Score should have increased");
}
```

## Performance Comparison

| Metric | Playwright | Probar |
|--------|-----------|--------|
| Test startup | ~3s | ~0.1s |
| Per-test overhead | ~500ms | ~10ms |
| 39 tests total | ~45s | ~3s |
| CI setup time | ~2min | 0 |
| Memory usage | ~500MB | ~50MB |

## When to Use Each

### Use Probar for:
- Unit tests
- Integration tests
- Deterministic replay
- Fuzzing
- Performance benchmarks
- CI/CD pipelines

### Use Browser Testing for:
- Visual regression (golden master)
- Cross-browser compatibility
- Real user interaction testing
- Production smoke tests
