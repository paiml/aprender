# Testing Philosophy: Zero External Dependencies

> **CRITICAL DESIGN CONSTRAINT:** Zero external dependencies. No playwright. No selenium. No puppeteer. No npm. No C bindings. Pure Rust + WASM only.

This is **non-negotiable**.

## Why No External Dependencies?

External browser automation tools introduce:

| Problem | Impact |
|---------|--------|
| Security vulnerabilities | C/C++ codebases with CVEs |
| Non-deterministic behavior | Flaky tests, CI failures |
| Platform-specific failures | "Works on my machine" |
| Dependency hell | Version conflicts, breaking updates |
| License contamination | GPL/LGPL transitive deps |
| Bundle bloat | 200MB+ for browser automation |

## What We Build Instead

`presentar-test` is a first-party testing framework built on Trueno primitives:

```
┌─────────────────────────────────────────────────────────────────┐
│  presentar-test (Pure Rust, ~50KB)                              │
├─────────────────────────────────────────────────────────────────┤
│  TestRunner                                                     │
│  ├── Discovers #[presentar_test] functions                      │
│  ├── Spawns isolated App instances                              │
│  └── Collects results, generates reports                        │
├─────────────────────────────────────────────────────────────────┤
│  Harness                                                        │
│  ├── Event simulation (click, type, scroll, keyboard)           │
│  ├── Widget queries (CSS-like selectors)                        │
│  ├── State inspection                                           │
│  └── Async waiting (for animations, data loading)               │
├─────────────────────────────────────────────────────────────────┤
│  Framebuffer                                                    │
│  ├── Software rasterizer (Trueno SIMD)                          │
│  ├── Pixel capture for visual regression                        │
│  ├── Fixed DPI (1.0) for determinism                            │
│  └── PNG encode/decode (pure Rust)                              │
├─────────────────────────────────────────────────────────────────┤
│  A11yChecker                                                    │
│  ├── WCAG 2.1 AA rule engine                                    │
│  ├── Contrast ratio (Trueno color math)                         │
│  └── Focus order, ARIA validation                               │
└─────────────────────────────────────────────────────────────────┘
```

## Line Count Breakdown

| Component | Lines | Dependencies |
|-----------|-------|--------------|
| Selector parser | ~100 | None |
| PNG encode/decode | ~300 | Trueno |
| Pixel diff | ~50 | Trueno SIMD |
| WCAG contrast | ~30 | Trueno color |
| Event simulation | ~200 | None |
| **Total** | **~700** | **Zero external** |

## Determinism Guarantees

To ensure pixel-perfect reproducibility across platforms:

- **Fixed DPI**: `1.0` (no system scaling)
- **Font antialiasing**: Grayscale only (no subpixel/ClearType)
- **Fixed viewport**: `1280x720` default
- **No system fonts**: Embedded test font (Inter, Apache 2.0)

## Why Not Alternatives?

| Tool | Problem |
|------|---------|
| playwright | npm + Chrome DevTools Protocol + 200MB |
| selenium | Java + WebDriver + non-deterministic |
| wasm-bindgen-test | Requires browser install, no event sim |
| cypress | npm + Electron + 500MB |

## Example Test

```rust
use presentar_test::*;

#[presentar_test]
fn app_renders() {
    let h = Harness::new(include_bytes!("fixtures/app.tar"));

    // Widget exists
    h.assert_exists("[data-testid='app-root']");

    // Text content
    h.assert_text("[data-testid='title']", "My App");

    // Click interaction
    h.click("[data-testid='submit-btn']");
    h.assert_text_contains("[data-testid='result']", "Success");
}

#[presentar_test]
fn visual_regression() {
    let mut h = Harness::new(include_bytes!("fixtures/app.tar"));
    Snapshot::assert_match("app-default", h.screenshot("[data-testid='app-root']"), 0.001);
}

#[presentar_test]
fn accessibility() {
    let h = Harness::new(include_bytes!("fixtures/app.tar"));
    A11yChecker::check(&h.app).assert_pass();
}
```

## Toyota Way: Jidoka

> **Jidoka (Automation with a human touch):** Build quality in, don't inspect it out.

By owning the entire test stack, we:
- Control determinism completely
- Catch regressions at compile time
- Eliminate "works in CI but not locally" issues
- Maintain sub-second test execution

## Next Steps

- [Test Harness](./test-harness.md) - Harness API reference
- [Selectors](./selectors.md) - CSS-like query syntax
- [Visual Regression](./visual-regression.md) - Snapshot testing
