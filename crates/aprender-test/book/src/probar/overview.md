# Probar: WASM-Native Game Testing

**Probar** (Spanish: "to test/prove") is a pure Rust testing framework for WASM games that provides full Playwright feature parity while adding WASM-native capabilities.

## Installation

| Crate | Purpose | Install |
|-------|---------|---------|
| **jugar-probar** | Library for writing tests | `cargo add jugar-probar --dev` |
| **probador** | CLI tool | `cargo install probador` |

## Why Probar?

| Aspect | Playwright | Probar |
|--------|-----------|--------|
| **Language** | TypeScript | Pure Rust |
| **Browser** | Required (Chromium) | Not needed |
| **Game State** | Black box (DOM only) | Direct API access |
| **CI Setup** | Node.js + browser | Just `cargo test` |
| **Zero JS** | ❌ Violates constraint | ✅ Pure Rust |

## Key Features

### Playwright Parity
- CSS, text, testid, XPath, role-based locators
- All standard assertions (visibility, text, count)
- All actions (click, fill, type, hover, drag)
- Auto-waiting with configurable timeouts
- Network interception and mobile emulation

### WASM-Native Extensions
- **Zero-copy memory views** - Direct WASM memory inspection
- **Type-safe entity selectors** - Compile-time verified game object access
- **Deterministic replay** - Record inputs with seed, replay identically
- **Invariant fuzzing** - Concolic testing for game invariants
- **Frame-perfect timing** - Fixed timestep control
- **WCAG accessibility** - Color contrast and photosensitivity checking

## Quick Example

```rust
use jugar_probar::Assertion;
use jugar_web::{WebConfig, WebPlatform};

#[test]
fn test_game_starts() {
    let config = WebConfig::new(800, 600);
    let mut platform = WebPlatform::new_for_test(config);

    // Run a frame
    let output = platform.frame(0.0, "[]");

    // Verify output
    assert!(output.contains("commands"));

    // Use Probar assertions
    let assertion = Assertion::in_range(60.0, 0.0, 100.0);
    assert!(assertion.passed);
}
```

## Running Tests

```bash
# All Probar E2E tests
cargo test -p jugar-web --test probar_pong

# Verbose output
cargo test -p jugar-web --test probar_pong -- --nocapture

# Via Makefile
make test-e2e
make test-e2e-verbose
```

## Test Suites

| Suite | Tests | Coverage |
|-------|-------|----------|
| Pong WASM Game (Core) | 6 | WASM loading, rendering, input |
| Pong Demo Features | 22 | Game modes, HUD, AI widgets |
| Release Readiness | 11 | Stress tests, performance, edge cases |

**Total: 39 tests**

## Architecture

### Dual-Runtime Strategy

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│  WasmRuntime (wasmtime)         │     │  BrowserController (Chrome)     │
│  ─────────────────────────      │     │  ─────────────────────────      │
│  Purpose: LOGIC-ONLY testing    │     │  Purpose: GOLDEN MASTER         │
│                                 │     │                                 │
│  ✓ Unit tests                   │     │  ✓ E2E tests                    │
│  ✓ Deterministic replay         │     │  ✓ Visual regression            │
│  ✓ Invariant fuzzing            │     │  ✓ Browser compatibility        │
│  ✓ Performance benchmarks       │     │  ✓ Production parity            │
│                                 │     │                                 │
│  ✗ NOT for rendering            │     │  This is the SOURCE OF TRUTH    │
│  ✗ NOT for browser APIs         │     │  for "does it work?"            │
└─────────────────────────────────┘     └─────────────────────────────────┘
```

## Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| **Poka-Yoke** | Type-safe selectors prevent typos at compile time |
| **Muda** | Zero-copy memory views eliminate serialization |
| **Genchi Genbutsu** | ProbarDriver abstraction for swappable backends |
| **Andon Cord** | Fail-fast mode stops on first critical failure |
| **Jidoka** | Quality built into the type system |

## Next Steps

- [Why Probar?](./why-probar.md) - Detailed comparison with Playwright
- [Quick Start](./quick-start.md) - Get started testing
- [Assertions](./assertions.md) - Available assertion types
- [Coverage Tooling](./coverage-tooling.md) - Advanced coverage analysis
