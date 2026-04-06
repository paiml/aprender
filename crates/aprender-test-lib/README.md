# jugar-probar

[![Crates.io](https://img.shields.io/crates/v/jugar-probar.svg)](https://crates.io/crates/jugar-probar)
[![Documentation](https://docs.rs/jugar-probar/badge.svg)](https://docs.rs/jugar-probar)
[![CI](https://github.com/paiml/probar/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/probar/actions/workflows/ci.yml)

**jugar-probar** is the Rust library for [Probar](https://github.com/paiml/probar) - a Playwright-compatible testing framework for WASM games and applications.

> **Note:** The CLI tool is published separately as [probador](https://crates.io/crates/probador).

## Installation

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
jugar-probar = "0.3"
```

With specific features:

```toml
[dev-dependencies]
jugar-probar = { version = "0.3", features = ["browser", "runtime", "derive"] }
```

## Quick Start

```rust
use jugar_probar::prelude::*;

#[test]
fn test_game_starts() {
    // Create test platform
    let config = WebConfig::new(800, 600);
    let mut platform = WebPlatform::new_for_test(config);

    // Run initial frame
    let output = platform.frame(0.0, "[]");

    // Verify game started
    assert!(output.contains("commands"));
}
```

## Features

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

## Feature Flags

| Feature | Description |
|---------|-------------|
| `browser` | CDP browser automation (chromiumoxide, tokio) |
| `runtime` | WASM runtime testing (wasmtime) |
| `derive` | Type-safe derive macros (probar-derive) |

## Examples

```bash
# Deterministic simulation with replay verification
cargo run --example pong_simulation -p jugar-probar

# Playwright-style locator API demo
cargo run --example locator_demo -p jugar-probar

# WCAG accessibility checking
cargo run --example accessibility_demo -p jugar-probar

# GUI coverage tracking
cargo run --example gui_coverage -p jugar-probar
```

## API Overview

### Assertions

```rust
use jugar_probar::Assertion;

// Value equality
let eq = Assertion::equals(&actual, &expected);

// Numeric range
let range = Assertion::in_range(value, 0.0, 100.0);

// Boolean checks
let truthy = Assertion::is_true(condition);

// Approximate equality (floats)
let approx = Assertion::approx_eq(3.14159, std::f64::consts::PI, 0.001);
```

### GUI Coverage

```rust
use jugar_probar::gui_coverage;

let mut gui = gui_coverage! {
    buttons: ["start", "pause", "quit"],
    screens: ["title", "playing", "game_over"]
};

gui.click("start");
gui.visit("title");

println!("{}", gui.summary());  // "GUI: 33% (1/3 elements, 1/3 screens)"
assert!(gui.meets(80.0));       // Fail if below 80%
```

### Simulation

```rust
use jugar_probar::{run_simulation, run_replay, SimulationConfig, InputEvent};

// Record a simulation
let config = SimulationConfig::new(seed, frames);
let recording = run_simulation(config, |frame| {
    vec![InputEvent::key_press("ArrowUp")]
});

// Replay and verify determinism
let replay = run_replay(&recording);
assert!(replay.determinism_verified);
```

## CLI Tool

For command-line usage, install the CLI separately:

```bash
cargo install probador
```

```bash
# Validate playbook state machines
probador playbook login.yaml --validate

# Run mutation testing
probador playbook login.yaml --mutate

# Export state diagrams
probador playbook login.yaml --export svg -o diagram.svg
```

## Documentation

- [Book](https://paiml.github.io/probar/) - Comprehensive guide
- [API Docs](https://docs.rs/jugar-probar) - Rust documentation
- [Examples](https://github.com/paiml/probar/tree/main/crates/probar/examples) - 20+ runnable examples

## License

MIT OR Apache-2.0
