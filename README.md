<p align="center">
  <img src="https://raw.githubusercontent.com/paiml/probar/main/docs/assets/probar-hero.svg?v=2" alt="Probar - Playwright-Compatible Testing for WASM + TUI" width="800"/>
</p>

<h1 align="center">Probar</h1>

<p align="center">
  <strong>Playwright-Compatible Testing for WASM + TUI Applications</strong><br>
  <em>Pure Rust &bull; Zero JavaScript &bull; Games &bull; Simulations &bull; Web Apps</em>
</p>

<p align="center">
  <a href="https://crates.io/crates/jugar-probar">
    <img src="https://img.shields.io/crates/v/jugar-probar.svg" alt="Crates.io">
  </a>
</p>

---

## Installation

**Probar** (Spanish: "to test/prove") is distributed as two crates:

| Crate | Purpose | Install |
|-------|---------|---------|
| **[jugar-probar](https://crates.io/crates/jugar-probar)** | Library for writing tests | `cargo add jugar-probar --dev` |
| **[probador](https://crates.io/crates/probador)** | CLI tool for running tests | `cargo install probador` |

### Library (jugar-probar)

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
jugar-probar = "1.0"
```

```rust
use jugar_probar::prelude::*;
```

### CLI (probador)

```bash
cargo install probador
```

```bash
# Validate a playbook state machine
probador playbook login.yaml --validate

# Run with mutation testing (M1-M5 falsification)
probador playbook login.yaml --mutate

# Export state diagram
probador playbook login.yaml --export svg -o diagram.svg

# Start dev server for WASM
probador serve --port 8080
```

---

## Overview

Probar is a **Playwright-compatible** testing framework written in **pure Rust**. It provides comprehensive testing for:

- **WASM Applications** - Games, simulations, web apps running in browsers
- **TUI Applications** - Terminal interfaces built with ratatui/crossterm
- **Headless Testing** - Fast CI/CD without browser overhead

## Key Features

### GUI Coverage Tracking

Probar introduces **GUI Coverage** - a new paradigm for measuring UI test completeness:

```rust
use jugar_probar::gui_coverage;

// Define what needs testing (one line!)
let mut gui = gui_coverage! {
    buttons: ["start", "pause", "quit"],
    screens: ["title", "playing", "game_over"]
};

// Record interactions during tests
gui.click("start");
gui.visit("title");

// Get coverage - one line!
println!("{}", gui.summary());  // "GUI: 33% (1/3 elements, 1/3 screens)"
assert!(gui.meets(80.0));       // Fail if below 80%
```

### Playwright-Compatible API

```rust
// Familiar Playwright-style locators and assertions
let button = page.locator("button").with_text("Start Game");
button.click().await?;

expect(&score).to_have_text("100").await?;
```

### Test Targets

| Target | Description | Use Case |
|--------|-------------|----------|
| **WASM Browser** | Chrome DevTools Protocol (CDP) | Games, web apps, simulations |
| **WASM Headless** | wasmtime runtime | Fast CI, logic testing |
| **TUI** | ratatui/crossterm backends | Terminal applications |

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Browser Automation** | Chrome DevTools Protocol (CDP) via chromiumoxide |
| **WASM Runtime Testing** | Logic-only testing via wasmtime (no browser overhead) |
| **TUI Testing** | Frame capture and assertion for terminal UIs |
| **GUI Coverage** | Provable UI element and interaction coverage |
| **Visual Regression** | Image comparison for UI stability |
| **Accessibility Auditing** | WCAG compliance checking |
| **Deterministic Replay** | Record and replay sessions with seed control |
| **Monte Carlo Fuzzing** | Random input generation with invariant checking |
| **Zero-JS Validation** | Enforce WASM-first architecture (no user JS) |
| **Worker Harness** | Web Worker lifecycle, ring buffers, shared memory |
| **Docker Testing** | Cross-browser testing via Docker containers |
| **Brick Architecture** | Widget-level testing with assertions, budgets, and verification |

## Quick Start

```rust
use jugar_probar::prelude::*;

#[test]
fn test_calculator_gui() {
    // Create driver (works for TUI or WASM)
    let mut driver = WasmDriver::new();

    // Track GUI coverage
    let mut gui = gui_coverage! {
        buttons: ["btn-7", "btn-times", "btn-6", "btn-equals"],
        screens: ["calculator"]
    };

    // Test: 7 x 6 = 42
    driver.type_input("7 * 6");
    gui.click("btn-7");
    gui.click("btn-times");
    gui.click("btn-6");

    driver.click_equals();
    gui.click("btn-equals");
    gui.visit("calculator");

    assert_eq!(driver.get_result(), "42");
    assert!(gui.is_complete());  // 100% GUI coverage!
}
```

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `tui` | TUI testing support (default) | ratatui, crossterm |
| `browser` | CDP browser automation | chromiumoxide, tokio |
| `runtime` | WASM runtime testing | wasmtime |
| `derive` | Type-safe derive macros | probar-derive |
| `docker` | Docker cross-browser testing | bollard, tokio |
| `gpu` | GPU compute support | trueno |
| `brick` | Brick Architecture for widget testing | — |

## Brick Architecture

Brick Architecture is a revolutionary approach where **tests ARE the interface**. Instead of writing widgets first and then tests, you define widgets by their test assertions.

### Core Concepts

```rust
use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::time::Duration;

struct StatusButton {
    label: String,
    is_enabled: bool,
}

impl Brick for StatusButton {
    fn brick_name(&self) -> &'static str { "StatusButton" }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5),  // WCAG 2.1 AA
            BrickAssertion::MaxLatencyMs(16),    // 60fps budget
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)  // 16ms total for 60fps
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        if self.is_enabled && !self.label.is_empty() {
            passed.push(BrickAssertion::TextVisible);
        } else {
            failed.push((BrickAssertion::TextVisible, "Not visible".into()));
        }

        BrickVerification { passed, failed, verification_time: Duration::from_micros(50) }
    }

    fn to_html(&self) -> String {
        format!(r#"<button class="status-btn">{}</button>"#, self.label)
    }

    fn to_css(&self) -> String {
        ".status-btn { background: #000; color: #fff; }".into()
    }
}
```

### BrickHouse: Budgeted Composition

Compose multiple bricks with a total performance budget using Jidoka (stop-the-line) principles:

```rust
use jugar_probar::brick_house::{BrickHouse, BrickHouseBuilder};

let house = BrickHouseBuilder::new("whisper-app")
    .budget_ms(1000)  // 1 second total budget
    .brick(status_brick, 50)       // 50ms for status
    .brick(waveform_brick, 100)    // 100ms for waveform
    .brick(transcription_brick, 600)  // 600ms for transcription
    .build()?;

// Render with budget enforcement
let html = house.render()?;
let report = house.last_report().unwrap();
println!("Budget utilization: {}%", report.utilization());
```

### web_sys_gen: Zero Hand-Written web_sys

The `web_sys_gen` module provides generated abstractions that replace hand-written `web_sys` calls:

```rust
use jugar_probar::brick::web_sys_gen::{PerformanceTiming, CustomEventDispatcher, EventDetail};

// Instead of: web_sys::window().unwrap().performance().unwrap().now()
let start = PerformanceTiming::now();

// Measure operations with automatic timing
let (result, duration_ms) = PerformanceTiming::measure(|| {
    expensive_computation()
});

// Dispatch custom events without boilerplate
let dispatcher = CustomEventDispatcher::new("transcription-complete");
dispatcher.dispatch_with_detail(EventDetail::json(&transcript_data))?;
```

### Run the Brick Example

```bash
cargo run --example brick_demo -p jugar-probar
```

## Visual Regression Testing

Probar provides pure Rust visual regression testing with perceptual diffing for catching unintended UI changes.

### Basic Comparison

```rust
use jugar_probar::{VisualRegressionTester, VisualRegressionConfig};

let tester = VisualRegressionTester::new(
    VisualRegressionConfig::default()
        .with_threshold(0.01)       // 1% of pixels can differ
        .with_color_threshold(10)   // Allow minor color variations
);

// Compare screenshot against baseline
let result = tester.compare_against_baseline("login-page", &screenshot)?;

if !result.matches {
    println!("Visual regression detected!");
    println!("  Diff pixels: {}", result.diff_pixel_count);
    println!("  Diff percentage: {:.2}%", result.diff_percentage);
}
```

### Masking Dynamic Areas

```rust
use jugar_probar::{ScreenshotComparison, MaskRegion};

let comparison = ScreenshotComparison::new()
    .with_threshold(0.01)
    .with_mask(MaskRegion::new(10, 10, 200, 50))  // Exclude header
    .with_mask(MaskRegion::new(0, 500, 300, 100)); // Exclude footer
```

### Perceptual Diff

Human-vision-weighted comparison for more accurate results:

```rust
use jugar_probar::perceptual_diff;
use image::Rgba;

let pixel_a = Rgba([255, 0, 0, 255]);
let pixel_b = Rgba([200, 50, 50, 255]);

// Uses weighted RGB (Green most sensitive to human eye)
let diff = perceptual_diff(pixel_a, pixel_b);
```

### Run the Visual Regression Example

```bash
cargo run --example visual_regression_demo -p jugar-probar
```

## Usage

### Running Tests

```bash
# Run all tests
cargo test

# Run with coverage
cargo llvm-cov --html

# Watch mode
cargo watch -x test
```

### Using probador CLI

```bash
# Run tests with probador
probador test

# Generate coverage reports
probador coverage --html

# Watch mode with hot reload
probador watch tests/

# Serve WASM application
probador serve --port 8080 --cors
```

### GUI Coverage Example

```bash
# Run the GUI coverage example
cargo run --example gui_coverage -p jugar-probar
```

Output:
```
=== GUI Coverage Example ===

1. Using gui_coverage! macro (simplest)...
   GUI: 50% (1/3 elements, 2/3 screens)

2. Calculator preset (20 buttons + 2 screens)...
   GUI: 60% (14/20 elements, 1/2 screens)

3. Achieving 100% coverage...
   GUI: 100% (3/3 elements, 1/1 screens)
   Complete? true
```

### Showcase Calculator

The repository includes a full showcase calculator demonstrating 100% test coverage:

```bash
# Run GUI coverage report
cargo run -p showcase-calculator --example gui_coverage_report

# Run TUI version
cargo run -p showcase-calculator --example calculator_tui

# View WASM version
cd crates/showcase-calculator/www && python3 -m http.server 8080
# Open http://localhost:8080
```

## Probar Principles

Probar is built on pragmatic testing principles:

| Principle | Description |
|-----------|-------------|
| **Error Prevention** | Type-safe selectors prevent runtime errors |
| **Efficiency** | Zero-copy memory views minimize overhead |
| **Fail-Fast** | Immediate feedback on test failures |
| **Balanced Testing** | Even coverage across all UI elements |
| **Continuous Improvement** | Mutation testing for test quality |

## Documentation

- **[Book](https://paiml.github.io/probar/)** - Comprehensive guide
- **[API Docs](https://docs.rs/jugar-probar)** - Rust documentation
- **[GUI Coverage Guide](book/src/probar/ux-coverage.md)** - GUI coverage tutorial
- **[Examples](crates/probar/examples/)** - 20+ runnable examples
- 🤖 [Coursera Hugging Face AI Development Specialization](https://www.coursera.org/specializations/hugging-face-ai-development) - Build Production AI systems with Hugging Face in Pure Rust

## Examples

| Example | Description |
|---------|-------------|
| `gui_coverage` | GUI coverage tracking |
| `soft_assertions` | Collect multiple failures |
| `retry_assertions` | Retry with backoff |
| `pong_simulation` | Game simulation testing |
| `accessibility_demo` | WCAG compliance |
| `watch_mode` | Hot-reload testing |
| `zero_js_demo` | WASM-first validation |
| `worker_harness_demo` | Web Worker testing |
| `docker_demo` | Docker cross-browser (requires `docker` feature) |
| `streaming_ux_demo` | Real-time streaming validation |
| `brick_demo` | Brick Architecture with BrickHouse |
| `web_sys_gen_demo` | web_sys_gen timing and events |
| `visual_regression_demo` | Visual regression testing |

Run any example:
```bash
cargo run --example <name> -p jugar-probar
```

## Project Structure

```
probar/
├── crates/
│   ├── probar/              # jugar-probar library
│   ├── probar-cli/          # probador CLI
│   ├── probar-derive/       # Derive macros
│   └── showcase-calculator/ # 100% coverage demo
├── book/                    # mdBook documentation
└── docs/                    # Specifications
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create your branch from `main`
2. **Run tests** before submitting: `cargo test`
3. **Ensure formatting**: `cargo fmt`
4. **Check lints**: `cargo clippy --all-targets --all-features`
5. **Update documentation** if you change public APIs
6. **Add tests** for new functionality

### Development Setup

```bash
git clone https://github.com/paiml/probar.git
cd probar
cargo build
cargo test
```

### Quality Gates

```bash
make lint      # Clippy checks
make test      # All tests
make coverage  # Coverage report
```

## License

MIT OR Apache-2.0

---

<p align="center">
  <strong>Probar</strong> - by <a href="https://paiml.com">Pragmatic AI Labs</a>
</p>
