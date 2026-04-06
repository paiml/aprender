# Probar

**Probar** (Spanish: "to test/prove") is a Rust-native testing framework for WASM games, providing a pure Rust alternative to Playwright/Puppeteer.

![Probar Coverage Visualization](probar/assets/coverage_viridis.png)

## Installation

Probar is distributed as two crates:

| Crate | Purpose | Install |
|-------|---------|---------|
| **[jugar-probar](https://crates.io/crates/jugar-probar)** | Library for writing tests | `cargo add jugar-probar --dev` |
| **[probador](https://crates.io/crates/probador)** | CLI tool for running tests | `cargo install probador` |

### Library (jugar-probar)

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
jugar-probar = "0.3"
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

# Run with mutation testing
probador playbook login.yaml --mutate

# Export state diagram
probador playbook login.yaml --export svg -o diagram.svg

# Start dev server for WASM
probador serve --port 8080
```

## Features

- **Browser Automation**: Chrome DevTools Protocol (CDP) via chromiumoxide
- **WASM Runtime Testing**: Logic-only testing via wasmtime (no browser overhead)
- **Visual Regression**: Image comparison for UI stability
- **Accessibility Auditing**: WCAG compliance checking
- **Deterministic Replay**: Record and replay game sessions
- **Monte Carlo Fuzzing**: Random input generation with invariant checking
- **Type-Safe Selectors**: Compile-time checked entity/component queries
- **GUI Coverage**: Provable UI element and interaction coverage

## Feature Flags

| Feature | Description |
|---------|-------------|
| `browser` | CDP browser automation (chromiumoxide, tokio) |
| `runtime` | WASM runtime testing (wasmtime) |
| `derive` | Type-safe derive macros (probar-derive) |

## Why Probar?

| Aspect | Playwright | Probar |
|--------|-----------|--------|
| **Language** | TypeScript | Pure Rust |
| **Browser** | Required (Chromium) | Optional |
| **Game State** | Black box (DOM only) | Direct API access |
| **CI Setup** | Node.js + browser | Just `cargo test` |
| **Zero JS** | Violates constraint | Pure Rust |

## Design Principles

Probar is built on pragmatic testing principles:

- **Poka-Yoke** (Mistake-Proofing): Type-safe selectors prevent runtime errors
- **Muda** (Waste Elimination): Zero-copy memory views for efficiency
- **Jidoka** (Autonomation): Fail-fast with configurable error handling
- **Genchi Genbutsu** (Go and See): Abstract drivers allow swapping implementations
- **Heijunka** (Level Loading): Superblock scheduling for consistent performance
