# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL CONSTRAINTS

### ABSOLUTE ZERO JAVASCRIPT

```
❌ FORBIDDEN                         ✅ REQUIRED
─────────────────────────────────────────────────────
• JavaScript files (.js/.ts)         • Pure Rust only
• npm/node_modules/package.json      • wasm32-unknown-unknown
• Any JS bundler                     • web-sys (Rust bindings)
• JS interop beyond web-sys          • Single .wasm binary output
```

**Rationale**: JavaScript introduces non-determinism and GC pauses. Probar compiles to a single `.wasm` file with ZERO JS for testing WASM games.

### Crates.io Name

Published as `jugar-probar` on crates.io (the name `probar` was taken).

```toml
[dev-dependencies]
jugar-probar = "0.2"
```

```rust
use jugar_probar::prelude::*;
```

## Project Overview

Probar (Spanish: "to test/prove") is a Rust-native testing framework for WASM games. It provides:
- Browser automation via Chrome DevTools Protocol (CDP)
- WASM runtime testing via wasmtime
- Visual regression testing
- Accessibility auditing (WCAG)
- Deterministic simulation and replay
- Monte Carlo fuzzing

## Build Commands

```bash
# Build
cargo build                    # Host target (dev)
cargo build --all-features     # All features enabled

# Test
cargo test                     # Run all tests
cargo test --all-features      # Run all tests with all features

# Quality
cargo clippy -- -D warnings    # Lint with strict warnings
cargo fmt --check              # Check formatting
```

## Architecture

### Crate Structure

```
crates/
├── probar/           # Main testing framework
└── probar-derive/    # Proc-macro crate for type-safe selectors
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `browser` | Enable CDP browser control (chromiumoxide, tokio) |
| `runtime` | Enable WASM runtime (wasmtime) for logic testing |
| `derive` | Enable derive macros for type-safe ECS selectors |

### Key Components

- **Locator API**: Playwright-style element selection with auto-waiting
- **WasmRuntime**: Logic-only testing without browser overhead
- **StateBridge**: Game state snapshot and diffing
- **InputFuzzer**: Monte Carlo input generation
- **VisualRegressionTester**: Image comparison for UI testing

## Toyota Way Principles

- **Poka-Yoke**: Type-safe selectors prevent stringly-typed errors at compile time
- **Muda**: Zero-copy memory views eliminate serialization waste
- **Jidoka**: Fail-fast with soft Jidoka (LogAndContinue vs Stop)
- **Genchi Genbutsu**: Abstract ProbarDriver allows swapping browser implementations
- **Heijunka**: Superblock tiling for amortized scheduling

## Quality Standards

- **95% minimum test coverage**
- **Zero tolerance for panic paths** (clippy: deny unwrap_used, expect_used)
- **Comprehensive inline documentation**

### FORBIDDEN: cargo-tarpaulin

```
❌ FORBIDDEN                         ✅ REQUIRED
─────────────────────────────────────────────────────
• cargo-tarpaulin                    • cargo-llvm-cov
• tarpaulin in CI                    • make coverage (llvm-cov)
• Any tarpaulin config               • cargo-nextest for speed
```

**Rationale**: tarpaulin is slow, unreliable, and produces inconsistent results. Use `cargo-llvm-cov` + `cargo-nextest` exclusively (bashrs pattern).

**Coverage Commands**:
```bash
make coverage          # Generate HTML report (<5 min target)
make coverage-summary  # Show coverage percentage
make coverage-ci       # Generate LCOV for CI
```

## Dogfooding: Book Screenshots

Probar generates its own documentation screenshots using its pixel coverage heatmap feature. This validates our PNG export functionality.

### Generating Screenshots for Documentation

```bash
# Generate heatmap PNGs to /tmp/
cargo run --example pixel_coverage_heatmap -p jugar-probar

# Copy to book assets
cp /tmp/coverage_viridis.png /tmp/coverage_magma.png \
   /tmp/coverage_pattern.png /tmp/coverage_combined.png \
   book/src/assets/
```

### Screenshot Files

| File | Description |
|------|-------------|
| `coverage_viridis.png` | Viridis palette (colorblind-safe) |
| `coverage_magma.png` | Magma palette (dark to bright) |
| `coverage_pattern.png` | Gradient pattern with gap highlighting |
| `coverage_combined.png` | Combined line + pixel coverage report |

### Using in Markdown

```markdown
![Viridis Heatmap](../assets/coverage_viridis.png)
```

### Regenerating After Changes

When modifying `pixel_coverage` or `PngHeatmap`:

1. Run the example to generate new PNGs
2. Copy to `book/src/assets/`
3. Build the book: `mdbook build book`
4. Verify images render correctly

This workflow validates our PNG export code every time documentation is updated.


## Contract-First Development

All new features and bug fixes must follow provable-contract-first methodology:
1. Write or update the contract YAML in `../provable-contracts/contracts/<crate>/`
2. Run `pmat comply check` to validate compliance
3. Implement the code to satisfy the contract
4. Run `pmat comply check` again to confirm

## Stack Documentation Search (RAG Oracle)

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up patterns from other stack components
- Finding cross-language equivalents (Python HuggingFace → Rust)
- Understanding how similar problems are solved elsewhere in the stack

```bash
# Search across the entire Sovereign AI Stack
batuta oracle --rag "your question here"

# Reindex after changes (auto-runs via post-commit hook + ora-fresh)
batuta oracle --rag-index

# Check index freshness (runs automatically on shell login)
ora-fresh
```

The RAG index covers 5000+ documents across the Sovereign AI Stack.
Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
