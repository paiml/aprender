# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aprender is a next-generation machine learning library written in pure Rust. v0.1.0 implements core ML algorithms with 103 unit tests and a pmat TDG score of A (92.6/100).

## Build Commands

```bash
# Standard Cargo commands
cargo build --release        # Optimized build
cargo test                   # Full test suite (103 unit tests + doctests)
cargo test --lib             # Unit tests only
cargo fmt --check            # Check formatting
cargo clippy -- -D warnings  # Strict linting
cargo doc --no-deps --open   # Generate and view docs
cargo bench                  # Run criterion benchmarks

# Makefile shortcuts
make tier1                   # Fast feedback (<1s): fmt, clippy, check
make tier2                   # Pre-commit (<5s): tests + strict clippy
make tier3                   # Pre-push (1-5min): full validation
make tier4                   # CI/CD: includes pmat analysis
```

### Tiered Quality Gates (Certeza Methodology)

```bash
# Tier 1: On-Save (<1s) - Fast feedback
cargo fmt --check && cargo clippy -- -W all && cargo check

# Tier 2: Pre-Commit (<5s)
cargo test --lib && cargo clippy -- -D warnings

# Tier 3: Pre-Push (1-5 min)
cargo test --all
cargo llvm-cov --all-features --workspace    # Coverage (target: 95%)
pmat analyze complexity                       # Max 10 cyclomatic/function
pmat analyze satd                            # Zero TODO/FIXME/HACK

# Tier 4: CI/CD (5-60 min)
cargo mutants --no-times                     # Mutation testing (target: 85%)
pmat tdg . --include-components              # TDG score (target: A+ = 95.0+)
```

## Architecture

### Core Design Patterns

1. **Trait-Based Multiple Dispatch** - Julia-inspired pattern for different data types/backends
2. **Backend Agnostic** - Algorithms transparent to CPU (SIMD), GPU, WASM via Trueno
3. **Three-Tier API**:
   - High: `Estimator` trait (sklearn-like `fit`/`predict`/`score`)
   - Mid: `Optimizer`, `Loss`, `Regularizer` abstractions
   - Low: Direct Trueno primitives

### Dependencies

**Runtime:** Only `trueno = "0.3+"` (provides Vector, Matrix, backend dispatch)

**Dev/Quality Tools:**
- `proptest` - Property-based testing (10K+ cases)
- `criterion` - Benchmarking
- `pmat` - Technical debt grading
- `renacer` - Profiling
- `cargo-mutants` - Mutation testing

### Banned Dependencies
serde, rayon, tokio, thiserror, ndarray, polars, arrow - see spec for rationale

## Testing Strategy

Target distribution: 60% unit, 30% property tests, 10% integration

```bash
cargo test                              # All tests
cargo llvm-cov --html                   # Coverage report
cargo mutants --no-times                # Mutation testing
```

## Phase 1 Scope (v0.1.0)

- **DataFrame:** Minimal column naming (~300 LOC)
- **Linear Regression:** OLS via normal equations
- **K-Means:** Lloyd's + k-means++ initialization
- **Metrics:** R², MSE, MAE, inertia, silhouette

## Key Files

- `src/lib.rs` - Library entry point with module exports
- `src/traits.rs` - Core traits (Estimator, UnsupervisedEstimator, Transformer)
- `src/primitives/` - Vector and Matrix types with Cholesky solver
- `src/data/mod.rs` - DataFrame implementation (~250 LOC)
- `src/linear_model/mod.rs` - Linear regression with OLS
- `src/cluster/mod.rs` - K-Means with k-means++ initialization
- `src/metrics/mod.rs` - R², MSE, MAE, inertia, silhouette
- `docs/specifications/aprender-spec-v1.md` - Full v1.0 specification

## Integration

Ruchy integration for Python-like syntax. Data flow:
```
polars::DataFrame → ruchy bridge → aprender::DataFrame → .to_matrix() → trueno primitives
```
