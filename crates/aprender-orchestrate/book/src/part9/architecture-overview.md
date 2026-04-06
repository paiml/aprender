# Architecture Overview

Batuta is structured as a modular Rust binary with clearly separated concerns. Each module handles one aspect of the orchestration pipeline, and feature flags control which capabilities are compiled into the binary.

## Module Structure

```
src/
├── main.rs                 # CLI entry point (native feature)
├── lib.rs                  # Library root, feature-gated exports
├── pipeline.rs             # 5-phase transpilation pipeline
├── backend.rs              # Cost-based GPU/SIMD/Scalar selection
├── oracle/                 # Knowledge graph and query engine
│   ├── mod.rs              # Oracle entry point
│   ├── recipes.rs          # 34 cookbook recipes + test companions
│   └── recommender.rs      # Component recommendation engine
├── serve/                  # Model serving infrastructure
│   ├── mod.rs              # Serve entry point
│   ├── failover.rs         # Circuit breakers, retry logic
│   └── privacy.rs          # Sovereign/Private/Standard tiers
├── stack/                  # Stack coordination
│   ├── mod.rs              # Stack entry point
│   ├── dependencies.rs     # Dependency graph management
│   ├── quality.rs          # Quality gates across components
│   └── release.rs          # Release orchestration
├── cli/                    # Command-line interface
│   ├── mod.rs              # Clap argument parsing
│   ├── oracle.rs           # Oracle subcommand
│   └── stack.rs            # Stack subcommand
├── numpy_converter.rs      # NumPy -> Trueno mapping
├── sklearn_converter.rs    # scikit-learn -> Aprender mapping
└── pytorch_converter.rs    # PyTorch -> Realizar mapping
```

## Feature Flags

| Feature | Purpose | Default | Key Dependencies |
|---------|---------|---------|-----------------|
| `native` | Full CLI, filesystem, tracing, TUI | Yes | clap, tracing, ratatui |
| `wasm` | Browser-compatible build | No | None (removes filesystem) |
| `trueno-integration` | SIMD/GPU tensor operations | No | trueno |
| `oracle-mode` | Knowledge graph queries | No | trueno-graph, trueno-db |

Build variants:

```bash
# Standard CLI build
cargo build --release

# WASM build (browser)
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm

# Full-featured build
cargo build --release --features trueno-integration,oracle-mode
```

## Dependency Graph

```
batuta
├── pipeline.rs ──────> depyler, decy, bashrs (external, via PATH)
├── backend.rs ───────> trueno (SIMD), repartir (distributed)
├── oracle/ ──────────> trueno-graph, trueno-db, trueno-rag
├── serve/ ───────────> realizar (inference), pacha (registry)
├── stack/ ───────────> All stack crates (version checking)
├── numpy_converter ──> trueno (operation mapping)
├── sklearn_converter > aprender (algorithm mapping)
└── pytorch_converter > realizar (inference mapping)
```

## Data Flow

A typical transpilation run flows through the modules in order:

```
User Input ─> CLI (parse args)
           ─> Pipeline Phase 1: Analysis (language detection, TDG)
           ─> Pipeline Phase 2: Transpilation (tool dispatch)
           ─> Pipeline Phase 3: Optimization (backend selection)
           ─> Pipeline Phase 4: Validation (renacer trace, tests)
           ─> Pipeline Phase 5: Build (cargo build --release)
           ─> Output
```

Each phase reads from and writes to the `.batuta/` state directory, enabling resumption after failures and inspection of intermediate results.

## Design Principles

- **Jidoka**: Pipeline halts at the first failure in any phase
- **Poka-Yoke**: Privacy tiers in `serve/` prevent accidental data exposure
- **Heijunka**: Backend selector balances load across CPU/GPU/distributed
- **Kaizen**: Quality gates in `stack/` enforce improvement over time

---

**Navigate:** [Table of Contents](../SUMMARY.md)
