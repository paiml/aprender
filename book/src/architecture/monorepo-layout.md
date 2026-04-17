<!-- PCU: architecture-monorepo-layout | contract: contracts/apr-page-architecture-monorepo-layout-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# Monorepo Layout

Aprender is a monorepo with 70 workspace crates in flat `crates/aprender-*` layout,
following the same pattern as [Polars](https://github.com/pola-rs/polars) (28 crates),
[Burn](https://github.com/tracel-ai/burn) (33 crates), and
[Nushell](https://github.com/nushell/nushell) (40+ crates).

## Directory Structure

```
paiml/aprender/
├── Cargo.toml                    # Workspace root + cargo install aprender
├── src/bin/apr.rs                # Binary entry point
├── crates/
│   ├── aprender-core/            # ML library (use aprender::*)
│   ├── apr-cli/                  # CLI logic (57 commands)
│   ├── aprender-compute/         # SIMD/GPU compute
│   ├── aprender-gpu/             # CUDA PTX kernels
│   ├── aprender-serve/           # Inference server
│   ├── aprender-train/           # Training loops
│   ├── aprender-orchestrate/     # Agents, RAG
│   ├── aprender-contracts/       # Provable contracts
│   ├── aprender-profile/         # Profiling
│   └── ... (70 crates total)
├── contracts/                    # 405 provable YAML contracts
├── book/                         # This book
└── docs/specifications/          # Design specs
```

## Consolidated from 20 Repos

| Old Repo | New Crate | Role |
|----------|-----------|------|
| paiml/trueno | `aprender-compute` | SIMD/GPU compute |
| paiml/trueno-gpu | `aprender-gpu` | CUDA PTX kernels |
| paiml/realizar | `aprender-serve` | Inference server |
| paiml/entrenar | `aprender-train` | Training loops |
| paiml/batuta | `aprender-orchestrate` | Agents, RAG |
| paiml/provable-contracts | `aprender-contracts` | Contract enforcement |
| paiml/renacer | `aprender-profile` | Profiling |
| paiml/presentar | `aprender-present-*` | TUI framework |
| paiml/trueno-db | `aprender-db` | Embedded analytics |
| paiml/trueno-graph | `aprender-graph` | Graph database |
| paiml/trueno-rag | `aprender-rag` | RAG pipeline |
| + 9 more | `aprender-*` | Various |

All old repos are archived (read-only). Development happens here.

## Why Monorepo?

- **Zero version skew**: all 70 crates share one version (0.29.0)
- **Atomic changes**: cross-crate refactoring in one PR
- **One CI pipeline**: `cargo test --workspace` catches everything
- **One install**: `cargo install aprender` gets the whole stack

See: [APR-MONO spec](../../docs/specifications/aprender-monorepo-consolidation.md)
