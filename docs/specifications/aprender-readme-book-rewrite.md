# APR-DOCS: Monorepo README + Book Rewrite

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0
**Date**: 2026-04-07
**Status**: PROPOSAL
**Priority**: P1 — First impression for all visitors
**Contract**: `contracts/apr-docs-v1.yaml`
**Prerequisite**: APR-MONO consolidation complete (70 crates, 20 repos merged)

---

## Problem

The README.md and book were written for aprender-as-library (v0.27, single crate).
After the monorepo consolidation:

1. **README.md** (338 lines) describes aprender as an "ML library" — it's now a
   **full ML framework** with 70 crates, 57 CLI commands, and `cargo install aprender`
2. **book/** (296 chapters) covers the ML library only — missing inference, training,
   serving, orchestration, GPU compute, contracts, profiling
3. **apr-cookbook** (252 chapters) lives in a separate repo — should be merged
4. No mention of `apr` CLI, `apr run`, `apr serve`, etc.
5. No "getting started" path for the most common use case: run a model

## Goal

Rewrite README.md and book/ so that **paiml/aprender is THE single destination**
for the Sovereign AI Stack. A visitor should understand in 30 seconds:

1. What aprender is (ML framework in Rust)
2. How to install it (`cargo install aprender`)
3. How to use it (`apr run model.gguf "What is 2+2?"`)
4. The full scope (70 crates, inference → training → serving → orchestration)

## README.md Rewrite

### Structure (target: ~200 lines, scannable)

```markdown
# aprender

> Next-generation ML framework in pure Rust. One install, one binary, full stack.

## Quick Start (3 lines)

cargo install aprender
apr pull qwen2.5-coder-1.5b
apr run qwen2.5-coder-1.5b "What is 2+2?"

## What is aprender?

- 70 workspace crates (was: 20 separate repos)
- 57 CLI commands via `apr`
- 25,391 tests, 405 provable contracts
- Inference + Training + Serving + Orchestration

## CLI Overview (table: command → description)

## Architecture (Polars-style crate map)

## Install

cargo install aprender       # latest
cargo install aprender@0.29  # specific version

## Library Usage

use aprender::linear_regression::LinearRegression;

## Performance (tok/s table)

## Contributing

## License
```

### Falsification Conditions

| ID | Condition | Check |
|----|-----------|-------|
| FALSIFY-README-001 | Missing `cargo install aprender` | grep |
| FALSIFY-README-002 | Missing `apr run` example | grep |
| FALSIFY-README-003 | References `apr-cli` as install target | grep -v |
| FALSIFY-README-004 | References old repo names as active | grep -v |
| FALSIFY-README-005 | Crate count doesn't match workspace | cargo metadata vs README |
| FALSIFY-README-006 | No performance claims | grep tok/s |

## Book Rewrite

### Strategy

1. **Keep**: `book/` (296 chapters) — ML library reference
2. **Keep separate**: `apr-cookbook` (252 chapters) — recipes have own release cadence
3. **Add new sections to book/**:
   - Getting Started (install, first model, first inference)
   - CLI Reference (auto-generated from `apr <cmd> --help`)
   - Architecture (monorepo crate map)
   - Inference Guide (apr run, apr serve, apr chat)
   - Training Guide (apr finetune, apr train, apr distill)
   - Contracts Reference (405 provable contracts)
   - Migration Guide (trueno → aprender-compute, etc.)

### Target Book Structure

```
book/src/
├── introduction.md
├── getting-started/
│   ├── installation.md          # cargo install aprender
│   ├── first-inference.md       # apr run model "prompt"
│   ├── first-training.md        # apr finetune model
│   └── first-server.md          # apr serve model
├── cli-reference/               # Auto-generated from --help
│   ├── apr-run.md
│   ├── apr-serve.md
│   ├── apr-chat.md
│   └── ... (57 commands)
├── architecture/
│   ├── monorepo-layout.md       # 70 crates, flat layout
│   ├── crate-map.md             # dependency graph
│   └── naming-convention.md     # aprender-* pattern
├── cookbook/                     # Merged from apr-cookbook
│   ├── recipes/
│   ├── advanced/
│   └── concepts/
├── ml-library/                  # Existing book content
│   ├── linear-regression.md
│   ├── decision-trees.md
│   └── ...
├── inference/
│   ├── gguf-loading.md
│   ├── quantization.md
│   ├── kv-cache.md
│   └── gpu-acceleration.md
├── training/
│   ├── lora-qlora.md
│   ├── distillation.md
│   └── declarative-config.md
├── contracts/
│   ├── overview.md
│   ├── writing-contracts.md
│   └── falsification.md
├── migration/
│   ├── from-trueno.md
│   ├── from-entrenar.md
│   └── crate-rename-table.md
└── appendix/
    ├── changelog.md
    └── benchmarks.md
```

## Implementation Plan

### Phase 1: README.md rewrite (now)
- Rewrite from scratch following the structure above
- Verify with FALSIFY-README-* conditions
- Contract: `contracts/apr-docs-v1.yaml`

### Phase 2: Merge apr-cookbook
- `git subtree add --prefix=book/src/cookbook ../apr-cookbook main`
- Update book/src/SUMMARY.md with cookbook chapters

### Phase 3: Add new book sections
- Getting Started, CLI Reference, Architecture, Migration
- Auto-generate CLI reference from `apr <cmd> --help`

### Phase 4: Validate
- `mdbook build` passes
- All internal links resolve
- `pv lint` on contracts referenced in book
- provable-contracts doc_integrity validators pass
