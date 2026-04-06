# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: ComputeBlock Architecture

**TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

This is not a guideline. It is an architectural constraint enforced at compile-time.

```
┌─────────────────────────────────────────────────────────────────┐
│  YOU CANNOT BUILD PRESENTAR WITHOUT TESTS                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Write test FIRST (references non-existent interface)        │
│  2. Build fails → Add interface (struct fields, methods)        │
│  3. Test fails → Implement logic                                │
│  4. Test passes → Interface is now guaranteed                   │
└─────────────────────────────────────────────────────────────────┘
```

### Enforcement Mechanisms (Active)

| Layer | Mechanism | Location |
|-------|-----------|----------|
| **build.rs** | `panic!` if test files missing | `crates/presentar-terminal/build.rs` |
| **include_str!** | Compile error if test missing | `src/ptop/mod.rs` |
| **Proc macros** | `#[interface_test]`, `#[requires_interface]` | `presentar-test-macros` |
| **Git hooks** | Block commits without tests | `scripts/install-hooks.sh` |
| **CI** | Block PRs without test coverage | `.github/workflows/enforce.yml` |

### To Add Any Feature

```bash
# 1. Write interface test FIRST
echo '#[test] fn test_new_field() { let _ = &snapshot.new_field; }' > tests/new_feature.rs

# 2. Add enforcement line to src/ptop/mod.rs
# pub const _ENFORCE_NEW: &str = include_str!("../../tests/new_feature.rs");

# 3. Now implement (impossible without steps 1-2)
```

See: `docs/specifications/pixel-by-pixel-demo-ptop-ttop.md` Part 0: Epistemological Foundation

---

## CRITICAL: Contract-First Design

**NEVER write code before writing a provable contract.**

All code changes MUST have a corresponding contract (YAML in ../provable-contracts/contracts/<project>/ or .pmat-work/<TICKET>/contract.json) BEFORE implementation. This is enforced by `pmat comply` CB-1400.

- Use `pmat comply check` to verify contract coverage
- Minimum verification level: L1 (recommended L3+)
- See docs/agent-instructions/provable-contract-first-agents.md for the full workflow

## Project Overview

Presentar is a **WASM-first visualization and rapid application framework** built on the Sovereign AI Stack (Trueno, Aprender, Realizar, Pacha). It eliminates Python/CUDA/cloud dependencies for fully self-hosted AI workloads.

**Key characteristics:**
- Pure Rust targeting `wasm32-unknown-unknown`
- 80% Sovereign Stack (trueno-viz GPU primitives), 20% minimal external (winit, fontdue)
- YAML-driven declarative app configuration
- 60fps GPU-accelerated rendering via WebGPU/WGSL shaders
- Unidirectional data flow (Event → State → Widget → Draw)

## Architecture (Layer Hierarchy)

```
Layer 9: App Runtime        - YAML parser, .apr/.ald loaders, Pacha integration
Layer 8: Presentar          - Widget tree, layout engine, event dispatch, state management
Layer 7: Trueno-Viz         - Paths, fills, strokes, text, charts, WGSL shaders
Layer 6: Trueno             - SIMD/GPU tensor ops, backend dispatch, memory management
```

## Build Commands

```bash
# Development
cargo build -p presentar-terminal --features ptop
cargo run -p presentar-terminal --features ptop --bin ptop

# Testing (MANDATORY before any implementation)
cargo test -p presentar-terminal --features ptop
cargo test -p presentar-terminal --features ptop --test cpu_exploded_async  # Interface tests

# Install git hooks (run once after clone)
./scripts/install-hooks.sh

# Quality gates
cargo clippy --all-features -- -D warnings
cargo fmt --check
cargo llvm-cov --features ptop
```

## Core File Types

- `app.yaml` - Presentar application manifest (layout, data sources, interactions)
- `.apr` - Aprender model files
- `.ald` - Alimentar dataset files
- `.presentar-gates.toml` - Quality gate configuration

## Testing Framework (presentar-test)

**Zero external dependencies** - no playwright, selenium, puppeteer, npm, or C bindings. Pure Rust + WASM only.

Key testing patterns:
- `#[presentar_test]` - Standard test attribute
- `#[interface_test(Name)]` - Defines interface (generates proof type)
- `#[requires_interface(Name)]` - Requires interface test to exist
- `Harness::new(include_bytes!("fixtures/app.tar"))` for fixture loading
- `TuiTestBackend` + `expect_frame()` for TUI testing
- Visual regression via `TuiSnapshot::assert_match()`

## Quality Standards

- **Test coverage:** ≥85% (enforced by CI)
- **Interface tests:** MANDATORY for all async data flow
- **Performance:** <1ms full redraw, <0.1ms diff update
- **Visual regression:** Zero tolerance pixel-perfect tests

## Key Dependencies

- `trueno` - SIMD-accelerated tensor operations (always use latest from crates.io)
- `presentar-test` - Testing framework with architectural enforcement
- `sysinfo` - System metrics (for ptop)
- `crossterm` - Terminal backend

## Specification Documents

| Spec | Purpose |
|------|---------|
| `docs/specifications/pixel-by-pixel-demo-ptop-ttop.md` | **UNIFIED SPEC** - TUI architecture, Popperian falsification, pixel-perfect ttop parity |

## Expression Language (in YAML)

```
{{ source | transform | transform }}

Transforms: filter(), select(), sort(), limit(), count(), sum(), mean(), rate(), percentage(), join()
```

All transforms execute client-side in WASM - no server round-trips.


## Code Search Policy

**NEVER use grep/glob for code search. ALWAYS prefer `pmat query`.**

`pmat query` returns quality-annotated, semantically ranked results with TDG grades, complexity, and --faults patterns.

```bash
# Find functions by intent
pmat query "layout engine" --limit 10

# Find with fault patterns
pmat query "unwrap" --faults --exclude-tests

# Include source code
pmat query "render" --include-source

# Regex search
pmat query --regex "fn\s+draw_\w+" --limit 10
```

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
