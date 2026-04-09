# Testing with Probar

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §13
**Crate**: `jugar-probar` (lib), `probador` (CLI)

---

## 1. Overview

Probar ("to test/prove" in Spanish) provides three testing capabilities
used throughout the Aprender stack:

1. **Visual regression** — golden activation snapshots for model operations
2. **GUI coverage** — widget and screen coverage for TUI/WASM interfaces
3. **E2E validation** — Playwright-compatible testing for WASM deployments

Published as `jugar-probar` on crates.io (the name `probar` was taken).

---

## 2. Visual Regression Testing

### 2.1 Purpose

Every model operation that modifies weights (merge, finetune, prune,
quantize, distill) must prove that the resulting activations are within
tolerance of a golden reference. This is `apr probar`.

### 2.2 Workflow

```bash
# Step 1: Capture golden reference
apr probar model.apr --golden golden/ --format png

# Step 2: Run model operation
apr quantize model.apr --scheme q4k -o model-q4k.apr

# Step 3: Validate against golden
apr probar model-q4k.apr --golden golden/ --assert
#   → PASS: all layers within tolerance
#   → FAIL: layer 12 attention diverges (cosine 0.94 < 0.98)
```

### 2.3 CLI Flags

| Flag | Effect |
|------|--------|
| `--golden DIR` | Path to golden reference snapshots |
| `--assert` | Exit non-zero on divergence (CI mode) |
| `--format json\|png\|both` | Output format |
| `--layer PATTERN` | Filter layers by name pattern |
| `--tolerance FLOAT` | Cosine similarity threshold (default 0.98) |
| `--profile` | Combine with BrickProfiler timing |
| `--output DIR` | Write activation snapshots to directory |

### 2.4 What Gets Compared

For each layer in the model:

| Metric | Golden | Candidate | Pass If |
|--------|--------|-----------|---------|
| Activation mean | µ_gold | µ_cand | \|µ_gold - µ_cand\| < ε |
| Activation std | σ_gold | σ_cand | \|σ_gold - σ_cand\| < ε |
| Cosine similarity | — | — | cosine(gold, cand) ≥ tolerance |
| NaN/Inf count | 0 | n | n == 0 |
| Shape | [d1, d2] | [d1', d2'] | exact match |

### 2.5 Golden Snapshots in CI

Golden snapshots are committed to the repository under `tests/golden/`.
CI runs `apr probar --golden tests/golden/ --assert` in tier2 gates.

Any weight-modifying PR must update goldens:
```bash
# Regenerate after intentional model changes
apr probar model.apr --golden tests/golden/ --format png
git add tests/golden/
```

---

## 3. GPU Kernel Testing

### 3.1 PTX Static Analysis

Probar includes a `gpu_pixels` module for CUDA kernel correctness:

- Shared memory bounds verification
- Loop bounds analysis
- Thread addressing correctness
- Atomic operation validation

```rust
use jugar_probar::gpu_pixels::PtxAnalyzer;

let analysis = PtxAnalyzer::new(ptx_source)
    .check_shared_memory()
    .check_loop_bounds()
    .check_thread_addressing();
assert!(analysis.is_safe());
```

### 3.2 Cross-Backend Validation

Probar golden tests validate equivalence across compute backends:

```bash
# Capture CPU reference
TRUENO_BACKEND=cpu apr probar model.apr --golden golden-cpu/

# Validate GPU against CPU golden
TRUENO_BACKEND=cuda apr probar model.apr --golden golden-cpu/ --assert
TRUENO_BACKEND=wgpu apr probar model.apr --golden golden-cpu/ --assert
```

This complements the contract-based parity gate with empirical validation.

---

## 4. GUI Coverage (TUI/WASM)

### 4.1 Coverage Tracking

```rust
use jugar_probar::prelude::*;

let mut gui = gui_coverage! {
    buttons: ["run", "stop", "trace", "profile"],
    screens: ["model_select", "inference", "profile_view", "settings"]
};

// Exercise the TUI
simulate_click("run");
gui.record_button("run");
gui.record_screen("inference");

// Assert coverage threshold
assert!(gui.meets(80.0)); // 80% of buttons and screens exercised
```

### 4.2 Pixel-Level Heatmaps

For WASM deployments, probar generates pixel-level coverage heatmaps:

```bash
probador coverage --html --palette viridis
```

Shows which UI regions were exercised during testing. Cold spots indicate
untested interaction paths.

### 4.3 TUI Snapshot Testing

```rust
use jugar_probar::tui::*;

let snapshot = capture_tui_frame(&mut terminal);
assert_snapshot!(snapshot, "expected_frame.txt");
```

Tests `apr tui` and other terminal UIs via exact character-grid comparison
with semantic diff on divergence.

---

## 5. Playbook-Driven Testing

### 5.1 YAML Playbooks

State machine tests defined as YAML:

```yaml
name: inference_flow
states: [idle, loading, running, complete, error]
transitions:
  - from: idle
    to: loading
    trigger: "apr run model.apr"
  - from: loading
    to: running
    trigger: model_loaded
  - from: running
    to: complete
    trigger: generation_done
  - from: running
    to: error
    trigger: nan_detected
```

### 5.2 Mutation Testing (M1–M5)

Probar mutates playbooks to test robustness:

| Class | Mutation | Purpose |
|-------|----------|---------|
| M1 | Remove transition | Dead state detection |
| M2 | Swap trigger | Wrong ordering detection |
| M3 | Add invalid state | State explosion detection |
| M4 | Remove guard | Invariant violation detection |
| M5 | Duplicate transition | Non-determinism detection |

```bash
probador playbook inference.yaml --mutate --validate
```

---

## 6. Integration with apr-cli

### 6.1 The `apr probar` Command

`apr probar` is a first-class CLI command, not a dev-only tool:

```bash
# Daily use: validate model after any operation
apr merge a.apr b.apr -o merged.apr
apr probar merged.apr --golden golden/ --assert

# CI: full validation with profiling
apr probar model.apr --golden golden/ --profile --assert --format json
```

### 6.2 Probar in Tiered Quality Gates

| Tier | Probar Usage |
|------|-------------|
| tier1 | — (too slow) |
| tier2 | `apr probar --golden tests/golden/ --assert` (fast golden check) |
| tier3 | `apr probar --golden tests/golden/ --profile --assert` (with brick profiling) |
| tier4 | Full playbook mutation + cross-backend golden validation |

### 6.3 Probar + BrickProfiler

When `--profile` is passed to `apr probar`, each layer snapshot also
records BrickProfiler timing. This catches both correctness regressions
(activation divergence) and performance regressions (slow bricks) in a
single pass.

---

## 7. Probar vs Other Testing Tools

| Tool | Purpose | Probar Relationship |
|------|---------|-------------------|
| `cargo test` | Unit + integration tests | Orthogonal — probar is E2E |
| `proptest` | Property-based testing | Used internally by probar |
| `criterion` | Micro-benchmarks | Probar uses for perf baselines |
| `cargo-mutants` | Code mutation testing | Probar adds playbook mutation |
| `Kani` | Bounded model checking | Contract-level, not E2E |
| `certeza` | Test methodology | Probar implements certeza tiers |

Probar is the **E2E validation layer** that sits on top of all other
testing. It validates that the assembled system (apr-cli → realizar →
trueno → GPU) produces correct, regression-free results end to end.

---

## 8. Feature Flags

```toml
[dev-dependencies]
jugar-probar = { version = "0.4", features = ["tui", "gpu", "proptest"] }
```

| Feature | What |
|---------|------|
| `browser` | CDP browser automation |
| `runtime` | WASM headless testing (wasmtime) |
| `tui` | TUI snapshot testing (default) |
| `gpu` | GPU kernel validation (trueno) |
| `media` | Visual regression with PNG/GIF |
| `proptest` | Property-based falsification |
| `docker` | Cross-browser via Docker |
| `llm` | LLM output scoring |
