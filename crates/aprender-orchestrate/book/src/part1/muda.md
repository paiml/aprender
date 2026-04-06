# Muda: Waste Elimination

**Muda** (無駄) means "waste" -- any activity that consumes resources without producing value. The Toyota Production System identifies seven types of waste and systematically eliminates each one.

## The Seven Wastes in Software

| Toyota Waste | Software Equivalent | Batuta Mitigation |
|-------------|---------------------|-------------------|
| **Overproduction** | Building features nobody uses | Targeted transpilation of requested files only |
| **Waiting** | Idle CPU during I/O or serial builds | Parallel tool execution via Repartir |
| **Transport** | Unnecessary data movement | Cost-based backend selection (5x PCIe rule) |
| **Overprocessing** | Redundant analysis passes | Incremental analysis with state caching |
| **Inventory** | Stale build artifacts | Deterministic builds, no artifact hoarding |
| **Motion** | Context switching between tools | Single `batuta transpile` entry point |
| **Defects** | Bugs that require rework | Jidoka quality gates at every phase |

## Waste Elimination in Batuta

### Caching and Incremental Compilation

Batuta tracks pipeline state in `.batuta-state.json`. When a phase completes successfully, it is not re-run unless inputs change.

```bash
# First run: all 5 phases execute
$ batuta transpile --input ./project
Phase 1: Analysis       [2.1s]
Phase 2: Transpilation   [8.4s]
Phase 3: Optimization    [3.2s]
Phase 4: Validation      [5.1s]
Phase 5: Deployment      [1.0s]

# Second run: only changed phases re-execute
$ batuta transpile --input ./project
Phase 1: Analysis       [cached]
Phase 2: Transpilation   [1.2s]  # Only modified files
Phase 3: Optimization    [cached]
Phase 4: Validation      [5.1s]  # Re-validates changed output
Phase 5: Deployment      [1.0s]
```

### Cost Circuit Breakers

GPU dispatch is expensive. Batuta prevents waste by applying the Gregg 5x rule: GPU is only selected when the compute benefit exceeds five times the data transfer cost.

```rust
// Muda: avoid wasteful GPU transfers for small operations
let backend = if data_size > threshold && compute_ratio > 5.0 {
    Backend::Gpu
} else {
    Backend::Simd  // SIMD avoids PCIe transfer entirely
};
```

### Eliminating Redundant Analysis

PMAT quality analysis uses hash-based invalidation. If source files have not changed, the cached TDG score is reused. Cold cache takes approximately 7 seconds; warm cache responds in under 100 milliseconds. Invalidation triggers are explicit: Cargo.toml changes, git HEAD moves, or TTL expiration.

### Eliminating Unnecessary Transpilation

Batuta only transpiles files that match a known source language with an available transpiler. Files already in Rust or belonging to unsupported languages are skipped:

```bash
$ batuta transpile --input ./mixed_project
Skipping: src/lib.rs (already Rust)
Transpiling: scripts/preprocess.py (via Depyler)
Transpiling: vendor/parser.c (via Decy)
```

The goal is not zero time per phase, but zero time spent on work that does not change the output.

## Benefits

1. **Faster iteration** -- cached phases complete in milliseconds
2. **Lower cost** -- circuit breakers prevent unnecessary GPU spend
3. **Focused effort** -- only changed files are reprocessed
4. **Predictable builds** -- deterministic state tracking eliminates surprise rebuilds

---

**Navigate:** [Table of Contents](../SUMMARY.md)
