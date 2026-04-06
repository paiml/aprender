# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contract-First Design

This project follows contract-first development with provable-contracts.
Contracts live in `../provable-contracts/contracts/trueno-graph/`.
Run `pmat comply check` to validate contract compliance.

## Project Overview

**trueno-graph** is a GPU-first embedded graph database for code analysis (call graphs, dependencies, AST traversals). Built on PAIML's proven infrastructure (trueno, aprender), it provides 10-250× speedups through hybrid CPU/GPU execution.

**Status**: Phases 1-4 complete (CPU algorithms, GPU acceleration, community detection, pattern matching). Phase 5 planned (GPU memory paging, advanced patterns).

**Quality Standards**:
- **EXTREME TDD**: ≥95% test coverage (currently 98%+), zero SATD
- **Quality tools**: pmat + certeza + bashrs enforcement
- **Zero tolerance**: No clippy warnings with `-D warnings`

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "graph traversal" --limit 10

# Find high-quality code
pmat query "pagerank" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "edge indexing" --faults

# Filter by complexity
pmat query "shortest path" --max-complexity 10

# Cross-project search
pmat query "simd acceleration" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix cycle detection" -G
pmat query "fix cycle detection" --git-history

# Enrichment flags (combine freely)
pmat query "graph query" --churn              # git volatility (commit count, churn score)
pmat query "index builder" --duplicates          # code clone detection (MinHash+LSH)
pmat query "traversal" --entropy             # pattern diversity (repetitive vs unique)
pmat query "graph algorithm" --churn --duplicates --entropy --faults -G  # full audit
```

## Build & Development Commands

### Basic Development
```bash
make build          # Debug build (--all-features)
make release        # Optimized release build
make test           # Run all tests (CPU + GPU if hardware available)
make bench          # Run Criterion benchmarks
make clean          # Clean build artifacts
```

### Quality Gates
```bash
make lint           # Check clippy + formatting (CI mode)
make lint-fix       # Auto-fix clippy + formatting issues
make fmt            # Format code with rustfmt
make coverage       # Generate coverage report (≥95% required)
make quality        # Run pmat quality gates + bashrs validation
make mutation-test  # Run mutation testing (≥80% score)
make validate-all   # Full validation (quality + test + bench + mutation)
```

### Running Tests
```bash
# All tests
cargo test --all-features

# Single test (filter by name)
cargo test test_name_here

# GPU tests (requires hardware)
cargo test --features gpu

# Specific module
cargo test --lib storage::csr

# Integration tests
cargo test --test '*'
```

### Benchmarks
```bash
# CPU benchmarks
cargo bench --bench graph_algorithms

# GPU benchmarks (requires hardware)
cargo bench --bench gpu_algorithms --features gpu

# Phase 4 benchmarks (community detection, pattern matching)
cargo bench --bench phase4_algorithms

# Save baseline
make bench-save
```

## Architecture & Code Structure

### Core Architecture

**Three-layer design**:
1. **Storage Layer** (`src/storage/`): CSR (Compressed Sparse Row) graph representation
2. **Algorithm Layer** (`src/algorithms/`): Graph algorithms leveraging aprender
3. **GPU Layer** (`src/gpu/`): Optional GPU acceleration via wgpu (feature flag)

### Storage Layer (`src/storage/`)

**CSR Format** (Compressed Sparse Row):
- **Forward CSR**: O(1) outgoing edge queries via `row_offsets` + `col_indices`
- **Reverse CSR**: O(1) incoming edge queries via `rev_row_offsets` + `rev_col_indices`
- **Parquet persistence**: DuckDB-inspired columnar storage (via Arrow)

```rust
// Example: src/storage/csr.rs
CsrGraph {
    row_offsets: Vec<u32>,      // Forward CSR offsets
    col_indices: Vec<u32>,      // Forward CSR targets
    edge_weights: Vec<f32>,     // Forward CSR weights
    rev_row_offsets: Vec<u32>,  // Reverse CSR offsets (Phase 3.1)
    rev_col_indices: Vec<u32>,  // Reverse CSR sources
    rev_edge_weights: Vec<f32>, // Reverse CSR weights
}
```

**Key files**:
- `src/storage/csr.rs`: Core CSR graph data structure
- `src/storage/parquet.rs`: Parquet serialization/deserialization

### Algorithm Layer (`src/algorithms/`)

**Algorithms** (Phase 1-4):
- **Traversal** (`traversal.rs`): BFS, find_callers (reverse BFS)
- **PageRank** (`pagerank.rs`): Sparse matrix ops via aprender
- **Community Detection** (`louvain.rs`): Louvain clustering
- **Pattern Matching** (`pattern.rs`): Anti-pattern detection (God Class, Circular Dependencies, Dead Code)

**Design principle**: Delegates sparse matrix operations to aprender (shared SIMD primitives via trueno).

### GPU Layer (`src/gpu/`) - Optional Feature

**GPU acceleration** (enabled via `--features gpu`):
- **Device management** (`device.rs`): wgpu device initialization
- **Buffer management** (`buffer.rs`): CSR data upload to GPU VRAM
- **GPU BFS** (`bfs.rs`): Level-synchronous WGSL compute shader
- **GPU PageRank** (`pagerank.rs`): SpMV iteration with buffer ping-pong
- **Shaders** (`shaders/`): WGSL compute shaders (bfs_simple.wgsl, pagerank.wgsl)

**Performance**: 10-250× speedups vs NetworkX baseline (requires wgpu-capable hardware).

## Dependencies

### Core Dependencies
- **trueno** (0.16.4): SIMD primitives (shared with aprender)
- **aprender** (0.27.5): Sparse matrix operations, PageRank
- **arrow** (53): RecordBatch zero-copy
- **parquet** (53): Persistent storage
- **tokio** (1): Async runtime

### GPU Dependencies (Optional)
- **wgpu** (22): WebGPU API for compute shaders
- **bytemuck** (1): Zero-copy buffer casting
- **futures-intrusive** (0.5): Async GPU operations

### Development Dependencies
- **criterion** (0.6): Benchmarking framework
- **proptest** (1.9): Property-based testing
- **serial_test** (3): Sequential test execution

## Quality Enforcement

### Coverage Requirements
- **Minimum**: 95% line coverage (enforced by `make coverage`)
- **Current**: 98%+ coverage
- **Report**: `target/coverage/html/index.html` (after `make coverage`)

### Linting Standards
```bash
# Zero warnings enforced
cargo clippy --all-features -- -D warnings

# Workspace lints (Cargo.toml):
# - correctness: deny
# - suspicious: warn
# - perf: warn
# - unwrap_used: warn (prefer Result<T, E>)
# - expect_used: warn (prefer anyhow context)
```

### Pre-commit Hooks
```bash
make install-hooks  # Install pmat + bashrs pre-commit hooks
```

Hooks validate:
- pmat complexity (≤20 per function)
- pmat dead-code analysis
- pmat SATD detection (fail on any)
- bashrs Makefile validation
- clippy + formatting

## GPU Development

### GPU Feature Flag
```toml
# Enable GPU acceleration
[dependencies]
trueno-graph = { version = "0.1", features = ["gpu"] }
```

### GPU Tests & Benchmarks
```bash
# GPU tests (automatically skipped if no GPU)
cargo test --features gpu

# GPU benchmarks (automatically skipped if no GPU)
cargo bench --bench gpu_algorithms --features gpu
```

**Important**: GPU tests use `#[ignore]` attribute and are automatically skipped in CI environments without GPU hardware.

### GPU Architecture
- **WGSL shaders**: `src/gpu/shaders/*.wgsl` (compute shaders)
- **Zero-copy uploads**: CSR data → GPU VRAM via `GpuCsrBuffers`
- **Async execution**: Non-blocking GPU compute with futures-intrusive

## Academic Foundation

Based on 10 peer-reviewed papers:
1. **Gunrock** (Wang et al., ACM ToPC 2017): GPU graph traversal primitives
2. **GraphBLAST** (Yang et al., ACM ToMS 2022): GPU linear algebra for graphs
3. **DuckDB** (Raasveldt et al., SIGMOD 2019): Columnar storage patterns
4. **Page et al.** (1999): PageRank algorithm
5. **Ligra** (Shun & Blelloch, PPoPP 2013): Frontier-based traversal

See `docs/specifications/graph-db-spec.md` for full citation list and design rationale.

## Examples

**Examples** are located in `examples/`:
- `simple_graph.rs`: Basic CSR graph construction
- `graph_algorithms.rs`: BFS, PageRank, find_callers
- `comprehensive_demo.rs`: Full feature demonstration (Phases 1-4)

Run examples:
```bash
cargo run --example simple_graph
cargo run --example graph_algorithms
cargo run --example comprehensive_demo
```

## Testing Strategy

### Test Organization
- **Unit tests**: Inline with modules (`#[cfg(test)]` blocks)
- **Integration tests**: `tests/` directory (if present)
- **Property-based tests**: Using proptest for CSR invariants
- **GPU tests**: Marked with `#[ignore]`, run via `cargo test --features gpu`

### GPU Test Strategy
GPU tests are marked `#[ignore]` and skip gracefully if hardware unavailable:
```rust
#[test]
#[ignore] // Run explicitly: cargo test --features gpu
async fn test_gpu_bfs() { ... }
```

## Common Patterns

### Error Handling
- Use `anyhow::Result<T>` for public APIs
- Avoid `.unwrap()` and `.expect()` (clippy warnings enabled)
- Prefer `?` operator with anyhow context

### Async Operations
- Parquet I/O is async (requires tokio runtime)
- GPU operations are async (futures-intrusive)
- Tests use `#[tokio::test]` attribute

### Performance
- Release profile: `opt-level = 3`, LTO enabled, single codegen unit
- Bench profile: Inherits release, keeps symbols for flamegraph profiling

## Documentation

```bash
make docs  # Open rustdoc in browser (cargo doc --all-features --no-deps --open)
```

**Key docs**:
- `docs/specifications/graph-db-spec.md`: Full specification with citations
- `GPU_BFS_STATUS.md`: GPU implementation details (if present)
- Inline rustdoc: All public APIs documented

## Backend Story Policy (CRITICAL - NEVER VIOLATE)

### Zero Tolerance Backend Requirements

**ALL operations in trueno-graph MUST work on ALL backends:**

| Backend | Description | When Used |
|---------|-------------|-----------|
| **CPU/SIMD** | aprender sparse ops (trueno SIMD underneath) | Default path |
| **GPU** | wgpu compute shaders (BFS, PageRank) | Large graphs (>10K nodes) |
| **Scalar** | Fallback when SIMD unavailable | Testing, compatibility |

### Adding New Algorithms - Step by Step

When adding ANY new algorithm to trueno-graph:

1. **Implement CPU version first** using aprender/trueno (SIMD-accelerated)
2. **Add unit tests** verifying correctness
3. **If performance-critical**, implement GPU version in `src/gpu/`
4. **Add equivalence test** to `tests/backend_story.rs` (CPU == GPU)
5. **Verify** with `cargo test --test backend_story`

### Enforcement Mechanisms

1. **Pre-commit hook**: Runs `cargo test --test backend_story` before every commit
2. **CI pipeline**: Blocks PRs that break backend story tests
3. **CLAUDE.md**: This policy is read by Claude Code for enforcement
4. **Code review**: Backend equivalence is mandatory review criteria

### Common Violations to Avoid

```rust
// BAD: GPU-only algorithm
pub async fn new_algorithm(&self) -> Result<Vec<u32>> {
    self.gpu_device.compute()  // NO! What about CPU fallback?
}

// GOOD: CPU + optional GPU acceleration
pub fn new_algorithm_cpu(graph: &CsrGraph) -> Result<Vec<u32>> { ... }

#[cfg(feature = "gpu")]
pub async fn new_algorithm_gpu(graph: &CsrGraph) -> Result<Vec<u32>> { ... }
```

### Backend Story Tests

Run these tests before ANY commit:

```bash
# CPU backends (always runs)
cargo test --test backend_story

# With GPU backends (requires hardware)
cargo test --test backend_story --features gpu
```

## Related Projects

**PAIML Infrastructure**:
- **trueno**: SIMD primitives (foundation for aprender)
- **aprender**: Machine learning library with sparse matrix ops
- **trueno-db**: Columnar database (storage patterns inspired graph persistence)

**Dependency relationship**:
```
trueno-graph → aprender → trueno (SIMD)
            → parquet/arrow (storage)
            → wgpu (GPU, optional)
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
