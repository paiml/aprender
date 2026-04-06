# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contract-First Design

This project follows contract-first development with provable-contracts.
Contracts live in `../provable-contracts/contracts/trueno-db/`.
Run `pmat comply check` to validate contract compliance.

## Project Overview

**Trueno-DB** is a GPU-first, SIMD-fallback embedded analytics database built on Apache Arrow and Trueno. It provides high-performance aggregations with graceful degradation from GPU → AVX-512 → AVX2 → SSE2 → Scalar, including WASM/WebGPU support for browser deployment.

**Design Philosophy:**
- GPU-first compute with automatic fallback to SIMD
- Zero-copy operations via Arrow columnar format
- WASM-portable for browser analytics
- Multi-GPU scaling capability

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "query execution" --limit 10

# Find high-quality code
pmat query "sql parser" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "index scan" --faults

# Filter by complexity
pmat query "join algorithm" --max-complexity 10

# Cross-project search
pmat query "gpu kernel" --include-project ../trueno

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix query planner" -G
pmat query "fix query planner" --git-history

# Enrichment flags (combine freely)
pmat query "executor" --churn              # git volatility (commit count, churn score)
pmat query "scan operator" --duplicates          # code clone detection (MinHash+LSH)
pmat query "aggregate" --entropy             # pattern diversity (repetitive vs unique)
pmat query "query engine" --churn --duplicates --entropy --faults -G  # full audit
```

## Architecture

### Core Components

**Backend Selection Hierarchy:**
1. Multi-GPU distribution (if multiple GPUs available and workload supports it)
2. Single GPU via wgpu (for datasets >100K rows)
3. SIMD via Trueno (AVX-512 → AVX2 → Auto-detect)
4. Scalar fallback

**Storage Layer:**
- Apache Arrow columnar format (Int32Array, Float32Array, StringArray, etc.)
- Zero-copy GPU transfers (Arrow buffers → GPU VRAM)
- Parquet and CSV readers

**Query Engine:**
- SQL-like subset (SELECT, WHERE, GROUP BY, aggregations, JOINs, window functions)
- Cost-based backend dispatcher
- LRU cache for hot queries

**Compute Backends:**
- GpuBackend: wgpu compute shaders (Vulkan/Metal/DX12/WebGPU)
- TruenoBackend: SIMD acceleration via trueno crate
- WASM: WebGPU + SIMD128 for browser deployment

## Development Commands

### Project Setup
```bash
make build                            # Build the project
make build-release                    # Release build
make test                             # Run all tests
make lint                             # Lint with clippy (zero tolerance)
make check                            # Lint + test
```

### Quality Gates (EXTREME TDD)
```bash
# All of these must pass before any commit:
make test                             # 100% tests pass
make lint                             # Zero clippy warnings
make coverage                         # >90% code coverage required
make tdg                              # TDG score ≥B+ (85/100)
make mutants                          # ≥80% mutation kill rate (optional)
make quality-gate                     # Run all gates (lint + test + coverage)
```

**IMPORTANT**: Always use `make coverage` instead of raw `cargo llvm-cov` commands.
The Makefile handles mold linker conflicts automatically.

### WASM Build
```bash
cargo build --target wasm32-unknown-unknown --release
```

### Benchmarking
```bash
# Benchmarks are required for all performance claims
cargo bench                           # Run benchmarks
cargo bench --bench aggregations      # Specific benchmark suite
```

## Critical Dependencies

### trueno Integration
```toml
[dependencies]
trueno = "0.3.0"           # SIMD fallback (ALWAYS use crates.io version)
wgpu = "22"                # GPU compute
arrow = "53"               # Columnar format
parquet = "53"             # Parquet reader
sqlparser = "0.52"         # SQL parsing
tokio = { version = "1", features = ["full"] }
rayon = "1.8"              # CPU parallelism
```

**IMPORTANT:** Always use the latest trueno from crates.io, never git dependencies.

Check for updates:
```bash
cargo search trueno                   # Check latest version
cargo tree | grep trueno              # Current version
cargo update trueno                   # Update to latest
```

## Testing Requirements

### Backend Equivalence Testing
GPU results MUST exactly match SIMD and scalar results for correctness:

```rust
#[test]
fn test_sum_backend_equivalence() {
    let data = generate_test_data(1_000_000);

    let gpu_result = execute_on_backend(Backend::Gpu, &data);
    let simd_result = execute_on_backend(Backend::Trueno(Auto), &data);
    let scalar_result = execute_on_backend(Backend::Scalar, &data);

    assert_eq!(gpu_result, simd_result);
    assert_eq!(simd_result, scalar_result);
}
```

### Property-Based Testing
Use property tests for operations:
```rust
#[quickcheck]
fn prop_sum_commutative(xs: Vec<f32>) -> bool {
    let result1 = db.query("SELECT sum(x) FROM data").execute();
    let shuffled = shuffle(xs);
    let result2 = db.query("SELECT sum(x) FROM shuffled").execute();
    result1 == result2
}
```

### Performance Regression Tests
Every benchmark must prove claimed speedups:
```rust
// Target: 50-100x faster than CPU for 100M row SUM
assert!(gpu_time < cpu_time / 50.0);
```

## GPU Kernel Development

GPU operations are implemented in WGSL (WebGPU Shading Language):

```wgsl
// Example: Parallel sum reduction
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: atomic<u32>;

@compute @workgroup_size(256)
fn sum_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < arrayLength(&input)) {
        atomicAdd(&output, bitcast<u32>(input[idx]));
    }
}
```

### Performance Targets
| Operation | Speedup (vs CPU) | Requirement |
|-----------|------------------|-------------|
| SUM       | 50-100x          | Must meet   |
| AVG       | 50-100x          | Must meet   |
| COUNT     | 30-50x           | Must meet   |
| MIN/MAX   | 50-100x          | Must meet   |
| FILTER    | 20-40x           | Must meet   |

## Code Quality Standards

### Pre-commit Requirements
- All tests pass (cargo test --all-features)
- Zero clippy warnings (cargo clippy -- -D warnings)
- Code coverage >90%
- TDG score ≥B+ (85/100)
- Mutation testing ≥80% kill rate

### Pull Request Requirements
- Include benchmarks proving performance claims
- Add property-based tests for correctness
- Backend equivalence tests (GPU == SIMD == Scalar)
- Update CHANGELOG.md (keep-a-changelog format)

### Release Requirements
- Repository score ≥90/110 (pmat repo-score)
- Performance regression tests vs previous version
- WASM build succeeds
- Multi-GPU tests pass (if GPUs available)

## Implementation Phases

### Phase 1: Core Engine (Current)
- Arrow storage backend (Parquet, CSV readers)
- SQL parser (SELECT, WHERE, GROUP BY, aggregations)
- GPU kernels (sum, avg, count, min, max)
- SIMD fallback via Trueno
- 100+ unit tests with property-based and equivalence testing
- Benchmarks vs DuckDB, SQLite, Polars

### Phase 2: Multi-GPU
- Local multi-GPU data partitioning (range/hash)
- Cost-based query planner
- Multi-GPU aggregation with reduce
- 2 GPU vs 1 GPU vs CPU benchmarks

### Phase 3: Distribution
- gRPC worker protocol
- Distributed query execution
- Fault tolerance (retry, failover)
- Remote multi-GPU benchmarks

### Phase 4: WASM
- wasm32-unknown-unknown build target
- WebGPU backend integration
- Browser example dashboard
- WebGPU vs SIMD128 browser benchmarks

## Example Usage Patterns

### Embedded Rust
```rust
use trueno_db::Database;

let db = Database::builder()
    .backend(Backend::Gpu)
    .fallback(Backend::Trueno(trueno::Backend::Auto))
    .cache_size_mb(512)
    .build()?;

db.load_table("events", "data/events.parquet").await?;

let result = db.query(
    "SELECT category, sum(value) as total
     FROM events
     WHERE timestamp > '2025-11-01'
     GROUP BY category
     ORDER BY total DESC"
).execute().await?;
```

### WASM/Browser
```javascript
const db = new Database({
    backend: 'webgpu',
    fallback: 'simd128'
});
await db.loadTable('events', '/data/events.parquet');
const result = await db.query('SELECT category, sum(value) FROM events GROUP BY category');
```

## Dogfooding Opportunities

This database is designed to be used in:
- **assetsearch**: GPU-accelerated aggregations replacing PostgreSQL
- **assetgen**: Real-time AI model metadata dashboards
- **bashrs**: Command history analytics
- **auth-billing**: Real-time usage tracking
- **interactive.paiml.com**: WASM analytics dashboards

## Academic Foundations

Key papers informing the architecture:
- MonetDB/X100: Vectorized execution (CIDR 2005)
- HeavyDB: GPU database patterns (SIGMOD 2017)
- Apache Arrow: Columnar format (VLDB 2020)
- Volcano Optimizer: Cost-based optimization (IEEE 1993)
- Morsel-Driven Parallelism: NUMA-aware execution (SIGMOD 2014)

See docs/specifications/db-spec-v1.md section 8 for complete references.

## Backend Story Policy (CRITICAL - NEVER VIOLATE)

### Zero Tolerance Backend Requirements

**ALL operations in trueno-db MUST work on ALL backends:**

| Backend | Description | When Used |
|---------|-------------|-----------|
| **Scalar** | Reference implementation | Fallback, testing |
| **SIMD** | trueno Vector ops (SSE2/AVX/AVX2/AVX512/NEON) | Default CPU path |
| **GPU** | wgpu compute shaders (Vulkan/Metal/DX12/WebGPU) | Large datasets (>10MB) |
| **WASM** | SIMD128 + WebGPU | Browser deployment |

### Adding New Operations - Step by Step

When adding ANY new operation to trueno-db:

1. **Implement Scalar version first** (reference implementation)
2. **Implement SIMD version** using `trueno::Vector` ops
3. **Implement GPU version** in `src/gpu/` with WGSL shader
4. **Add equivalence test** to `tests/backend_story.rs`
5. **Update BackendDispatcher** FLOP estimation if needed
6. **Verify** with `cargo test --test backend_story`

### Enforcement Mechanisms

1. **Pre-commit hook**: Runs `cargo test --test backend_story` before every commit
2. **CI pipeline**: Blocks PRs that break backend story tests
3. **CLAUDE.md**: This policy is read by Claude Code for enforcement
4. **Code review**: Backend equivalence is mandatory review criteria

### Common Violations to Avoid

```rust
// BAD: GPU-only implementation
pub async fn new_aggregation(&self) -> Result<f32> {
    self.gpu_engine.compute()  // NO! What about SIMD/Scalar?
}

// GOOD: All backends supported
pub fn new_aggregation_scalar(data: &[f32]) -> f32 { ... }
pub fn new_aggregation_simd(data: &[f32]) -> f32 { Vector::from_slice(data).operation()... }
pub async fn new_aggregation_gpu(&self, data: &Arrow) -> Result<f32> { ... }
```

### Backend Story Tests

Run these tests before ANY commit:

```bash
# CPU backends (always runs)
cargo test --test backend_story

# With GPU backends (requires hardware)
cargo test --test backend_story --features gpu
```

## Toyota Way Principles

### Jidoka (Built-in Quality)
- EXTREME TDD: Tests before implementation
- Pre-commit hooks enforce quality gates
- Backend equivalence: GPU == SIMD == Scalar

### Kaizen (Continuous Improvement)
- Benchmarks required for all optimizations
- Performance regression tests detect slowdowns
- Mutation testing finds coverage gaps

### Respect for People
- Simple DuckDB-like API
- Graceful degradation (works on any hardware)
- Clear error messages explaining GPU failures and fallbacks


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
