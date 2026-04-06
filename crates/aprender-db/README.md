<div align="center">
<h1 align="center">trueno-db</h1>

<img src=".github/trueno-db-hero.svg" alt="trueno-db" width="600">

**GPU-first embedded analytics database built on Apache Arrow and Parquet.**

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)


**GPU-First Embedded Analytics with SIMD Fallback**

[![CI](https://github.com/paiml/trueno-db/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/trueno-db/actions)
[![Crates.io](https://img.shields.io/crates/v/trueno-db.svg)](https://crates.io/crates/trueno-db)
[![Documentation](https://docs.rs/trueno-db/badge.svg)](https://docs.rs/trueno-db)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

</div>

---

GPU-first embedded analytics database with graceful degradation: **GPU → SIMD → Scalar**

## Features

- **Cost-based dispatch**: GPU only when compute > 5x transfer time
- **Morsel-based paging**: Out-of-core execution (128MB chunks)
- **JIT WGSL compiler**: Kernel fusion for single-pass execution
- **GPU kernels**: SUM, MIN, MAX, COUNT, AVG, fused filter+sum
- **SIMD fallback**: Trueno integration (AVX-512/AVX2/SSE2)
- **SQL interface**: SELECT, WHERE, aggregations, ORDER BY, LIMIT

## Installation

```toml
[dependencies]
trueno-db = "0.3"

# Optional: GPU acceleration
trueno-db = { version = "0.3", features = ["gpu"] }
```

## Quick Start

```rust
use trueno_db::query::{QueryEngine, QueryExecutor};
use trueno_db::storage::StorageEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = StorageEngine::load_parquet("data/events.parquet")?;
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse(
        "SELECT COUNT(*), SUM(value), AVG(value) FROM events WHERE value > 100"
    )?;
    let result = executor.execute(&plan, &storage)?;
    Ok(())
}
```

## Performance

**SIMD Aggregation (1M rows, AMD Threadripper 7960X)**:

| Operation | SIMD | Scalar | Speedup |
|-----------|------|--------|---------|
| SUM | 228µs | 634µs | 2.78x |
| MIN | 228µs | 1,048µs | 4.60x |
| MAX | 228µs | 257µs | 1.13x |
| AVG | 228µs | 634µs | 2.78x |

## Architecture

```
┌─────────────────────────────────────────────┐
│              SQL Interface                   │
│         (QueryEngine / Parser)               │
├─────────────────────────────────────────────┤
│           Query Executor                     │
│    (cost-based dispatch, morsel paging)       │
├──────────┬──────────┬───────────────────────┤
│  GPU     │  SIMD    │  Scalar               │
│  (WGSL)  │ (Trueno) │  (fallback)           │
├──────────┴──────────┴───────────────────────┤
│         Storage Engine                       │
│   (columnar, Parquet, morsel-based I/O)      │
└─────────────────────────────────────────────┘
```

- **Query Layer**: SQL parser produces logical plans, cost-based optimizer selects GPU/SIMD/Scalar backend
- **Execution Layer**: Morsel-based paging (128MB chunks) enables out-of-core processing
- **GPU Backend**: JIT-compiled WGSL kernels with fused filter+aggregate passes
- **SIMD Backend**: Trueno-powered AVX-512/AVX2/SSE2 vectorized aggregations
- **Storage Layer**: Columnar storage with Parquet I/O and late materialization

## API Reference

### `StorageEngine`

Load and manage columnar data:

```rust
let storage = StorageEngine::load_parquet("data.parquet")?;
let storage = StorageEngine::from_batches(batches);
```

### `QueryEngine` / `QueryExecutor`

Parse and execute SQL queries:

```rust
let engine = QueryEngine::new();
let plan = engine.parse("SELECT SUM(value) FROM data WHERE id > 10")?;
let result = QueryExecutor::new().execute(&plan, &storage)?;
```

### `GpuEngine`

Direct GPU kernel access (requires `gpu` feature):

```rust
let gpu = GpuEngine::new().await?;
let sum = gpu.sum_i32(&data).await?;
let filtered = gpu.fused_filter_sum(&data, threshold, "gt").await?;
```

### `TopK`

Heap-based top-K selection for ORDER BY ... LIMIT queries:

```rust
let top = top_k_descending(&batch, column_idx, k)?;
```

## Examples

```bash
cargo run --example sql_query_interface --release
cargo run --example benchmark_shootout --release
cargo run --example gaming_leaderboards --release
cargo run --example gpu_aggregations --features gpu --release
```

## Testing

```bash
cargo test --lib          # Unit tests (66 tests)
cargo test                # All tests including integration
cargo test --doc          # Documentation tests
make quality-gate         # Full quality gate: lint + test + coverage
```

Property-based tests cover storage invariants and top-K correctness via `proptest`.

## Development

```bash
make build           # Build
make test            # Run tests
make quality-gate    # lint + test + coverage
make bench-comparison
```

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) guide for details.


## MSRV

Minimum Supported Rust Version: **1.75**

## See Also

- [Cookbook](examples/) — 14 runnable examples

## License

MIT
