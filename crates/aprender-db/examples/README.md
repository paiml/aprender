# Trueno-DB Examples

This directory contains comprehensive examples demonstrating Trueno-DB's features and performance.

## Quick Start

All examples can be run with:

```bash
cargo run --example <name>
```

For optimal performance, use release mode:

```bash
cargo run --example <name> --release
```

## Available Examples

### 1. Basic Usage (`basic_usage.rs`)

**Purpose**: Introduction to storage engine and morsel iteration

**Demonstrates**:
- Loading data with `StorageEngine`
- OLAP append-only write pattern
- Morsel-based iteration (128MB chunks for out-of-core execution)
- Schema validation

**Run**:
```bash
cargo run --example basic_usage
```

**Key Concepts**:
- Columnar storage (Arrow/Parquet)
- Morsel iteration prevents GPU VRAM OOM
- OLAP vs OLTP design patterns

---

### 2. Top-K Selection (`topk_selection.rs`)

**Purpose**: High-performance Top-K selection algorithm

**Demonstrates**:
- O(N log K) heap-based algorithm vs O(N log N) full sort
- 28.75x speedup for K=10, N=1M (measured in release build)
- Descending (largest K values) and ascending (smallest K values)
- Support for Int32, Int64, Float32, Float64 columns

**Run**:
```bash
cargo run --example topk_selection --release
```

**Performance**:
- **1M rows**: <80ms (release build)
- **Algorithm**: Min-heap for descending, max-heap for ascending
- **Use case**: `ORDER BY ... LIMIT` queries

---

### 3. Backend Selection (`backend_selection.rs`)

**Purpose**: Cost-based GPU vs SIMD dispatcher

**Demonstrates**:
- Physics-based cost model (5x rule)
- PCIe Gen4 x16 bandwidth: 32 GB/s
- GPU compute throughput: 100 GFLOP/s (conservative)
- Decision tree based on arithmetic intensity

**Run**:
```bash
cargo run --example backend_selection
```

**Algorithm**:
1. If data < 10 MB → SIMD (transfer overhead dominates)
2. Calculate PCIe transfer time = bytes / 32 GB/s
3. Estimate GPU compute time = FLOPs / 100 GFLOP/s
4. If compute > 5x transfer → GPU, otherwise → SIMD

**Toyota Way**: Genchi Genbutsu (Go and See) - physics-based decision making

---

### 4. SIMD Acceleration (`simd_acceleration.rs`)

**Purpose**: Demonstrate SIMD backend via trueno crate

**Demonstrates**:
- SIMD feature detection (AVX-512, AVX2, SSE2)
- Auto-vectorization with compiler hints
- Performance comparison: scalar vs SIMD
- Trueno crate integration

**Run**:
```bash
cargo run --example simd_acceleration --release
```

**SIMD Vector Widths**:
- AVX-512: 64 bytes (16 × f32 or 8 × f64)
- AVX2: 32 bytes (8 × f32 or 4 × f64)
- SSE2: 16 bytes (4 × f32 or 2 × f64)

**Expected Speedup**: 2-8x vs scalar operations

---

### 5. Complete Pipeline (`complete_pipeline.rs`)

**Purpose**: End-to-end workflow demonstration

**Demonstrates**:
- Complete data pipeline from load to Top-K
- All Phase 1 MVP features integrated
- Performance metrics and throughput reporting
- Pretty-printed output with box-drawing characters

**Run**:
```bash
cargo run --example complete_pipeline --release
```

**Pipeline Steps**:
1. **Data Loading**: Create 5M row dataset
2. **Storage Engine**: OLAP append-only pattern
3. **Morsel Iteration**: 128MB chunks for out-of-core execution
4. **Backend Selection**: Cost-based GPU vs SIMD
5. **Top-K Selection**: O(N log K) heap algorithm

**Measured Performance** (5M rows, release build):
- **Data load**: ~420ms
- **Top-K selection**: ~5.5ms
- **Throughput**: ~906 M rows/sec

---

## Phase 1 MVP Features (v0.1.0)

All examples demonstrate features available in the Phase 1 MVP:

✅ **Implemented**:
- Arrow/Parquet storage engine
- Morsel-based iteration (128 MB chunks)
- OLAP write pattern (`append_batch`)
- Backend dispatcher (cost-based selection)
- Top-K selection (heap-based algorithm)
- SIMD integration (via trueno crate)

❌ **Deferred to Phase 2**:
- Actual GPU compute kernels (wgpu shaders)
- GPU device initialization
- PCIe bandwidth runtime calibration
- Multi-GPU data partitioning

## Performance Notes

### Debug vs Release

Debug builds are **significantly slower** due to:
- No compiler optimizations
- Bounds checking
- Debug assertions

Always use `--release` for performance testing:

```bash
cargo run --example topk_selection --release
```

### Expected Performance (Release Build)

| Example | Dataset | Time | Throughput |
|---------|---------|------|------------|
| basic_usage | 1M rows | ~50ms | 550 MB/s |
| topk_selection | 1M rows | ~80ms | 28.75x speedup |
| complete_pipeline | 5M rows | ~430ms | 906 M rows/sec |

### Hardware Considerations

SIMD performance depends on CPU features:
- **Best**: AVX-512 (2014+ Intel Xeon, some consumer CPUs)
- **Good**: AVX2 (2013+ Intel, 2017+ AMD)
- **Universal**: SSE2 (all x86-64 CPUs)

## Toyota Way Principles Demonstrated

### Jidoka (Built-in Quality)
- EXTREME TDD: All examples have corresponding tests
- Backend equivalence: GPU == SIMD == Scalar (when implemented)

### Kaizen (Continuous Improvement)
- Top-K optimization: O(N log N) → O(N log K)
- Morsel iteration: Prevents OOM, enables out-of-core

### Muda (Waste Elimination)
- SIMD: Process multiple elements per instruction
- Morsel paging: Only load data when needed

### Poka-Yoke (Mistake Proofing)
- OLAP contract: Prevents OLTP misuse
- Schema validation: Catch errors early

### Genchi Genbutsu (Go and See)
- Cost model based on real PCIe measurements
- Benchmarks validate claimed speedups

## Troubleshooting

### Example won't build

Make sure you have the latest dependencies:

```bash
cargo update
cargo build --examples
```

### Performance is slower than expected

1. Use release mode: `--release`
2. Check CPU SIMD features: Run `simd_acceleration` example
3. Verify you're not thermal throttling

### Out of memory

The examples use large datasets (1-5M rows). If you have <4GB RAM:
- Reduce dataset size in example source code
- Or run with `ulimit -v` memory limits

## Contributing

To add a new example:

1. Create `examples/your_example.rs`
2. Add documentation header with purpose and usage
3. Test with both debug and release builds
4. Update this README with your example
5. Ensure it demonstrates a unique feature or use case

## References

- [Trueno-DB Documentation](../docs/)
- [CHANGELOG](../CHANGELOG.md)
- [Toyota Way Principles](../docs/specifications/db-spec-v1.md)
