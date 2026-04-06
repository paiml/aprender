# Benchmarking Methodology

Trueno-DB uses [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical benchmarking following Toyota Way principle of **Genchi Genbutsu** (Go and See).

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench storage_benchmarks

# Run specific benchmark
cargo bench --bench storage_benchmarks parquet_loading
```

## Benchmark Suites

### Storage Benchmarks (`storage_benchmarks.rs`) ✅

Measures Arrow storage backend performance:
- Parquet file loading
- Morsel iteration overhead
- RecordBatch slicing (zero-copy)
- Memory calculation performance

### Aggregation Benchmarks (`aggregations.rs`) 🚧

Future: GPU vs SIMD vs Scalar comparison for:
- SUM aggregations
- AVG, COUNT, MIN, MAX
- Target: 50-100x GPU speedup for 100M+ rows

### Backend Comparison (`backend_comparison.rs`) 🚧

Future: Backend equivalence verification:
- GPU == SIMD == Scalar (correctness)
- Performance comparison
- Cost model validation

## Statistical Rigor

Criterion.rs provides:
- **Warm-up phase** (3 seconds) - eliminate cold cache effects
- **100 samples** - statistical significance
- **Outlier detection** - identify anomalies
- **Confidence intervals** - quantify measurement uncertainty
- **Regression detection** - track performance over time

## Interpreting Results

### Time Measurements

```
parquet_loading/10000   time:   [124.53 µs 125.77 µs 127.17 µs]
                                  ^^^^^^^^  ^^^^^^^^  ^^^^^^^^
                                  Lower CI  Estimate  Upper CI
```

- **Estimate**: Best measured performance (median)
- **Confidence Interval**: 95% confidence bounds
- **Outliers**: Measurements outside expected range

### Throughput Calculation

```rust
// For 10,000 rows in 125.77 µs:
throughput = 10_000 / 0.12577 ms = 79,500 rows/ms
           = 79.5M rows/second
```

## Toyota Way Principles

### Genchi Genbutsu (Go and See)

✅ **Measure actual performance** - Don't guess, benchmark
- Real Parquet files (not synthetic data)
- Multiple dataset sizes (1K, 10K, 100K, 1M rows)
- Realistic workloads (storage → morsel → GPU queue)

### Kaizen (Continuous Improvement)

✅ **Track performance over time**
- Criterion saves historical data
- Regression detection on every run
- Identify performance regressions early

Example output:
```
parquet_loading/10000   time:   [125.77 µs 126.34 µs 126.93 µs]
                        change: [-2.3421% -1.8934% -1.4128%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### Muda (Waste Elimination)

✅ **Identify bottlenecks before optimizing**
- Measure morsel iteration overhead
- Quantify zero-copy benefits
- Validate architectural assumptions

## Benchmark-Driven Development

1. **RED**: Write failing performance test
   ```rust
   // Target: < 1ms for 100K rows
   assert!(duration < Duration::from_millis(1));
   ```

2. **GREEN**: Implement until benchmark passes
   ```
   parquet_loading/100000  time:   [881.1 µs ...]  ✅ PASS
   ```

3. **REFACTOR**: Optimize with benchmarks as safety net
   - Change implementation
   - Re-run benchmarks
   - Ensure no regression

## Performance Targets

### Phase 1 (Current)

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Parquet loading (100K) | < 1 ms | 881 µs | ✅ |
| Morsel iteration | < 500 ns | 119 ns | ✅ |
| Batch slicing | < 200 ns | 108 ns | ✅ |

### Phase 2 (Future)

| Component | Target | Status |
|-----------|--------|--------|
| GPU SUM (100M rows) | < 100 ms | 🚧 |
| Backend selection | < 10 µs | 🚧 |
| JIT compilation | < 1 ms | 🚧 |

## Profiling Integration

For detailed performance analysis:

```bash
# CPU profiling with perf
cargo bench --bench storage_benchmarks --profile-time 60

# Memory profiling with valgrind
valgrind --tool=massif target/release/deps/storage_benchmarks-*

# Flame graph generation
cargo flamegraph --bench storage_benchmarks
```

## CI/CD Integration

Benchmarks run on every PR:
- Detect performance regressions
- Require < 5% slowdown for approval
- Historical tracking in `target/criterion/`

## Next Steps

- [Storage Benchmark Results](./storage-results.md)
- [Backend Comparison](./backend-comparison.md)
- [Optimization Techniques](./optimization.md)
