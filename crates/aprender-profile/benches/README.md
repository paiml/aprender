# Renacer Performance Benchmarks (Sprint 36)

This directory contains performance benchmarks for Renacer, designed to measure overhead, detect regressions, and validate optimizations.

## Prerequisites

1. **Build Renacer in release mode**:
   ```bash
   cargo build --release
   ```

2. **Compile benchmark fixtures**:
   ```bash
   rustc benches/fixtures/syscall_heavy.rs -o benches/fixtures/syscall_heavy
   ```

## Running Benchmarks

### All Benchmarks
```bash
cargo bench
```

This will run all benchmark suites and generate HTML reports in `target/criterion/`.

### Specific Benchmark Suite
```bash
# Syscall overhead benchmarks
cargo bench --bench syscall_overhead

# OTLP export benchmarks
cargo bench --bench otlp_export

# Memory pool benchmarks
cargo bench --bench memory_pool
```

### Specific Benchmark
```bash
cargo bench --bench syscall_overhead -- native
cargo bench --bench otlp_export -- batch_export
cargo bench --bench memory_pool -- pool_cycle
```

### Save Baseline for Comparison
```bash
# Save current performance as baseline
cargo bench -- --save-baseline main

# After making changes, compare against baseline
cargo bench -- --baseline main
```

## Benchmark Suites

### 1. Syscall Overhead (`syscall_overhead.rs`)

Measures the performance overhead of Renacer's tracing compared to native execution.

**Benchmarks**:
- `native` - Baseline: fixture without any tracing
- `basic_tracing` - Renacer without OTLP export
- `with_statistics` - Renacer with `-c` flag
- `with_timing` - Renacer with `-T` flag
- `full_stack` - All features enabled (no OTLP)
- `overhead_comparison` - Side-by-side comparison
- `throughput` - Syscalls per second measurement

**Goals**:
- Basic tracing: <5% overhead vs. native
- Full stack: <10% overhead vs. native
- Throughput: >100K syscalls/sec

**Run**:
```bash
cargo bench --bench syscall_overhead
```

### 2. OTLP Export (`otlp_export.rs`)

Measures OTLP span export performance, focusing on batching strategies.

**Benchmarks**:
- `individual_export` - Baseline: one span at a time
- `batch_export` - Batch sizes: 16, 32, 64, 128, 256, 512, 1024, 2048
- `span_creation` - Cost of creating span data structures
- `span_serialization` - Protobuf encoding simulation
- `buffer_management` - Buffer operations (push/take)
- `realistic_workload` - 10K spans with batch size 512
- `queue_saturation` - High-pressure scenario
- `batch_vs_individual` - Direct comparison

**Goals**:
- Batch export: 40-60% faster than individual
- Throughput: >10K spans/sec
- Optimal batch size: 512 spans

**Run**:
```bash
cargo bench --bench otlp_export
```

### 3. Memory Pool (`memory_pool.rs`)

Measures memory pooling performance for span data structures.

**Benchmarks**:
- `heap_allocation` - Baseline: direct heap allocation
- `pool_allocation` - Object pool allocation
- `pool_cycle` - Single acquire/release cycle
- `pool_sizes` - Different pool sizes: 128, 256, 512, 1024, 2048, 4096
- `allocation_pressure` - Realistic workload with bursts
- `memory_footprint` - 10K spans comparison

**Goals**:
- Pool allocation: 20-30% faster than heap
- Acquire/release: <100ns per operation
- Memory footprint: Comparable to heap for 10K spans

**Run**:
```bash
cargo bench --bench memory_pool
```

## Benchmark Fixtures

### `syscall_heavy.rs`

A program that generates many file system syscalls:
- 100 file operations (create, write, read, remove)
- ~300 total syscalls (open, write, close, read, unlink)
- Predictable, deterministic workload

**Build**:
```bash
rustc benches/fixtures/syscall_heavy.rs -o benches/fixtures/syscall_heavy
```

## Interpreting Results

### HTML Reports

Criterion generates detailed HTML reports in `target/criterion/`:
- Line charts showing performance over time
- Violin plots showing distribution
- Regression detection
- Statistical analysis

Open `target/criterion/report/index.html` in a browser to view.

### Command-Line Output

```
syscall_overhead/native/syscall_heavy_native
                        time:   [145.23 ms 147.56 ms 149.89 ms]

syscall_overhead/basic_tracing/syscall_heavy_basic
                        time:   [152.11 ms 154.45 ms 156.79 ms]
                        change: [+4.2% +4.7% +5.1%] (p < 0.05)
```

- `time`: Median and 95% confidence interval
- `change`: Percentage change from previous run (if baseline exists)
- `p < 0.05`: Statistically significant change

### Overhead Calculation

```
Overhead = ((traced_time - native_time) / native_time) * 100%

Example:
Native: 147.56 ms
Traced: 154.45 ms
Overhead: ((154.45 - 147.56) / 147.56) * 100% = 4.67%
```

## Performance Goals (Sprint 36)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Basic tracing overhead | <5% | TBD | 🔄 |
| Full stack overhead | <10% | TBD | 🔄 |
| Syscall throughput | >100K/sec | TBD | 🔄 |
| OTLP export throughput | >10K spans/sec | TBD | 🔄 |
| Pool acquire/release | <100ns | TBD | 🔄 |
| Memory footprint (10K) | <50MB | TBD | 🔄 |

## Continuous Integration

Benchmarks should be run on every significant change to detect regressions:

```bash
# Before making changes
cargo bench -- --save-baseline before

# After making changes
cargo bench -- --baseline before

# Check for regressions
# If any benchmark shows >5% slowdown, investigate before merging
```

## Tips for Accurate Benchmarks

1. **Minimize background processes**: Close unnecessary applications
2. **Disable CPU frequency scaling**: Use performance governor
   ```bash
   sudo cpupower frequency-set --governor performance
   ```
3. **Run multiple times**: Criterion automatically runs multiple iterations
4. **Use release builds**: Always benchmark with `--release`
5. **Warm up**: Criterion includes warm-up iterations
6. **Stable environment**: Run on same machine for comparisons

## Troubleshooting

### "Failed to run renacer"

Ensure Renacer is built in release mode:
```bash
cargo build --release
```

### "Failed to run fixture"

Compile the benchmark fixtures:
```bash
rustc benches/fixtures/syscall_heavy.rs -o benches/fixtures/syscall_heavy
```

### Inconsistent Results

- Check for background processes consuming CPU
- Verify CPU frequency scaling is disabled
- Increase `sample_size` in benchmark code
- Increase `measurement_time` for longer-running benchmarks

## Future Enhancements

- [ ] Add I/O-heavy fixture (large file operations)
- [ ] Add compute-heavy fixture (CPU-bound workload)
- [ ] Add multi-process fixture (fork/clone scenarios)
- [ ] Add OTLP backend integration (requires Jaeger)
- [ ] Add memory profiling (valgrind/heaptrack integration)
- [ ] Add flamegraph generation for hotspot analysis
- [ ] Automated regression detection in CI

## References

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/index.html)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [How to Write Fast Rust Code](https://likebike.com/posts/How_To_Write_Fast_Rust_Code.html)
