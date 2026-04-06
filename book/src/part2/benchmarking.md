# Benchmarking

Benchmarking measures the performance of the transpiled Rust binary against the original program. It is the final check in Phase 4, providing quantitative evidence that the migration preserved or improved performance.

## Benchmark Method

Batuta runs both binaries multiple times and computes average execution time:

```
Original program   x3 iterations --> avg: 1.24s
Transpiled program x3 iterations --> avg: 0.31s
                                     Speedup: 4.0x
```

The number of iterations is configurable. Three iterations is the default to balance accuracy against validation time.

## Benchmark Report

```bash
$ batuta validate --benchmark

Performance Benchmark
---------------------
Original:    1.243s (avg of 3 runs)
Transpiled:  0.312s (avg of 3 runs)
Speedup:     3.99x

Breakdown:
  Run 1: 1.251s vs 0.315s
  Run 2: 1.238s vs 0.310s
  Run 3: 1.241s vs 0.311s

Status: PASS (informational -- regression does not fail validation)
```

## Criterion Integration

For micro-benchmarking individual functions, transpiled projects can include Criterion benchmarks. Criterion provides statistical analysis, regression detection, and HTML reports:

```bash
# Run Criterion benchmarks in the transpiled project
cd rust-output && cargo bench
```

## Regression Detection

While the Phase 4 benchmark is informational (it does not fail the pipeline), Criterion benchmarks can detect regressions between runs:

```
matmul_1024x1024    time: [312.45 us 315.21 us 318.02 us]
                    change: [+2.1% +3.4% +4.8%] (p = 0.02 < 0.05)
                    Performance has regressed.
```

## Before/After Comparison

| Metric | Original (Python) | Transpiled (Rust) | Change |
|--------|------------------|-------------------|--------|
| Startup time | 450ms | 2ms | 225x faster |
| Peak memory | 128 MB | 12 MB | 10.7x less |
| Throughput | 1.2K ops/s | 48K ops/s | 40x faster |
| Binary size | N/A (interpreter) | 3.2 MB | Standalone |

## CLI Usage

```bash
# Run performance benchmark
batuta validate --benchmark

# With custom iteration count
batuta validate --benchmark --iterations 10

# Save benchmark results to file
batuta validate --benchmark --output benchmark-results.json
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
