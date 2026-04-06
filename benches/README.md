# Benchmarks

This directory contains performance benchmarks for presentar-terminal.

## Running Benchmarks

```bash
# Run all benchmarks
cargo criterion

# Run specific benchmark
cargo criterion --bench terminal

# Save baseline
cargo criterion --save-baseline baseline-name

# Compare against baseline
cargo criterion --baseline baseline-name
```

## Benchmark Configuration

### Sample Sizes

| Benchmark | Samples | Warmup | Measurement |
|-----------|---------|--------|-------------|
| full_render | 1000 | 100 | 5s |
| diff_update | 1000 | 100 | 5s |
| widget_layout | 500 | 50 | 3s |

### Environment Variables

```bash
# Fixed seed for reproducibility
export PRESENTAR_BENCH_SEED=12345

# Disable Criterion debug output
export CRITERION_DEBUG=0

# Sample size override
export CRITERION_SAMPLE_SIZE=100
```

## Statistical Reporting

All benchmarks report:
- Mean ± standard error
- 95% confidence interval
- Throughput (iterations/second)

### Example Output

```
full_render             time:   [0.79 ms 0.82 ms 0.85 ms]
                        change: [-2.1% +0.5% +3.0%] (p = 0.15)
                        No change in performance detected.
Found 0 outliers among 1000 measurements (0.00%)
```

## Baseline Management

Baselines are stored in `data/benchmarks/baselines/`:

```
baselines/
├── 2026-01-12.json     # Dated baseline with hardware info
├── latest.json -> 2026-01-12.json
└── README.md
```

## Hardware Specifications

Reference hardware for official benchmarks:

| Component | Specification |
|-----------|---------------|
| CPU | AMD Threadripper 7960X (48 cores) |
| RAM | 128GB DDR5-5200 |
| Storage | NVMe RAID-0 |
| OS | Ubuntu 24.04 LTS |
| Kernel | 6.8.0-90-generic |

## References

- [Criterion.rs Book](https://bheisler.github.io/criterion.rs/book/)
- [Statistical Protocol](../docs/reproducibility/statistical-protocol.md)
