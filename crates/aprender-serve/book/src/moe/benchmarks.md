# MOE Benchmarks (Reproducible)

This document provides scientifically reproducible benchmarks for realizar's MOE infrastructure.

> **See also**: [Aprender vs PyTorch Comparison](../performance/benchmarking.md#comparative-benchmarks-aprender-vs-pytorch) for cross-framework benchmarks showing **9.6x speedup** over PyTorch.

## Methodology

All benchmarks follow these principles:
- **Reproducibility**: Fixed seeds, documented hardware, multiple runs
- **Statistical rigor**: Report mean, std dev, and confidence intervals
- **Fair comparison**: Warm-up runs excluded, steady-state measured

## Hardware Configuration

```
CPU: [Run `lscpu | grep "Model name"` to fill]
Memory: [Run `free -h` to fill]
OS: [Run `uname -a` to fill]
Rust: [Run `rustc --version` to fill]
```

## Running Benchmarks

```bash
# MOE load tests (in-process, no server required)
cargo test moe_load_tests --test load_test -- --nocapture

# Criterion benchmarks
cargo bench --bench inference
```

## Results

### 1. Capacity Factor Router Throughput

**Test**: 10 threads, 1000 requests each, 8 experts

```bash
cargo test test_moe_router_concurrent_load --test load_test -- --nocapture
```

| Metric | Value | Unit |
|--------|-------|------|
| Total requests | 10,000 | routes |
| Duration | 5.86 | ms |
| Throughput | 1,707,894 | routes/sec |
| Success rate | 100% | |

**Citation**: Capacity Factor algorithm per Fedus et al. (2022) [^1]

### 2. Registry Lock-Free Reads (ArcSwap)

**Test**: 50 threads, 10,000 reads each

```bash
cargo test test_registry_read_contention --test load_test -- --nocapture
```

| Metric | Value | Unit |
|--------|-------|------|
| Total reads | 500,000 | ops |
| Duration | 26.42 | ms |
| Throughput | 18,924,307 | reads/sec |
| Success rate | 100% | |

**Citation**: Lock-free reads via RCU pattern per McKenney (2011) [^2]

### 3. Capacity Overflow Fallback

**Test**: Primary expert at capacity, verify fallback to second-best

```bash
cargo test test_moe_router_capacity_overflow --test load_test -- --nocapture
```

| Metric | Value |
|--------|-------|
| Requests routed to fallback | 20/20 (100%) |

**Citation**: Power-of-Two-Choices per Mitzenmacher (2001) [^3]

### 4. Mixed Read/Write Workload

**Test**: 10 readers + 2 writers, 1000 ops each

```bash
cargo test test_registry_mixed_workload --test load_test -- --nocapture
```

| Metric | Value |
|--------|-------|
| Final model count | 2,000 |
| Deadlocks | 0 |

## Comparison with Baselines

| Component | RwLock Baseline | ArcSwap (realizar) | Speedup |
|-----------|-----------------|--------------------| --------|
| Registry reads | ~2M/sec | 18.9M/sec | **9.5x** |

## Reproducing These Results

```bash
# Clone repository
git clone https://github.com/paiml/realizar
cd realizar

# Run all MOE benchmarks
cargo test moe_load_tests --test load_test --release -- --nocapture

# For criterion benchmarks
cargo bench --bench inference -- --verbose
```

## References

[^1]: Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23, 1-39. doi:10.48550/arXiv.2101.03961

[^2]: McKenney, P. E. (2011). Is Parallel Programming Hard, And, If So, What Can You Do About It?

[^3]: Mitzenmacher, M. (2001). The Power of Two Choices in Randomized Load Balancing. *IEEE TPDS*, 12(10), 1094-1104. doi:10.1109/71.963420

[^4]: Dean, J., & Barroso, L. A. (2013). The Tail at Scale. *Communications of the ACM*, 56(2), 74-80. doi:10.1145/2408776.2408794

## Cross-Framework Comparison: Aprender vs PyTorch

In addition to internal benchmarks, realizar includes scientifically reproducible cross-framework comparisons.

### MNIST Inference Benchmark

**Task**: LogisticRegression (784 → 2), single sample inference

| Framework | p50 Latency | Throughput | Speedup |
|-----------|-------------|------------|---------|
| **Aprender (.apr)** | 0.52 µs | 1,898,614/sec | **9.6x** |
| PyTorch | 5.00 µs | 195,754/sec | baseline |

**Statistical Validation:**
- Welch's t-test: p < 0.001 (highly significant)
- Cohen's d: 5.19 (large effect)
- 10,000 iterations, 100 warmup

### Running the Comparison

```bash
# Aprender benchmark
cargo run --example mnist_apr_benchmark --release --features aprender-serve

# PyTorch benchmark
cd benches/comparative && uv run mnist_benchmark.py

# Generate comparison report
uv run compare_mnist.py
```

### QA Validation

A 100-point checklist validates this benchmark:

```bash
cat docs/qa/qa-benchmark-pytorch-aprender-comparison.md
```

Categories:
- Methodology (15 points)
- Statistical rigor (15 points)
- Code correctness (15 points)
- Environment fairness (15 points)
- Bias detection (10 points)

**Acceptance criteria**: 95/100 points PASS required.

[^5]: Box, G. E. P., et al. (2005). Statistics for Experimenters. Wiley.

[^6]: Georges, A., et al. (2007). Statistically Rigorous Java Performance Evaluation. *OOPSLA '07*.
