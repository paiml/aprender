# MNIST Inference Benchmark: Aprender vs PyTorch

## Executive Summary

**Aprender (.apr) is 9.7x faster than PyTorch** for MNIST LogisticRegression inference.

- Statistical significance: p < 0.001
- Effect size: large (Cohen's d = 6.63)

## Methodology

Following Box et al. (2005) and Georges et al. (2007) guidelines for
statistically rigorous performance evaluation:

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Input dimensions | 784 (28x28 MNIST) |
| Output classes | 2 |
| Training samples | 1000 |
| Warmup iterations | 100 |
| Benchmark iterations | 10000 |

## Environment

### Aprender (Rust)
- Version: 0.2.1
- Rust: stable
- Platform: linux x86_64
- CPU: AMD Ryzen Threadripper 7960X 24-Cores

### PyTorch (Python)
- Version: 2.9.1+cu128
- Python: 3.13.1
- Platform: Linux 6.8.0-87-generic
- CPU: AMD Ryzen Threadripper 7960X 24-Cores

## Results

### Latency Comparison

| Metric | Aprender | PyTorch | Speedup |
|--------|----------|---------|---------|
| p50 (us) | 0.52 | 5.03 | **9.7x** |
| p95 (us) | 0.53 | 5.21 | 9.8x |
| p99 (us) | 0.53 | 8.03 | 15.2x |
| Mean (us) | 0.53 | 5.11 | 9.6x |
| Std Dev (us) | 0.07 | 0.97 | - |

### Throughput Comparison

| Framework | Inferences/sec |
|-----------|----------------|
| Aprender | 1,896,707 |
| PyTorch | 195,551 |
| **Ratio** | **9.7x** |

## Statistical Analysis

### Welch's t-test

Testing null hypothesis: mean(Aprender) = mean(PyTorch)

- t-statistic: -469.0406
- p-value: 0.000000
- Significant at α=0.05: **Yes (reject null)**

### Effect Size (Cohen's d)

- Cohen's d: 6.63
- Interpretation: **large** effect

### 95% Confidence Intervals

| Framework | Mean (us) | 95% CI |
|-----------|-----------|--------|
| Aprender | 0.53 | [0.53, 0.53] |
| PyTorch | 5.11 | [5.09, 5.13] |

## Analysis: Why Aprender is Faster

1. **Zero Python overhead**: No interpreter, no GIL, no dynamic dispatch
2. **Native compilation**: Rust compiles to optimized machine code
3. **No tensor framework overhead**: Direct matrix operations without PyTorch's abstraction layers
4. **Predictable performance**: No JIT warmup, consistent latency
5. **Memory efficiency**: No Python object overhead, minimal allocations

## Reproducibility

```bash
# Run Aprender benchmark
cargo run --example mnist_apr_benchmark --release --features aprender-serve

# Run PyTorch benchmark
cd benches/comparative
uv sync
uv run mnist_benchmark.py

# Generate comparison report
uv run compare_mnist.py
```

## References

1. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters*. Wiley.
2. Georges, A., Buytaert, D., & Eeckhout, L. (2007). Statistically Rigorous Java Performance Evaluation. OOPSLA '07.
3. Aprender: https://github.com/paiml/aprender
4. Realizar: https://github.com/paiml/realizar

---
*Generated: 2025-11-27 20:23:46 UTC*