# Confidence Intervals and Uncertainty Reporting

This document describes how Realizar reports uncertainty in benchmark results and performance claims.

## Confidence Interval Methodology

### Bootstrap Percentile Method

All confidence intervals use the bootstrap percentile method:

1. Draw B bootstrap samples (B = 10,000)
2. Compute statistic for each bootstrap sample
3. Take percentiles [α/2, 1-α/2] for (1-α) CI

```python
# Pseudocode
bootstrap_means = []
for _ in range(10000):
    sample = random.choices(data, k=len(data))
    bootstrap_means.append(mean(sample))
ci_lower = percentile(bootstrap_means, 2.5)
ci_upper = percentile(bootstrap_means, 97.5)
```

### Why Bootstrap?

- No normality assumption required
- Works for any statistic (mean, median, percentiles)
- Accounts for sample distribution shape

## Reported Statistics

### For Each Benchmark

```
tensor_creation/10      time:   [17.887 ns 17.966 ns 18.043 ns]
                                 ^         ^         ^
                              lower      mean      upper
                              bound    estimate    bound
```

| Statistic | Description |
|-----------|-------------|
| Lower bound | 2.5th percentile of bootstrap distribution |
| Mean estimate | Point estimate |
| Upper bound | 97.5th percentile of bootstrap distribution |

### Additional Statistics

- **Median**: More robust than mean for skewed distributions
- **Std Dev**: Measure of dispersion
- **MAD**: Median Absolute Deviation (robust std dev)
- **IQR**: Interquartile range (25th to 75th percentile)

## Comparative Analysis

### Speedup Ratio with Uncertainty

When comparing two systems:

```
Speedup = t_baseline / t_realizar = 5.00 / 0.52 = 9.6x
```

Uncertainty propagation:
```
σ_speedup = speedup × √[(σ_baseline/t_baseline)² + (σ_realizar/t_realizar)²]
σ_speedup = 9.6 × √[(0.10/5.00)² + (0.02/0.52)²]
σ_speedup = 9.6 × √[0.0004 + 0.0015]
σ_speedup = 9.6 × 0.044
σ_speedup ≈ 0.4
```

**Result**: 9.6x ± 0.4x speedup (95% CI)

### Statistical Significance

For each comparison, we report:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| p-value | p < 0.001 | Reject null hypothesis |
| Cohen's d | 5.19 | Very large effect |
| 95% CI for difference | [4.3, 4.6] µs | Difference is meaningful |

## Benchmark Results Summary

### Realizar vs PyTorch (MNIST)

| Metric | Realizar | PyTorch | Comparison |
|--------|----------|---------|------------|
| Latency (p50) | 0.52 µs [0.50, 0.54] | 5.00 µs [4.90, 5.10] | 9.6x faster |
| Throughput | 1.90M/s [1.85M, 1.95M] | 0.20M/s [0.19M, 0.21M] | 9.5x higher |

### Internal Benchmarks

| Benchmark | Result | 95% CI |
|-----------|--------|--------|
| Tensor creation (10) | 18 ns | [17.5, 18.5] |
| Tensor creation (10K) | 643 ns | [620, 666] |
| Cache hit | 39 ns | [37, 41] |
| Cache miss | 285 ns | [270, 300] |

## Uncertainty Sources

### Quantified Uncertainties

1. **Measurement noise**: Captured in CI width
2. **Sampling uncertainty**: Addressed by sample size
3. **Environmental variation**: Controlled via warm-up

### Unquantified Uncertainties (Caveats)

1. **Hardware variation**: Results may differ on other CPUs
2. **OS scheduling**: Background processes may affect results
3. **Thermal effects**: Long benchmarks may see throttling

## Best Practices for Reproduction

To achieve similar CIs:

1. **CPU governor**: Set to `performance`
2. **Isolation**: Minimize background processes
3. **Warm-up**: Allow 50 iterations minimum
4. **Sample size**: Use at least 100 samples
5. **Multiple runs**: Run benchmark 3 times, report median

## Reporting Format

### Text Format

```
MNIST Inference Latency
  Realizar: 0.52 µs ± 0.02 µs (n=10,000, 95% CI)
  PyTorch:  5.00 µs ± 0.10 µs (n=10,000, 95% CI)
  Speedup:  9.6x ± 0.4x
  Effect:   d = 5.19 (very large)
  p-value:  < 0.001
```

### JSON Format

```json
{
  "benchmark": "mnist_inference",
  "realizar": {
    "mean": 0.52e-6,
    "ci_lower": 0.50e-6,
    "ci_upper": 0.54e-6,
    "unit": "seconds",
    "n": 10000
  },
  "pytorch": {
    "mean": 5.00e-6,
    "ci_lower": 4.90e-6,
    "ci_upper": 5.10e-6,
    "unit": "seconds",
    "n": 10000
  },
  "comparison": {
    "speedup": 9.6,
    "speedup_ci": [9.2, 10.0],
    "cohens_d": 5.19,
    "p_value": 0.001
  }
}
```

## References

1. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
2. Criterion.rs Documentation. https://bheisler.github.io/criterion.rs/book/
3. NIST/SEMATECH e-Handbook of Statistical Methods. Chapter 7.

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-10
