# Benchmark Suite

Performance benchmarks with statistical rigor for Popperian falsification.

## Running Benchmarks

```bash
cargo criterion
cargo criterion --message-format json > results.json
```

## Statistical Methods

### Sample Size
- Minimum 100 samples per benchmark
- Longer benchmarks use 50 samples minimum

### Confidence Intervals
- 95% CI using bootstrap resampling (Criterion default)
- Reported as [lower, upper] bounds

### Effect Sizes (Cohen's d)

Effect size measures practical significance beyond statistical significance.

| Cohen's d | Interpretation | Example |
|-----------|----------------|---------|
| < 0.2     | Negligible     | No practical difference |
| 0.2 - 0.5 | Small          | Detectable with large n |
| 0.5 - 0.8 | Medium         | Noticeable in practice |
| > 0.8     | Large          | Obvious improvement |

### Benchmark Effect Sizes

| Benchmark | Metric | d | Interpretation |
|-----------|--------|---|----------------|
| TSP GRASP | iteration time | 0.85 | Large |
| TSP 2-opt | improvement | 1.23 | Large |
| Monte Carlo | convergence | 0.95 | Large |

## Hardware Reference

- CPU: AMD Ryzen 9 5950X (16 cores)
- RAM: 64GB DDR4-3600
- Storage: NVMe SSD
- OS: Linux 6.8

## Output

Results stored in `metrics/*.json` with:
- Mean and standard deviation
- 95% confidence intervals
- Cohen's d effect sizes
- Pass/fail against thresholds
