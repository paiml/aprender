# Benchmark Methodology

Statistical rigor for reproducible benchmarking (D1/D2 criteria).

## Sample Size Requirements

| Benchmark Type | Minimum n | Recommended n | Justification |
|---------------|-----------|---------------|---------------|
| Latency | 100 | 1000 | CLT convergence |
| Throughput | 50 | 500 | Steady-state |
| Memory | 30 | 100 | GC variance |
| CPU | 60 | 300 | Load variance |

### Power Analysis

Effect size detection at 80% power, α=0.05:

```
n = 2 × (1.96 + 0.84)² × σ² / δ²
n ≈ 15.7 × (σ/δ)²
```

For 10% difference detection with σ/mean ≈ 0.05:
- Required n ≈ 63 samples
- Recommended n = 100 (safety margin)

## Confidence Intervals

All metrics report 95% CI using bootstrap method (10,000 resamples):

```json
{
  "metric": "full_render_latency",
  "mean_ms": 0.82,
  "ci_95_lower": 0.79,
  "ci_95_upper": 0.85,
  "samples": 1000,
  "bootstrap_iterations": 10000
}
```

### CI Interpretation

- CI overlapping zero: No significant effect
- CI not overlapping baseline: Significant change
- CI width > 20% of mean: High variance, increase samples

## Effect Size Reporting

All comparisons include Cohen's d:

| d Value | Interpretation |
|---------|----------------|
| < 0.2 | Negligible |
| 0.2-0.5 | Small |
| 0.5-0.8 | Medium |
| > 0.8 | Large |

### Required Metrics

Every benchmark report MUST include:

1. ✓ Sample size (n)
2. ✓ Mean ± standard deviation
3. ✓ 95% confidence interval
4. ✓ Effect size (Cohen's d for comparisons)
5. ✓ Hardware/software environment
6. ✓ Random seed used
7. ✓ Git commit hash

## Warmup Protocol

To achieve steady-state measurements:

1. **Warmup iterations**: 100 (discarded)
2. **Measurement iterations**: As per sample size table
3. **Cooldown**: 1 second between benchmark groups
4. **Isolation**: `--test-threads=1` for sequential execution

## Baseline Comparisons

Performance regressions are detected using:

```
regression = (current_mean - baseline_mean) / baseline_stddev > 2.0
```

A change is flagged if:
- Mean shifts by > 2 standard deviations
- 95% CI does not overlap baseline CI
- Effect size |d| > 0.5 (medium)

## Reproducibility Checklist

- [ ] Fixed random seed (CRITERION_SEED=42)
- [ ] Pinned Rust version (rust-toolchain.toml)
- [ ] CPU governor set to performance
- [ ] No background processes
- [ ] Sufficient warmup iterations
- [ ] Sample size meets minimum requirements
