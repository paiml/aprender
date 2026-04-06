# Statistical Rigor Documentation

This document describes the statistical methods and requirements for simular simulations and benchmarks.

## Sample Size Justification

### Minimum Sample Sizes

| Analysis Type | Minimum n | Justification |
|---------------|-----------|---------------|
| Benchmark timing | 100 | CLT convergence for timing distributions |
| Monte Carlo | 10,000 | π estimation to 2 decimal places |
| Property tests | 256 | proptest default, covers edge cases |
| Verification runs | 3 | Reproducibility check |

### Power Analysis

For detecting effect size d = 0.5 (medium effect):
- α = 0.05 (Type I error rate)
- β = 0.20 (Type II error rate, power = 0.80)
- **Required n ≈ 64 per group**

## Confidence Intervals

### Benchmark Reporting

All benchmarks report 95% confidence intervals:

```
Mean: 1.234 ms ± 0.056 ms (95% CI)
```

### Calculation Method

Using Student's t-distribution for small samples:

```
CI = x̄ ± t(α/2, n-1) × (s / √n)
```

Where:
- x̄ = sample mean
- s = sample standard deviation
- n = sample size
- t(α/2, n-1) = t-critical value

### Implementation

```rust
fn confidence_interval(samples: &[f64], confidence: f64) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    let std_err = (variance / n).sqrt();

    // t-critical for 95% CI with n-1 degrees of freedom
    let t_crit = t_distribution_critical(0.025, n as u32 - 1);

    (mean - t_crit * std_err, mean + t_crit * std_err)
}
```

## Effect Sizes

### Cohen's d for Comparisons

When comparing algorithms:

```
d = (μ₁ - μ₂) / σ_pooled
```

Interpretation:
| d | Interpretation |
|---|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Practical Significance

Report both statistical significance (p-value) and effect size:

```
GRASP vs Random: p < 0.001, d = 1.23 (large effect)
```

## Hypothesis Testing

### Null Hypothesis Framework

Each falsifiable claim follows:

1. **H₀** (Null): Algorithm does not meet threshold
2. **H₁** (Alternative): Algorithm meets threshold
3. **α** = 0.05 (significance level)
4. **Test**: One-sample t-test or z-test

### Example: GRASP Optimality Gap

```
H₀: gap ≥ 25%
H₁: gap < 25%

Result: gap = 18.3%, t = -4.56, p < 0.001
Conclusion: Reject H₀, GRASP achieves < 25% gap
```

## Monte Carlo Error Analysis

### Convergence Rate

For Monte Carlo π estimation:

```
Error ∝ 1/√n
```

Expected errors:
| Samples | Expected Error |
|---------|----------------|
| 1,000 | ~0.03 |
| 10,000 | ~0.01 |
| 100,000 | ~0.003 |
| 1,000,000 | ~0.001 |

### Variance Reduction

Techniques implemented:
- Antithetic variates
- Control variates
- Importance sampling

## Benchmark Methodology

### Warm-up Protocol

1. Discard first 10 iterations (JIT, cache warming)
2. Run 100+ measurement iterations
3. Filter outliers using MAD (Median Absolute Deviation)

### Outlier Detection

Using MAD-based filtering:

```rust
fn filter_outliers(samples: &[f64]) -> Vec<f64> {
    let median = percentile(samples, 0.5);
    let mad = median_absolute_deviation(samples);
    let threshold = 3.0 * mad;

    samples.iter()
        .filter(|&x| (x - median).abs() <= threshold)
        .copied()
        .collect()
}
```

### Reporting Requirements

Each benchmark must report:
- Mean and standard deviation
- 95% confidence interval
- Sample size
- Hardware specification
- Software versions

## Reproducibility Requirements

### Numerical Reproducibility

- IEEE 754 strict mode
- Deterministic floating-point operations
- Fixed iteration order (no parallel non-determinism)

### Statistical Reproducibility

- Seeds documented for all RNG usage
- Same results within floating-point tolerance
- Cross-platform reproducibility verified

## Quality Metrics

### Test Coverage as Statistical Power

| Coverage | Interpretation |
|----------|----------------|
| < 80% | Insufficient evidence |
| 80-90% | Acceptable |
| 90-95% | Good |
| > 95% | Excellent |

### Mutation Testing as Sensitivity

Mutation score indicates test sensitivity:
- 80%+ required for high-stakes code
- Each surviving mutant = potential undetected bug

## Effect Size Guidelines

### Minimum Meaningful Effect Sizes

These thresholds define when performance differences are practically significant:

| Metric | Meaningful Threshold | Justification |
|--------|---------------------|---------------|
| TSP tour length | 5% improvement | Below noise for small instances |
| Energy conservation | 10x improvement | Order of magnitude required |
| Execution time | 20% speedup | User-perceptible difference |
| Memory usage | 30% reduction | Significant for constrained environments |

### Interpreting Benchmark Results

```
If |d| < 0.2:  Negligible difference - do not claim improvement
If |d| < 0.5:  Small effect - requires justification
If |d| < 0.8:  Medium effect - meaningful improvement
If |d| >= 0.8: Large effect - significant improvement
```

### Practical vs Statistical Significance

A result can be:
1. **Statistically significant but not practical**: p < 0.05, d < 0.2
2. **Practical but not statistically significant**: d > 0.5, p > 0.05
3. **Both**: p < 0.05, d > 0.5 (ideal)

Always report both p-value AND effect size.

### Effect Size Calculations Used

| Comparison | Formula | Notes |
|------------|---------|-------|
| Two groups | Cohen's d | Pooled standard deviation |
| Pre-post | Cohen's d_z | Paired samples |
| Correlations | r | Direct effect size |
| Proportions | h | Arcsine transformation |

### Benchmark Reporting Template

```markdown
## Performance Comparison: A vs B

| Metric | A (mean±CI) | B (mean±CI) | Difference | Cohen's d | p-value |
|--------|-------------|-------------|------------|-----------|---------|
| Time   | 1.23±0.05ms | 1.45±0.06ms | -15.2%     | 0.73      | <0.001  |

**Interpretation**: Medium effect (d=0.73), statistically significant (p<0.001).
Practically meaningful as threshold is 20% speedup.
```

## References

1. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
2. Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap
3. Georges, A. et al. (2007). Statistically Rigorous Java Performance Evaluation
4. Kalibera, T. & Jones, R. (2013). Rigorous Benchmarking in Reasonable Time
5. Wasserstein, R. & Lazar, N. (2016). ASA Statement on p-Values
