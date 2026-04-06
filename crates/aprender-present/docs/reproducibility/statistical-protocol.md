# Statistical Analysis Protocol

**Status:** Active
**Last Updated:** 2026-01-12

## Purpose

Define statistical standards for all quantitative claims in presentar to ensure reproducibility and falsifiability.

## Sample Size Requirements

### Minimum Sample Sizes by Measurement Type

| Type | Min N | Rationale |
|------|-------|-----------|
| Microbenchmark (<1ms) | 1000 | High variance requires more samples |
| Render benchmark | 100 | Moderate variance |
| Integration test | 30 | Lower variance, higher cost |
| User study | 20 | Per Cohen's power analysis |

### Power Analysis

For detecting medium effect sizes (Cohen's d = 0.5) with 80% power and α = 0.05:

```
n = 2 * ((z_α/2 + z_β) / d)²
n = 2 * ((1.96 + 0.84) / 0.5)²
n ≈ 64 per group
```

## Confidence Interval Protocol

### Reporting Format

All measurements must include 95% confidence intervals:

```
metric: mean ± sem (95% CI: [lower, upper])

Example: render_time: 0.82ms ± 0.03ms (95% CI: [0.79ms, 0.85ms])
```

### Calculation Method

For normally distributed data:
```
CI = mean ± (t_crit * (stddev / √n))

where t_crit = t(α/2, n-1) for 95% CI
```

For non-normal data, use bootstrap confidence intervals with 10,000 resamples.

## Effect Size Standards

### Cohen's d for Continuous Outcomes

| d Value | Interpretation |
|---------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

### Reporting Template

```json
{
  "comparison": "ptop vs ttop",
  "metric": "render_time_ms",
  "ptop_mean": 0.82,
  "ttop_mean": 0.95,
  "cohens_d": 0.82,
  "interpretation": "large effect",
  "p_value": 0.001,
  "n_per_group": 1000
}
```

## Multiple Comparisons Correction

When making multiple comparisons, apply Bonferroni correction:

```
α_adjusted = α / k

where k = number of comparisons
```

For 5 comparisons at α = 0.05: α_adjusted = 0.01

## Outlier Handling

### Detection Method

Use Tukey's fences:
- Lower fence: Q1 - 1.5 * IQR
- Upper fence: Q3 + 1.5 * IQR

### Documentation Requirement

When removing outliers:
1. Report number removed
2. Report reason for removal
3. Show results with and without outliers

## Reproducibility Checklist

- [ ] Sample size meets minimum requirements
- [ ] 95% CI calculated and reported
- [ ] Effect size (Cohen's d) calculated
- [ ] P-value reported (if applicable)
- [ ] Multiple comparisons corrected
- [ ] Outlier handling documented
- [ ] Raw data archived with DVC
- [ ] Random seed documented

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Cumming, G. (2014). The New Statistics: Why and How
- Criterion.rs Documentation: https://bheisler.github.io/criterion.rs/
