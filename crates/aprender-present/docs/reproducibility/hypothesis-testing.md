# Hypothesis Testing Protocol

Statistical rigor for Popperian falsification (D1/D2 criteria).

## Sample Size Requirements

| Metric Type | Minimum Samples | Recommended | Rationale |
|-------------|----------------|-------------|-----------|
| Latency (ms) | 100 | 1000 | Reduce noise from system jitter |
| Memory (MB) | 30 | 100 | Account for GC/allocation patterns |
| CPU (%) | 60 | 300 | Capture load variations |
| Throughput | 50 | 500 | Steady-state measurement |

### Power Analysis

For detecting a 10% performance difference with 80% power and α=0.05:

```
n = (Z_α/2 + Z_β)² × 2σ² / δ²

Where:
- Z_α/2 = 1.96 (95% confidence)
- Z_β = 0.84 (80% power)
- σ = estimated standard deviation
- δ = minimum detectable difference
```

## Confidence Intervals

All performance metrics must report 95% confidence intervals:

```json
{
  "mean_ms": 0.82,
  "stddev_ms": 0.03,
  "ci_lower_ms": 0.79,
  "ci_upper_ms": 0.85,
  "confidence_level": 0.95,
  "samples": 1000
}
```

### Bootstrap CI (Preferred)

For non-normal distributions:

```rust
fn bootstrap_ci(samples: &[f64], n_bootstrap: usize, alpha: f64) -> (f64, f64) {
    let mut means = Vec::with_capacity(n_bootstrap);
    let mut rng = SeededRng::new(get_seed());

    for _ in 0..n_bootstrap {
        let bootstrap_sample: Vec<f64> = (0..samples.len())
            .map(|_| samples[rng.next_u64() as usize % samples.len()])
            .collect();
        means.push(bootstrap_sample.iter().sum::<f64>() / samples.len() as f64);
    }

    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lower_idx = (n_bootstrap as f64 * alpha / 2.0) as usize;
    let upper_idx = (n_bootstrap as f64 * (1.0 - alpha / 2.0)) as usize;

    (means[lower_idx], means[upper_idx])
}
```

## Effect Size Calculations

### Cohen's d

For comparing two conditions:

```
d = (M₁ - M₂) / σ_pooled

Where:
σ_pooled = √[(σ₁² + σ₂²) / 2]
```

| Effect Size | d Value | Interpretation |
|-------------|---------|----------------|
| Small | 0.2 | Minimal practical difference |
| Medium | 0.5 | Noticeable difference |
| Large | 0.8 | Substantial difference |

### Practical Example

```json
{
  "comparison": "ptop vs ttop",
  "metrics": {
    "render_latency": {
      "ptop_mean_ms": 0.82,
      "ttop_mean_ms": 0.95,
      "cohens_d": 0.82,
      "interpretation": "Large effect size"
    }
  }
}
```

## Hypothesis Pre-Registration

Before running benchmarks, document:

1. **Null Hypothesis (H₀)**: No performance difference exists
2. **Alternative Hypothesis (H₁)**: ptop is ≥10% faster than baseline
3. **Alpha Level**: 0.05
4. **Sample Size**: Calculated via power analysis
5. **Effect Size of Interest**: Minimum practically important difference

Example pre-registration:

```yaml
# Pre-registered hypothesis: full_render_latency
hypothesis:
  h0: "ptop full render ≥ ttop full render"
  h1: "ptop full render < ttop full render by ≥10%"
  alpha: 0.05
  power: 0.80
  sample_size: 1000
  effect_size_threshold: 0.10
  registered_date: "2026-01-12"
  analysis_plan: "Two-sample t-test with Welch correction"
```

## Multiple Comparison Correction

When testing multiple hypotheses, apply Bonferroni or Holm-Bonferroni correction:

```
α_adjusted = α / n_comparisons
```

For 5 benchmarks at α=0.05:
- Adjusted α = 0.05 / 5 = 0.01

## Reporting Requirements

Every benchmark report must include:

1. ☑ Sample size (n)
2. ☑ Mean and standard deviation
3. ☑ 95% confidence interval
4. ☑ Effect size (Cohen's d for comparisons)
5. ☑ P-value (if hypothesis testing)
6. ☑ Random seed used
7. ☑ Hardware/software environment
