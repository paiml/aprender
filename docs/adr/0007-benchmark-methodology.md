# ADR-0007: Benchmark Methodology

## Status

Accepted

## Date

2024-12-01

## Context

Performance claims require rigorous benchmarking. Issues to address:
- Statistical significance
- Reproducibility across systems
- Fair comparison with baselines
- Avoiding common benchmarking pitfalls

## Decision

Follow scientific benchmarking methodology based on MLPerf and academic standards.

## Methodology

### Sample Size Calculation

Power analysis for 10% detectable difference:
```
n = 2 × (Z_α/2 + Z_β)² × (CV/δ)²
n = 2 × (1.96 + 0.84)² × (0.05/0.10)²
n ≈ 4 minimum → 100 actual (for robustness)
```

### Warm-up Protocol

1. Run 50 iterations without measurement
2. Verify steady-state reached (Mann-Whitney U test)
3. Begin timed measurement

### Confidence Intervals

Bootstrap percentile method (10,000 resamples):
```rust
fn bootstrap_ci(samples: &[f64], alpha: f64) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let mut bootstrap_means: Vec<f64> = (0..10_000)
        .map(|_| {
            let sample: Vec<f64> = (0..samples.len())
                .map(|_| samples[rng.gen_range(0..samples.len())])
                .collect();
            sample.iter().sum::<f64>() / sample.len() as f64
        })
        .collect();
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = ((alpha / 2.0) * 10_000.0) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * 10_000.0) as usize;

    (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
}
```

### Effect Size Reporting

Cohen's d for practical significance:
```
d = (μ₁ - μ₂) / σ_pooled

| d | Interpretation |
|---|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |
| >1.2 | Very large |
```

## Implementation

### Criterion.rs Configuration

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "inference"
harness = false
```

```rust
fn benchmark_inference(c: &mut Criterion) {
    let model = Model::demo();

    c.bench_function("mnist_inference", |b| {
        b.iter(|| {
            black_box(model.infer(black_box(&input)))
        })
    });
}
```

### Output Format

```
tensor_creation/10      time:   [17.887 ns 17.966 ns 18.043 ns]
                                 ^         ^         ^
                              lower      mean      upper
                              bound    estimate    bound
```

## Consequences

### Positive
- Statistically rigorous claims
- Reproducible across systems
- Clear methodology documentation
- Comparable with academic standards

### Negative
- Longer benchmark runs (~10 minutes for full suite)
- More complex reporting
- May catch fewer edge cases than longer soak tests

## Validation

**Falsifiable claim**: Benchmark results are reproducible within 5% variance across runs.

**Test**: Run full benchmark suite 10 times, compute CV.

## References

1. Georges, A., et al. (2007). Statistically rigorous Java performance evaluation.
2. Hoefler, T., & Belli, R. (2015). Scientific benchmarking of parallel computing systems.
3. MLCommons. MLPerf Inference Rules.
