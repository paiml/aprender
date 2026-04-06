# Sample Size Protocol

D1 Criterion: Statistical rigor through proper sample sizing.

## Minimum Sample Sizes

| Test Type | Minimum n | Standard n | Rationale |
|-----------|-----------|------------|-----------|
| Unit test | 1 | 1 | Deterministic, single execution |
| Property test | 100 | 1000 | Statistical confidence |
| Benchmark | 100 | 1000 | Reduce noise variance |
| Integration | 10 | 50 | End-to-end validation |

## Power Analysis

For detecting effect size d at power (1-β) with significance α:

```
n = 2 × ((z_α + z_β) / d)²

Where:
- z_α = 1.96 for α = 0.05
- z_β = 0.84 for power = 0.80
- d = Cohen's d effect size
```

### Sample Size Calculator

| Effect Size (d) | Power 0.80 | Power 0.90 | Power 0.95 |
|-----------------|------------|------------|------------|
| 0.2 (small) | 394 | 527 | 651 |
| 0.5 (medium) | 64 | 85 | 105 |
| 0.8 (large) | 26 | 34 | 42 |

## Benchmark Configuration

All Criterion benchmarks use:

```rust
group.sample_size(1000);  // n = 1000 samples
group.confidence_level(0.95);  // 95% CI
group.measurement_time(Duration::from_secs(5));  // 5s per benchmark
```

### Warmup Protocol

1. **Warmup iterations**: 100 (discarded)
2. **JIT compilation**: Complete before measurement
3. **Cache warming**: Ensure steady-state

## Reporting Requirements

Every benchmark must report:

1. **Sample size (n)**: Number of iterations
2. **Mean**: Central tendency
3. **Standard deviation**: Variability
4. **95% CI**: [lower, upper] bounds
5. **Effect size**: Cohen's d for comparisons

### Example Report Format

```json
{
  "benchmark": "full_render",
  "sample_size": 1000,
  "mean_ms": 0.82,
  "std_ms": 0.03,
  "ci_95": [0.79, 0.85],
  "effect_size": null
}
```

## PropTest Configuration

Property-based tests use:

```toml
[proptest]
cases = 1000  # Number of test cases
max_shrink_iters = 1000
fork = false  # Reproducibility
```

### Seed Management

```bash
export PROPTEST_SEED=0xdeadbeef
```

## Statistical Significance

Thresholds for accepting/rejecting hypotheses:

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| p-value | < 0.05 | Statistically significant |
| Cohen's d | > 0.5 | Practically significant |
| CI overlap | None | Clear difference |

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Criterion.rs Documentation: https://bheisler.github.io/criterion.rs/book/
- PropTest Documentation: https://proptest-rs.github.io/proptest/
