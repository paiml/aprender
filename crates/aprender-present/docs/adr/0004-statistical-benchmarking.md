# ADR-0004: Statistical Benchmarking Protocol

**Status:** Accepted
**Date:** 2026-01-12
**Decision Makers:** Engineering Team

## Context

Performance claims require statistical rigor to be falsifiable. Simple "it's fast" claims are meaningless without:
- Sample sizes
- Confidence intervals
- Effect sizes
- Comparison baselines

## Decision

We adopt **Criterion.rs** for statistically rigorous benchmarking with the following protocol:

### Sample Size Requirements

| Measurement Type | Minimum Samples | Rationale |
|-----------------|-----------------|-----------|
| Microbenchmark (<1ms) | 1000 | High variance requires more samples |
| Render benchmark | 100 | Moderate variance |
| Integration test | 30 | Lower variance, higher cost |

### Confidence Interval Protocol

All performance claims must include 95% confidence intervals:

```rust
/// CLAIM: Full render < 1ms (95% CI)
/// Hardware: AMD Threadripper 7960X, 128GB DDR5
/// Sample size: n=1000
/// Result: 0.82ms ± 0.03ms (95% CI: [0.79ms, 0.85ms])
#[bench]
fn bench_full_render(b: &mut Bencher) {
    b.iter(|| {
        let mut buffer = CellBuffer::new(180, 60);
        let app = App::new_deterministic();
        draw(&app, &mut buffer);
    });
}
```

### Effect Size Reporting

For comparisons (e.g., ptop vs ttop), report Cohen's d:

| Effect Size | d Value | Interpretation |
|-------------|---------|----------------|
| Small | 0.2 | Barely noticeable |
| Medium | 0.5 | Noticeable |
| Large | 0.8 | Obvious |

### Baseline Configuration

All benchmarks run on standardized configuration:

```toml
[benchmark.hardware]
cpu = "AMD Threadripper 7960X"
cores = 48
ram_gb = 128
storage = "NVMe RAID-0"

[benchmark.software]
os = "Ubuntu 24.04 LTS"
kernel = "6.8.0"
rust = "1.83.0"
```

## Consequences

### Positive
- Performance claims are falsifiable
- Regressions detectable with statistical significance
- Reproducible by third parties

### Negative
- Benchmarks take longer to run
- Requires consistent hardware for CI
- More complex reporting

## References

- `crates/presentar-terminal/benches/` - Benchmark implementations
- Criterion.rs documentation: https://bheisler.github.io/criterion.rs/
