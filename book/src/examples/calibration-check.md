# P-Value Calibration Check

This example verifies the statistical validity of drift detection tests by checking that p-values are uniformly
distributed under the null hypothesis (no drift).

## Running the Example

```bash
cargo run --example calibration_check
```

## Code

```rust
{{#include ../../../examples/calibration_check.rs}}
```

## Expected Output

```
=== P-Value Calibration Check ===

Testing drift detection statistical calibration under null hypothesis.

Under H0 (no drift), p-values should be uniformly distributed on [0, 1].
This means ~5% of tests should reject at alpha=0.05.

--- Kolmogorov-Smirnov Test Calibration ---
KS Test: 5.5% rejection rate (expected: ~5.0%)
Calibration: PASS

--- Population Stability Index Calibration ---
PSI Test: 2.1% rejection rate at threshold=0.1
Calibration: PASS (rejection rate under 15%)

--- Statistical Power Test (Ability to Detect Drift) ---
Power at effect size d=1.0: 100.0%
Power: ADEQUATE (>80%% required)

=== Summary ===
All calibration checks PASSED
```

## Why Calibration Matters

Statistical tests must be properly calibrated to be useful:

1. **Type I Error Rate**: Under H0 (no drift), the test should reject at exactly the specified alpha level (e.g., 5%)
2. **Statistical Power**: Under H1 (real drift), the test should detect drift with high probability

### Null Hypothesis Testing

When there's no actual drift (samples from same distribution):
- P-values should be uniformly distributed on [0, 1]
- Rejection rate should equal alpha (e.g., 5% at alpha=0.05)
- Higher rejection rates indicate false positives

### Power Analysis

When there's real drift (shifted distribution):
- The test should detect it most of the time
- Power > 80% is considered adequate
- Higher power means fewer false negatives

## Key Concepts

### Calibration Check Implementation

```rust
fn run_calibration_check(test: DriftTest) -> f64 {
    let mut rejections = 0;

    for trial in 0..NUM_TRIALS {
        // Generate two independent samples from SAME distribution
        let baseline = generate_normal_data(SAMPLE_SIZE, 50.0, 10.0, seed1);
        let current = generate_normal_data(SAMPLE_SIZE, 50.0, 10.0, seed2);

        // Run drift detection
        let mut detector = DriftDetector::new(vec![test.clone()]);
        detector.set_baseline(&baseline);
        let results = detector.check(&current);

        // Count false positives
        if results.iter().any(|r| r.drifted) {
            rejections += 1;
        }
    }

    rejections as f64 / NUM_TRIALS as f64
}
```

### Interpreting Calibration Results

| Metric | Expected | Acceptable Range |
|--------|----------|------------------|
| KS rejection @ alpha=0.05 | 5% | 3-7% |
| PSI false positive rate | <10% | <15% |
| Power @ d=1.0 | >80% | >80% |
