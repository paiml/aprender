# A/B Testing Statistics

Per Box et al. (2005), latency distributions are rarely normal - they're typically log-normal with heavy right tails.

## The Problem with T-Tests

Standard Welch's t-test assumes normality. For latency data:
- Distribution is right-skewed (outliers at high end)
- Mean is pulled up by tail
- T-test may be underpowered or biased

## Solution: Log-Transform

```rust
use realizar::stats::{analyze, AnalysisConfig, TestMethod};

let control = vec![10.0, 12.0, 15.0, 100.0, 11.0];   // Has outlier
let treatment = vec![8.0, 9.0, 10.0, 50.0, 8.5];

let config = AnalysisConfig {
    alpha: 0.05,
    auto_detect_skew: true,  // Automatically detect and handle skew
};

let result = analyze(&control, &treatment, &config);

// For skewed data, uses log-transform automatically
assert_eq!(result.method, TestMethod::LogTransformTTest);
```

## Skewness Detection

The module auto-detects skewed distributions:

```rust
// Skewness > 1.0 triggers log-transform
let skewed_data = vec![1.0, 1.0, 1.0, 1.0, 100.0];
// Skewness ≈ 2.2 → uses LogTransformTTest

let normal_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
// Skewness ≈ 0 → uses standard TTest
```

## Geometric Mean

Log-transform reports geometric mean (more robust to outliers):

```rust
// result.control_mean is geometric mean, not arithmetic
// Better represents "typical" latency
```

## Test Methods

| Method | When Used | Advantage |
|--------|-----------|-----------|
| `TTest` | Normal data | Standard, well-understood |
| `LogTransformTTest` | Skewed data (skewness > 1.0) | Robust to outliers |
| `MannWhitneyU` | Non-parametric fallback | No distribution assumptions |
