# Hansei Reports

Toyota Way Hansei (反省) principle: Reflection and continuous improvement through systematic post-training analysis.

## Usage

```rust
use entrenar::monitor::{HanseiAnalyzer, MetricsCollector, Metric};

let mut collector = MetricsCollector::new();

// During training
for epoch in 0..100 {
    collector.record(Metric::Loss, loss);
    collector.record(Metric::Accuracy, accuracy);
    collector.record(Metric::GradientNorm, grad_norm);
}

// Generate report
let analyzer = HanseiAnalyzer::new();
let report = analyzer.analyze("my-training", &collector, duration_secs);
println!("{}", analyzer.format_report(&report));
```

## Report Output

```
═══════════════════════════════════════════════════════════════
                    HANSEI POST-TRAINING REPORT
═══════════════════════════════════════════════════════════════

Training ID: my-training
Duration: 3600.00s
Total Steps: 10000

─── Metric Summaries ───────────────────────────────────────────

Loss:
  Mean: 0.123456  Std: 0.045678
  Min: 0.089012   Max: 0.567890
  Trend: ↑ Improving

Accuracy:
  Mean: 0.945678  Std: 0.023456
  Min: 0.800000   Max: 0.980000
  Trend: ↑ Improving

─── Issues Detected ────────────────────────────────────────────

[WARNING] Gradient Health
  Possible vanishing gradients: mean norm = 1.23e-08
  → Consider using residual connections or different activation functions

─── Recommendations ────────────────────────────────────────────
1. Training completed without critical issues.
2. Consider hyperparameter search for learning rate and batch size.

═══════════════════════════════════════════════════════════════
```

## Issue Detection

The analyzer automatically detects:

| Issue | Severity | Detection |
|-------|----------|-----------|
| NaN loss | Critical | `has_nan` flag |
| Inf loss | Critical | `has_inf` flag |
| Gradient explosion | Error | norm > 100 |
| Vanishing gradients | Warning | mean norm < 1e-7 |
| Loss increasing | Warning | trend analysis |
| Low accuracy | Warning | final < 50% |

## Trend Analysis

Trends are determined by comparing mean to midpoint:

- **Improving**: Mean closer to optimal end (low for loss, high for accuracy)
- **Degrading**: Mean closer to suboptimal end
- **Stable**: Small range relative to std
- **Oscillating**: High coefficient of variation (> 0.5)

## Custom Thresholds

```rust
let mut analyzer = HanseiAnalyzer::new();
analyzer.gradient_explosion_threshold = 50.0;  // Default: 100.0
analyzer.gradient_vanishing_threshold = 1e-8;  // Default: 1e-7
analyzer.min_accuracy_improvement = 0.02;      // Default: 0.01
```
