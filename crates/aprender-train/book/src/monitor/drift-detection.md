# Drift Detection

Sliding window anomaly detection using z-score analysis.

## Usage

```rust
use entrenar::monitor::{DriftDetector, DriftStatus};

let mut detector = DriftDetector::new(100); // 100-value window

// During training
for value in metrics {
    match detector.check(value) {
        DriftStatus::Normal => {},
        DriftStatus::Warning(z) => println!("Warning: z-score = {:.2}", z),
        DriftStatus::Drift(z) => {
            println!("DRIFT DETECTED: z-score = {:.2}", z);
            // Take corrective action
        }
    }
}
```

## Severity Levels

| Z-Score | Severity | Action |
|---------|----------|--------|
| < 3.0 | Normal | Continue |
| 3.0 - 4.0 | Warning | Log and monitor |
| 4.0 - 5.0 | High | Alert |
| > 5.0 | Critical | Stop training |

## Sliding Window Baseline

The detector maintains a sliding window for adaptive baselines:

```rust
use entrenar::monitor::SlidingWindowBaseline;

let mut baseline = SlidingWindowBaseline::new(100);

// Add values
baseline.update(0.5);
baseline.update(0.48);

// Check if value is anomalous
if let Some(anomaly) = baseline.detect_anomaly(0.9, 3.0) {
    println!("Anomaly: {:?}", anomaly.severity);
}
```

## Z-Score Calculation

```
z = (x - μ) / σ

where:
  x = current value
  μ = window mean
  σ = window standard deviation
```
