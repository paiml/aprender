# Drift Detection and Retraining

Aprender v0.10 introduces drift detection to monitor model performance and trigger retraining when data distribution shifts.

## Why Drift Detection?

ML models degrade over time due to:
- **Data drift**: Input feature distributions change
- **Concept drift**: Relationship between features and target changes
- **Model staleness**: World changes, model doesn't

```
Training Data          Production Data
┌─────────────┐        ┌─────────────┐
│  μ = 50     │   →    │  μ = 65     │  ← Data Drift!
│  σ = 10     │        │  σ = 15     │
└─────────────┘        └─────────────┘
```

## DriftDetector

### Basic Usage

```rust
use aprender::metrics::drift::{DriftDetector, DriftConfig, DriftStatus};
use aprender::primitives::Vector;

// Reference data (from training)
let reference = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

// Create detector with default thresholds
let detector = DriftDetector::new(DriftConfig::default());

// Check current production data
let current = Vector::from_slice(&[1.1, 2.1, 3.0, 4.0, 5.1]);
let status = detector.detect_univariate(&reference, &current);

match status {
    DriftStatus::NoDrift => println!("No drift detected"),
    DriftStatus::Warning { score } => println!("Warning: drift score {:.4}", score),
    DriftStatus::Drift { score } => println!("DRIFT DETECTED: score {:.4}", score),
}
```

### DriftStatus

```rust
pub enum DriftStatus {
    /// No significant drift detected
    NoDrift,

    /// Warning level drift (monitor closely)
    Warning { score: f32 },

    /// Significant drift detected (retrain recommended)
    Drift { score: f32 },
}

impl DriftStatus {
    /// Check if retraining is needed
    pub fn needs_retraining(&self) -> bool;

    /// Get drift score if available
    pub fn score(&self) -> Option<f32>;
}
```

## Configuration

### DriftConfig

```rust
pub struct DriftConfig {
    /// Threshold for warning level (default: 0.1)
    pub warning_threshold: f32,

    /// Threshold for drift level (default: 0.2)
    pub drift_threshold: f32,

    /// Minimum samples required (default: 30)
    pub min_samples: usize,

    /// Window size for rolling stats (default: 100)
    pub window_size: usize,
}
```

### Custom Thresholds

```rust
// Strict thresholds for critical models
let strict_config = DriftConfig::new(0.05, 0.10)
    .with_min_samples(100)
    .with_window_size(500);

// Relaxed thresholds for stable domains
let relaxed_config = DriftConfig::new(0.15, 0.30)
    .with_min_samples(20);
```

## RollingDriftMonitor

Monitor drift continuously in production:

```rust
use aprender::metrics::drift::{RollingDriftMonitor, DriftConfig};

// Initialize with reference statistics
let mut monitor = RollingDriftMonitor::new(
    reference_mean,
    reference_std,
    DriftConfig::default(),
);

// In production loop
for batch in production_data {
    monitor.update(&batch);

    if let Some(status) = monitor.check() {
        if status.needs_retraining() {
            trigger_retraining_pipeline();
        }
    }
}
```

## RetrainingTrigger

Automated retraining decisions:

```rust
use aprender::metrics::drift::{RetrainingTrigger, TriggerConfig};

let trigger = RetrainingTrigger::new(TriggerConfig {
    drift_threshold: 0.2,
    performance_threshold: 0.05,  // 5% accuracy drop
    min_samples_since_train: 1000,
    max_time_since_train_hours: 24,
});

// Check if retraining needed
let decision = trigger.evaluate(
    &drift_status,
    current_accuracy,
    baseline_accuracy,
    samples_processed,
    hours_since_training,
);

if decision.should_retrain {
    println!("Retraining triggered: {}", decision.reason);
}
```

## Integration with Lambda

```rust
use realizar::lambda::{LambdaHandler, LambdaMetrics};
use aprender::metrics::drift::{DriftDetector, DriftConfig, DriftStatus};

pub struct MonitoredHandler {
    handler: LambdaHandler,
    drift_detector: DriftDetector,
    reference_stats: ReferenceStats,
    metrics: LambdaMetrics,
}

impl MonitoredHandler {
    pub fn handle_with_monitoring(
        &mut self,
        request: &LambdaRequest,
    ) -> Result<LambdaResponse, LambdaError> {
        // Check for drift
        let status = self.drift_detector.detect_univariate(
            &self.reference_stats.features,
            &Vector::from_slice(&request.features),
        );

        // Record drift metrics
        if let Some(score) = status.score() {
            self.metrics.record_drift_score(score);
        }

        if status.needs_retraining() {
            self.metrics.record_retraining_trigger();
            // Alert or trigger retraining pipeline
        }

        // Normal inference
        self.handler.handle(request)
    }
}
```

## Drift Detection Methods

### Univariate (Single Feature)

```rust
// Kolmogorov-Smirnov inspired statistic
let status = detector.detect_univariate(&reference, &current);
```

### Multivariate (All Features)

```rust
// Check drift across all features
let status = detector.detect_multivariate(&reference_matrix, &current_matrix);
```

### Population Stability Index (PSI)

```rust
// PSI for distribution comparison
let psi = detector.compute_psi(&reference, &current, n_bins);

// PSI interpretation:
// < 0.1: No significant change
// 0.1 - 0.2: Moderate change
// > 0.2: Significant change
```

## Best Practices

### 1. Establish Baselines

```rust
// Save reference statistics during training
let reference_stats = ReferenceStats {
    mean: training_features.mean(),
    std: training_features.std(),
    percentiles: compute_percentiles(&training_features),
};
```

### 2. Monitor Multiple Signals

| Signal | What it Detects |
|--------|-----------------|
| Feature drift | Input distribution changes |
| Prediction drift | Output distribution changes |
| Performance drift | Accuracy/F1 degradation |
| Latency drift | Inference time changes |

### 3. Alert Thresholds

```rust
// Tiered alerting
match status {
    DriftStatus::NoDrift => { /* Log only */ },
    DriftStatus::Warning { .. } => {
        send_slack_alert("Drift warning - monitor closely");
    },
    DriftStatus::Drift { .. } => {
        send_pagerduty_alert("Drift detected - retrain needed");
        pause_predictions_if_critical();
    },
}
```

### 4. A/B Testing for Retraining

```rust
// Don't immediately replace model
// Use shadow deployment first
let old_pred = old_model.predict(&features);
let new_pred = new_model.predict(&features);

// Compare performance before switching
if new_model_better(&old_pred, &new_pred, &actuals) {
    promote_new_model();
}
```
