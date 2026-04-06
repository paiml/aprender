# Training Monitoring

This example shows real-time training monitoring with metrics collection, drift detection, and Andon alerts.

## Running the Example

```bash
cargo run --example monitoring
```

## Code

```rust
{{#include ../../../examples/monitoring.rs}}
```

## Key Features

### Metrics Collection

```rust
let mut collector = MetricsCollector::new();
collector.record("loss", loss_value);
collector.record("accuracy", accuracy);
```

### Drift Detection

```rust
let mut drift = DriftDetector::new(window_size);
if let DriftStatus::Drift(z_score) = drift.check(metric) {
    println!("Drift detected: z={:.2}", z_score);
}
```

### Andon Alerts (Toyota Way)

```rust
let mut andon = AndonSystem::new();
andon.on_alert(|alert| {
    match alert.severity {
        Severity::Warning => println!("⚠️ {}", alert.message),
        Severity::Critical => println!("🚨 {}", alert.message),
    }
});
```

## Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Loss spike | 2σ | 3σ |
| NaN detection | - | Immediate |
| Gradient explosion | 10x mean | 100x mean |
