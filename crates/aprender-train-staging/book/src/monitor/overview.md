# Real-Time Training Monitoring

The monitor module provides comprehensive real-time visibility into training runs, implementing Toyota Way principles
for quality assurance.

## Features

| Component | Purpose | Toyota Way Principle |
|-----------|---------|---------------------|
| MetricsCollector | Collect training metrics | Genchi Genbutsu (現地現物) |
| Dashboard | ASCII terminal visualization | Visual Management |
| DriftDetector | Anomaly detection | Jidoka (自働化) |
| AndonSystem | Alert management | Andon (行灯) |
| ModelLineage | Version tracking | Kaizen (改善) |
| HanseiAnalyzer | Post-training reports | Hansei (反省) |

## Quick Start

```rust
use entrenar::monitor::{MetricsCollector, Metric, Dashboard, HanseiAnalyzer};

// Create collector
let mut collector = MetricsCollector::new();

// During training loop
for epoch in 0..100 {
    let loss = train_epoch(&model, &data);
    let accuracy = evaluate(&model, &val_data);

    collector.record(Metric::Loss, loss);
    collector.record(Metric::Accuracy, accuracy);
    collector.record(Metric::Epoch, epoch as f64);
}

// Generate post-training report
let analyzer = HanseiAnalyzer::new();
let report = analyzer.analyze("my-training-run", &collector, duration_secs);
println!("{}", analyzer.format_report(&report));
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Forward │→│ Backward │→│ Optimize │→│ Metrics │        │
│  └─────────┘  └─────────┘  └─────────┘  └────┬────┘        │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                  MetricsCollector                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Welford's Algorithm: O(1) per update                  │  │
│  │ - Running mean, variance, min, max                    │  │
│  │ - NaN/Inf detection                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │Dashboard│   │  Drift  │   │  Andon  │   │ Hansei  │
    │ (ASCII) │   │Detector │   │ System  │   │Analyzer │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

## Performance

The monitor module is designed for minimal overhead:

| Operation | Performance | Complexity |
|-----------|-------------|------------|
| Record metric | < 1μs | O(1) |
| Get summary | < 100μs | O(metrics) |
| Dashboard render | < 100ms | O(history) |
| Drift detection | < 50μs | O(1) |

## Next Steps

- [Metrics Collection](./metrics-collection.md) - Recording training metrics
- [Terminal Dashboard](./dashboard.md) - Live visualization
- [Drift Detection](./drift-detection.md) - Anomaly detection
- [Andon Alerting](./andon.md) - Stop-the-line on critical failures
- [Hansei Reports](./hansei.md) - Post-training analysis
