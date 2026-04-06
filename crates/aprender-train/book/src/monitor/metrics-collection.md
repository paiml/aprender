# Metrics Collection

The `MetricsCollector` uses Welford's algorithm for numerically stable running statistics.

## Basic Usage

```rust
use entrenar::monitor::{MetricsCollector, Metric};

let mut collector = MetricsCollector::new();

// Record individual metrics
collector.record(Metric::Loss, 0.5);
collector.record(Metric::Accuracy, 0.85);

// Record batch of metrics
collector.record_batch(&[
    (Metric::Loss, 0.45),
    (Metric::Accuracy, 0.87),
    (Metric::GradientNorm, 1.2),
]);
```

## Available Metrics

| Metric | Purpose |
|--------|---------|
| `Metric::Loss` | Training loss |
| `Metric::Accuracy` | Model accuracy |
| `Metric::LearningRate` | Current LR |
| `Metric::GradientNorm` | Gradient L2 norm |
| `Metric::Epoch` | Current epoch |
| `Metric::Batch` | Current batch |
| `Metric::Custom(String)` | User-defined |

## Getting Statistics

```rust
let summary = collector.summary();

if let Some(loss_stats) = summary.get(&Metric::Loss) {
    println!("Loss - mean: {:.4}, std: {:.4}", loss_stats.mean, loss_stats.std);
    println!("       min: {:.4}, max: {:.4}", loss_stats.min, loss_stats.max);

    if loss_stats.has_nan {
        println!("WARNING: NaN values detected!");
    }
}
```

## Welford's Algorithm

The collector uses Welford's online algorithm for O(1) updates:

```
mean_new = mean_old + (x - mean_old) / n
M2_new = M2_old + (x - mean_old) * (x - mean_new)
variance = M2 / (n - 1)
```

This provides:
- Numerical stability for large datasets
- Constant memory usage
- O(1) per update
