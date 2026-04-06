# Terminal Dashboard

ASCII terminal dashboard for real-time training visualization.

## Usage

```rust
use entrenar::monitor::{Dashboard, MetricsCollector, Metric};

let mut collector = MetricsCollector::new();
let mut dashboard = Dashboard::new();

// During training
for epoch in 0..100 {
    collector.record(Metric::Loss, loss);
    collector.record(Metric::Accuracy, accuracy);

    // Update and render
    dashboard.update(collector.summary());
    println!("{}", dashboard.render_ascii());
}
```

## Output Example

```
═══════════════════════════════════════════════════════════════
                    TRAINING DASHBOARD
═══════════════════════════════════════════════════════════════

Loss:     0.1234  [▁▂▃▄▅▆▇█▇▆▅▄▃▂▁]  ↓ Improving
Accuracy: 0.9567  [▁▁▂▃▄▅▆▇▇▇▇▇▇▇█]  ↑ Improving

Statistics:
  Loss     - mean: 0.3421, std: 0.1234, min: 0.1234, max: 0.8901
  Accuracy - mean: 0.8234, std: 0.0567, min: 0.5000, max: 0.9567

═══════════════════════════════════════════════════════════════
```

## Sparklines

The dashboard uses Unicode sparkline characters to show metric history:

```rust
let sparkline = dashboard.sparkline(&Metric::Loss);
// Returns: "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
```

Characters map values to 8 levels: `▁▂▃▄▅▆▇█`

## Configuration

```rust
use entrenar::monitor::DashboardConfig;

let config = DashboardConfig {
    width: 80,
    height: 24,
    refresh_ms: 1000,
};

let dashboard = Dashboard::with_config(config);
```
