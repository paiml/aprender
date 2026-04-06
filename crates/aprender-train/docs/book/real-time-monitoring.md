# Real-Time Terminal Monitoring

This chapter covers entrenar's real-time terminal monitoring system, which provides
live visualization of training progress using trueno-viz.

## Overview

The monitoring system provides:

- **MetricsBuffer**: O(1) ring buffer for streaming metrics
- **Sparklines**: Unicode visualization for inline metrics
- **Progress bars**: Kalman-filtered ETA estimation
- **Loss curves**: trueno-viz integration with ASCII/Unicode/ANSI rendering
- **Health monitoring**: Andon-style alerts for NaN/Inf/divergence detection
- **Explainability**: Real-time feature importance and gradient flow

## Quick Start

```rust
use entrenar::train::{
    Trainer, TrainConfig, CallbackManager,
    TerminalMonitorCallback, DashboardLayout, TerminalMode,
};

// Create the monitor callback
let monitor = TerminalMonitorCallback::builder()
    .layout(DashboardLayout::Full)
    .mode(TerminalMode::Unicode)
    .sparkline_width(30)
    .refresh_interval_ms(100)
    .build();

// Add to trainer
let mut callbacks = CallbackManager::new();
callbacks.add(monitor);

let mut trainer = Trainer::new(params, optimizer, config)
    .with_callbacks(callbacks);

trainer.train(&dataloader)?;
```

## Components

### MetricsBuffer

A fixed-size ring buffer optimized for streaming metrics visualization:

```rust
use entrenar::train::MetricsBuffer;

let mut buffer = MetricsBuffer::new(100); // Keep last 100 values

// Push values (O(1) operation)
buffer.push(0.5);
buffer.push(0.4);
buffer.push(0.3);

// Query statistics
println!("Last: {:?}", buffer.last());      // Some(0.3)
println!("Min: {:?}", buffer.min());        // Some(0.3)
println!("Max: {:?}", buffer.max());        // Some(0.5)
println!("Mean: {:?}", buffer.mean());      // Some(0.4)

// Get recent values for visualization
let recent = buffer.last_n(10);
```

### Sparklines

Inline Unicode visualization using block characters:

```rust
use entrenar::train::{sparkline, sparkline_range, SPARK_CHARS};

let losses = vec![1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28];

// Auto-scaled sparkline
let spark = sparkline(&losses, 8);
println!("Loss: {}", spark);  // "█▆▄▃▂▂▁▁"

// Fixed-range sparkline (for comparisons)
let spark = sparkline_range(&losses, 8, 0.0, 1.0);
```

### Progress Bar with Kalman ETA

Progress tracking with statistically smoothed time estimates:

```rust
use entrenar::train::{ProgressBar, KalmanEta};

let mut progress = ProgressBar::new(100, 40); // 100 steps, 40 char width

for step in 0..100 {
    progress.update(step, 0.1); // Update with step duration
    println!("\r{}", progress.render());
}
// Output: [████████████████████████████████████████] 100.0% │ ETA: 0s
```

### Loss Curve Display

Integrated trueno-viz loss curve rendering:

```rust
use entrenar::train::{LossCurveDisplay, TerminalMode};

let mut display = LossCurveDisplay::new(80, 20)
    .terminal_mode(TerminalMode::Unicode)
    .smoothing(0.6);

// Push metrics
for epoch in 0..50 {
    let train_loss = 1.0 / (epoch as f32 + 1.0);
    let val_loss = train_loss * 1.1;
    display.push_losses(train_loss, val_loss);
}

// Render to terminal
println!("{}", display.render_terminal());

// Get summary
for (name, min, last, best_epoch) in display.summary() {
    println!("{}: min={:?}, last={:?}, best_epoch={:?}",
             name, min, last, best_epoch);
}
```

### Andon System (Health Monitoring)

Toyota Production System-inspired anomaly detection:

```rust
use entrenar::train::{AndonSystem, AlertLevel};

let mut andon = AndonSystem::new()
    .with_divergence_threshold(100.0)
    .with_stall_patience(10)
    .with_stop_on_critical(true);

// Check each loss value
for loss in losses.iter() {
    match andon.check_loss(*loss) {
        CallbackAction::Continue => {},
        CallbackAction::Stop => {
            println!("Training stopped by Andon!");
            break;
        }
    }
}

// Review alerts
for alert in andon.alerts() {
    match alert.level {
        AlertLevel::Warning => println!("⚠ {}", alert.message),
        AlertLevel::Critical => println!("🛑 {}", alert.message),
    }
}
```

Alert types:
- **NaN detected**: Loss became NaN (critical)
- **Inf detected**: Loss became infinite (critical)
- **Divergence**: Loss exceeds threshold (warning)
- **Stall**: No improvement for N epochs (warning)

### Dashboard Layouts

Three built-in layouts for different terminal sizes:

#### Minimal (< 40 columns)
```
E15 │ loss=0.234 │ ▁▂▃▄▅▆▇█
```

#### Compact (40-80 columns)
```
Epoch 15/100 │ loss=0.234 │ lr=1e-4 │ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
[████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 15.0% │ ETA: 2m 34s
```

#### Full (80+ columns)
```
╭─ Training Progress ──────────────────────────────────────────────────────────╮
│ Epoch: 15/100 │ Step: 450/3000 │ LR: 1.00e-04 │ Best: 0.189 @ epoch 12     │
├─ Metrics ─────────────────────────────────────────────────────────────────────┤
│ Loss:     0.234 ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▁▂▂▁ (min=0.189, avg=0.312)              │
│ Val Loss: 0.256 ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▁▂▂▂ (min=0.201, avg=0.334)              │
├─ Progress ────────────────────────────────────────────────────────────────────┤
│ [████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 15.0% │ ETA: 2m 34s            │
╰───────────────────────────────────────────────────────────────────────────────╯
```

### Terminal Capability Detection

Automatic detection of terminal features:

```rust
use entrenar::train::TerminalCapabilities;

let caps = TerminalCapabilities::detect();

println!("Unicode support: {}", caps.unicode);
println!("ANSI colors: {}", caps.ansi_colors);
println!("True color: {}", caps.true_color);
println!("Width: {} columns", caps.width);

// Get recommended mode
let mode = caps.recommended_mode();
```

### YAML Configuration

Configure monitoring via YAML:

```yaml
monitor:
  enabled: true
  layout: Full
  mode: Unicode
  sparkline_width: 20
  refresh_ms: 100
  andon:
    divergence_threshold: 100.0
    stall_patience: 10
    stop_on_critical: true
```

Load configuration:

```rust
use entrenar::train::MonitorConfig;

let config: MonitorConfig = serde_yaml::from_str(yaml)?;
let callback = config.to_callback();
```

## Feature Importance Display

Real-time visualization of feature importance:

```rust
use entrenar::train::FeatureImportanceChart;

let mut chart = FeatureImportanceChart::new(5); // Top 5 features

// Update with importance scores
let scores = vec![0.8, 0.3, 0.5, 0.1, 0.9, 0.2];
let names = vec!["age", "income", "tenure", "region", "credit_score", "balance"];
chart.update(&scores, Some(&names));

println!("{}", chart.render());
// Output:
// credit_score ████████████████████ 0.90
// age          ████████████████     0.80
// tenure       ██████████           0.50
// income       ██████               0.30
// balance      ████                 0.20
```

## Gradient Flow Heatmap

Monitor gradient magnitudes across layers:

```rust
use entrenar::train::GradientFlowHeatmap;

let mut heatmap = GradientFlowHeatmap::new(10); // Keep 10 time steps

// Update with layer gradients
let gradients = vec![0.01, 0.005, 0.002, 0.001]; // Gradients per layer
let layer_names = vec!["embed", "attn", "ffn", "head"];
heatmap.update(&gradients, Some(&layer_names));

println!("{}", heatmap.render());
// Shows heatmap of gradient flow over time
```

## Reference Curves (Standardized Work)

Compare training progress against a "golden trace":

```rust
use entrenar::train::ReferenceCurve;

// Load reference from previous successful run
let reference = ReferenceCurve::from_json(r#"
{
    "name": "baseline_v1",
    "values": [1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1]
}
"#)?;

// Current training progress
let current = vec![1.0, 0.55, 0.35, 0.22];

// Check deviation
let max_dev = reference.max_deviation(&current);
println!("Max deviation from reference: {:.1}%", max_dev * 100.0);

// Visual comparison
let comparison = reference.comparison_sparkline(&current, 20);
println!("Deviation: {}", comparison);
// Negative = better than reference, Positive = worse
```

## Performance Considerations

- **MetricsBuffer**: O(1) push, O(n) iteration for visualization
- **Sparklines**: O(n) where n = input length
- **Refresh policy**: Adaptive rate to avoid terminal flooding
- **Memory**: Ring buffers prevent unbounded growth

## Academic References

The monitoring system implements algorithms from peer-reviewed research:

1. Tufte, E. R. (2006). *Beautiful Evidence*. Graphics Press. (Sparklines)
2. Welch, G., & Bishop, G. (1995). "An Introduction to the Kalman Filter." (ETA estimation)
3. Ohno, T. (1988). *Toyota Production System*. Productivity Press. (Andon system)
4. Cleveland, W. S. (1993). *Visualizing Data*. Hobart Press.
5. Card, S. K., Mackinlay, J. D., & Shneiderman, B. (1999). *Readings in Information Visualization*.

## Example: Complete Training Monitor

```rust
use entrenar::train::{
    Trainer, TrainConfig, CallbackManager,
    TerminalMonitorCallback, DashboardLayout, TerminalMode,
    AndonSystem, MonitorConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load config from YAML
    let config: MonitorConfig = serde_yaml::from_str(r#"
        enabled: true
        layout: Full
        mode: Unicode
        sparkline_width: 25
        refresh_ms: 100
    "#)?;

    // Create monitor with Andon health checks
    let monitor = config.to_callback();

    // Setup callbacks
    let mut callbacks = CallbackManager::new();
    callbacks.add(monitor);

    // Create trainer
    let trainer = Trainer::new(params, optimizer, train_config)
        .with_callbacks(callbacks);

    // Train with real-time monitoring
    let result = trainer.train(&dataloader)?;

    println!("\nTraining complete!");
    println!("Final loss: {:.4}", result.final_loss);
    println!("Best loss: {:.4} (epoch {})", result.best_loss, result.best_epoch);

    Ok(())
}
```
