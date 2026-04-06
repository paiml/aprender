# Andon Alerting (Jidoka)

Toyota Way Jidoka (自働化) principle: Stop-the-line on critical failures.

## Concept

In Toyota manufacturing, the Andon cord allows any worker to stop the production line when they detect a defect. In
training, this translates to automatic stopping on critical issues.

## Usage

```rust
use entrenar::monitor::{AndonSystem, AlertLevel, AndonConfig};

let config = AndonConfig {
    stop_on_nan: true,
    stop_on_inf: true,
    loss_spike_threshold: 10.0,
};

let mut andon = AndonSystem::with_config(config);

// During training
if loss.is_nan() {
    andon.alert(AlertLevel::Critical, "NaN loss detected");
}

if andon.should_stop() {
    println!("Training stopped by Andon system");
    break;
}
```

## Alert Levels

| Level | Description | Default Action |
|-------|-------------|----------------|
| `Info` | Informational | Log only |
| `Warning` | Potential issue | Log + notify |
| `Error` | Serious issue | Log + pause |
| `Critical` | Training failure | Stop immediately |

## Automatic Detection

The AndonSystem automatically detects:

- **NaN values** in loss or gradients
- **Infinity values** in loss or gradients
- **Loss spikes** (sudden large increases)
- **Gradient explosion** (norm > threshold)

## Integration with Training Loop

```rust
let mut andon = AndonSystem::new();

for epoch in 0..max_epochs {
    let loss = train_step(&model);

    // Check for issues
    andon.check_loss(loss);
    andon.check_gradients(&gradients);

    if andon.should_stop() {
        println!("Alerts: {:?}", andon.get_alerts());
        break;
    }
}
```
