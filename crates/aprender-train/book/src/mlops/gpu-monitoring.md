# GPU Monitoring

Real-time GPU metrics collection with Andon alerting system for proactive issue detection.

## Toyota Principle: Andon (行灯)

Visual signaling system that alerts operators to problems. GPU monitoring provides real-time visibility into hardware
health with automatic alerts.

## Quick Start

```rust
use entrenar::monitor::gpu::{GpuMonitor, GpuMetrics, AndonSystem};

// Create monitor
let monitor = GpuMonitor::new()?;

// Collect metrics
let metrics = monitor.collect_metrics()?;

for gpu in &metrics {
    println!("GPU {}: {}°C, {}% util, {:.1} GB / {:.1} GB",
        gpu.device_id,
        gpu.temperature_celsius,
        gpu.utilization_percent,
        gpu.memory_used_bytes as f64 / 1e9,
        gpu.memory_total_bytes as f64 / 1e9
    );
}

// Check for alerts
let andon = AndonSystem::default();
let alerts = andon.check(&metrics);

for alert in alerts {
    eprintln!("ALERT [severity {}]: {}", alert.severity(), alert.message());
}
```

## GPU Metrics

```rust
use entrenar::monitor::gpu::GpuMetrics;

// Available metrics
pub struct GpuMetrics {
    pub device_id: u32,
    pub name: String,
    pub temperature_celsius: u32,
    pub utilization_percent: u32,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub power_watts: f32,
    pub power_limit_watts: f32,
}

// Derived metrics
let memory_percent = metrics.memory_percent();
let power_percent = metrics.power_percent();
```

## Andon Alert System

```rust
use entrenar::monitor::gpu::{AndonSystem, GpuAlert, AlertConfig};

// Default thresholds
let andon = AndonSystem::default();

// Custom thresholds
let config = AlertConfig {
    thermal_threshold: 80,      // °C
    memory_threshold: 90,       // %
    power_threshold: 95,        // %
    idle_timeout_secs: 300,     // 5 minutes
};
let andon = AndonSystem::with_config(config);

// Check for alerts
let alerts = andon.check(&metrics);
```

## Alert Types

```rust
use entrenar::monitor::gpu::GpuAlert;

// Thermal throttling
GpuAlert::ThermalThrottling {
    device: 0,
    temp: 85,
    threshold: 80,
}

// Memory pressure
GpuAlert::MemoryPressure {
    device: 0,
    used_percent: 95,
    threshold: 90,
}

// Power limit
GpuAlert::PowerLimit {
    device: 0,
    power_percent: 98,
    threshold: 95,
}

// GPU idle (possible hang)
GpuAlert::GpuIdle {
    device: 0,
    duration_secs: 600,
}
```

## Continuous Monitoring

```rust
use std::time::Duration;
use std::thread;

let monitor = GpuMonitor::new()?;
let andon = AndonSystem::default();

loop {
    let metrics = monitor.collect_metrics()?;

    // Log metrics
    for gpu in &metrics {
        log_metrics(gpu);
    }

    // Check alerts
    let alerts = andon.check(&metrics);
    for alert in alerts {
        send_alert(&alert);
    }

    thread::sleep(Duration::from_secs(5));
}
```

## Sparkline Visualization

```rust
use entrenar::monitor::gpu::sparkline;

// Create ASCII sparkline from values
let utilization_history = vec![45, 67, 82, 91, 88, 75, 60];
let spark = sparkline(&utilization_history);
println!("Utilization: {}", spark);  // ▃▅▆█▇▅▄
```

## Integration with Training

```rust
use entrenar::train::{Trainer, TrainerConfig};
use entrenar::train::callback::GpuMonitorCallback;

let config = TrainerConfig::default();
let mut trainer = Trainer::new(config);

// Add GPU monitoring callback
trainer.add_callback(GpuMonitorCallback::new()
    .with_interval_secs(10)
    .with_thermal_threshold(80)
    .on_alert(|alert| {
        eprintln!("GPU Alert: {}", alert.message());
        // Optionally pause training
    }));

trainer.fit(&model, &dataset)?;
```

## Prometheus Export

Export metrics for Prometheus scraping:

```rust
use entrenar::monitor::prometheus::PrometheusExporter;

let exporter = PrometheusExporter::new()
    .with_prefix("entrenar")
    .with_port(9090);

// Register GPU metrics
exporter.register_gpu_metrics(&monitor)?;

// Start HTTP server
exporter.start()?;
```

Metrics exposed:
- `entrenar_gpu_temperature_celsius{device="0"}`
- `entrenar_gpu_utilization_percent{device="0"}`
- `entrenar_gpu_memory_used_bytes{device="0"}`
- `entrenar_gpu_power_watts{device="0"}`

## Cargo Run Example

```bash
# Monitor GPUs
cargo run --example gpu_monitor

# With custom interval
cargo run --example gpu_monitor -- --interval 5

# Export to Prometheus
cargo run --example gpu_monitor -- --prometheus --port 9090
```

## Mock Backend for Testing

```rust
use entrenar::monitor::gpu::MockGpuBackend;

// Create mock metrics for testing
let mock = MockGpuBackend::new()
    .with_device(0, "Mock GPU 0")
    .with_temperature(75)
    .with_utilization(85)
    .with_memory(8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024);

let metrics = mock.collect_metrics()?;
```

## Best Practices

1. **Set appropriate thresholds** - Balance sensitivity vs noise
2. **Monitor continuously** - 5-10 second intervals recommended
3. **Log metrics for analysis** - Useful for post-mortem debugging
4. **Integrate with alerting** - PagerDuty, Slack, etc.
5. **Use mock backend for tests** - Don't require real GPUs

## NVML Integration

GPU metrics are collected via NVIDIA Management Library (NVML):

```bash
# Verify NVML is available
nvidia-smi

# Required for GPU monitoring
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## See Also

- [MLOps Overview](./overview.md)
- [Andon Alerting](../monitor/andon.md)
- [Prometheus Export](../monitor/export.md)
