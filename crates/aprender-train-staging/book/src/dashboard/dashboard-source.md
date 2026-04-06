# DashboardSource Trait

The `DashboardSource` trait provides a unified interface for accessing training run data from dashboards.

## Trait Definition

```rust
pub trait DashboardSource {
    /// Get the current run status.
    fn status(&self) -> RunStatus;

    /// Get recent metrics, limited to `limit` points per metric.
    fn recent_metrics(&self, limit: usize) -> HashMap<String, MetricSnapshot>;

    /// Subscribe to metric updates.
    fn subscribe(&self, callback: SubscriptionCallback);

    /// Get current resource usage.
    fn resource_usage(&self) -> ResourceSnapshot;
}
```

## MetricSnapshot

A snapshot of metric values for dashboard display:

```rust
pub struct MetricSnapshot {
    /// Metric key (e.g., "loss", "accuracy")
    pub key: String,
    /// Time-value pairs: (timestamp_ms, value)
    pub values: Vec<(u64, f64)>,
    /// Current trend direction
    pub trend: Trend,
}
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `latest()` | `Option<f64>` | Get the most recent value |
| `min()` | `Option<f64>` | Get the minimum value |
| `max()` | `Option<f64>` | Get the maximum value |
| `mean()` | `Option<f64>` | Get the average value |
| `len()` | `usize` | Number of data points |
| `is_empty()` | `bool` | Check if empty |

### Example

```rust
let metrics = run.recent_metrics(100);

if let Some(loss) = metrics.get("loss") {
    println!("Loss statistics:");
    println!("  Latest: {:.4}", loss.latest().unwrap_or(0.0));
    println!("  Min: {:.4}", loss.min().unwrap_or(0.0));
    println!("  Max: {:.4}", loss.max().unwrap_or(0.0));
    println!("  Mean: {:.4}", loss.mean().unwrap_or(0.0));
    println!("  Trend: {} {}", loss.trend, loss.trend.emoji());
    println!("  Points: {}", loss.len());
}
```

## Trend Enum

Represents the direction of a metric over time:

```rust
pub enum Trend {
    Rising,   // Metric is increasing
    Falling,  // Metric is decreasing
    Stable,   // Metric is relatively stable
}
```

### Trend Detection

Trends are computed using linear regression on the metric values:

```rust
impl Trend {
    pub fn from_values(values: &[f64]) -> Self {
        // Computes slope via linear regression
        // Normalizes by mean for relative change
        // Threshold: 5% relative change
    }
}
```

### Display

```rust
let trend = Trend::Rising;
println!("{}", trend);         // "rising"
println!("{}", trend.emoji()); // "↑"
```

| Trend | Display | Emoji |
|-------|---------|-------|
| Rising | "rising" | ↑ |
| Falling | "falling" | ↓ |
| Stable | "stable" | → |

## ResourceSnapshot

System resource utilization snapshot:

```rust
pub struct ResourceSnapshot {
    pub gpu_util: f64,           // GPU utilization (0.0-1.0)
    pub cpu_util: f64,           // CPU utilization (0.0-1.0)
    pub memory_used: u64,        // Memory used (bytes)
    pub memory_total: u64,       // Total memory (bytes)
    pub gpu_memory_used: Option<u64>,   // GPU memory used
    pub gpu_memory_total: Option<u64>,  // Total GPU memory
}
```

### Builder Pattern

```rust
let resources = ResourceSnapshot::new()
    .with_gpu_util(0.75)
    .with_cpu_util(0.50)
    .with_memory(4_000_000_000, 8_000_000_000)
    .with_gpu_memory(6_000_000_000, 16_000_000_000);

println!("Memory utilization: {:.1}%", resources.memory_util() * 100.0);
println!("GPU memory: {:.1}%", resources.gpu_memory_util().unwrap() * 100.0);
```

## Implementation for Run

The `Run` struct implements `DashboardSource`:

```rust
impl<S: ExperimentStorage> DashboardSource for Run<S> {
    fn status(&self) -> RunStatus {
        if self.is_finished() {
            // Query storage for actual status
        } else {
            RunStatus::Running
        }
    }

    fn recent_metrics(&self, limit: usize) -> HashMap<String, MetricSnapshot> {
        // Fetches metrics from storage
        // Limits to most recent `limit` points
        // Computes trends automatically
    }

    fn subscribe(&self, callback: SubscriptionCallback) {
        // Placeholder for real-time subscriptions
    }

    fn resource_usage(&self) -> ResourceSnapshot {
        // Returns current system metrics
    }
}
```

## Custom Implementations

You can implement `DashboardSource` for custom types:

```rust
struct MyTrainingMonitor {
    metrics: HashMap<String, Vec<f64>>,
}

impl DashboardSource for MyTrainingMonitor {
    fn status(&self) -> RunStatus {
        RunStatus::Running
    }

    fn recent_metrics(&self, limit: usize) -> HashMap<String, MetricSnapshot> {
        self.metrics.iter()
            .map(|(key, values)| {
                let recent: Vec<(u64, f64)> = values.iter()
                    .rev()
                    .take(limit)
                    .enumerate()
                    .map(|(i, &v)| (i as u64, v))
                    .collect();
                (key.clone(), MetricSnapshot::new(key, recent))
            })
            .collect()
    }

    fn subscribe(&self, _callback: SubscriptionCallback) {
        // Implement subscription logic
    }

    fn resource_usage(&self) -> ResourceSnapshot {
        ResourceSnapshot::new()
    }
}
```

## See Also

- [Dashboard Overview](./overview.md)
- [WASM Bindings](./wasm.md)
