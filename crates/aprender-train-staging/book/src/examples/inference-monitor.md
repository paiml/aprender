# Inference Monitoring

This example demonstrates real-time inference monitoring with latency tracking and throughput metrics.

## Running the Example

```bash
cargo run --example inference_monitor
```

## Code

```rust
{{#include ../../../examples/inference_monitor.rs}}
```

## Monitored Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Latency p50 | Median response time | - |
| Latency p95 | 95th percentile | 2x baseline |
| Latency p99 | 99th percentile | 3x baseline |
| Throughput | Requests/second | -20% baseline |
| Error rate | Failed requests | >1% |

## Usage

```rust
let mut monitor = InferenceMonitor::new()
    .with_window_size(1000)
    .with_percentiles([50.0, 95.0, 99.0]);

// Record inference
let start = Instant::now();
let result = model.forward(&input);
monitor.record(start.elapsed());

// Get statistics
let stats = monitor.stats();
println!("p95 latency: {:.2}ms", stats.p95_ms);
```

## Dashboard Integration

The monitor can export metrics for visualization:

```rust
monitor.export_prometheus("/metrics");
monitor.export_json("inference_stats.json");
```
