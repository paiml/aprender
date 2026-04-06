# Lambda Metrics

Production-grade observability with Prometheus-compatible metrics export.

## Metrics Structure

```rust
pub struct LambdaMetrics {
    /// Total requests processed
    pub requests_total: u64,
    /// Successful requests
    pub requests_success: u64,
    /// Failed requests
    pub requests_failed: u64,
    /// Total inference latency (for computing mean)
    pub latency_total_ms: f64,
    /// Cold start count
    pub cold_starts: u64,
    /// Batch requests processed
    pub batch_requests: u64,
    /// Total instances in batch requests
    pub batch_instances: u64,
}
```

## Recording Metrics

### Single Request

```rust
let mut metrics = LambdaMetrics::new();

let response = handler.handle(&request)?;
metrics.record_success(response.latency_ms, response.cold_start);
```

### Batch Request

```rust
let response = handler.handle_batch(&batch)?;
metrics.record_batch(
    response.success_count,
    response.error_count,
    response.total_latency_ms,
);
```

### Failures

```rust
if let Err(_) = handler.handle(&request) {
    metrics.record_failure();
}
```

## Prometheus Export

Export metrics in Prometheus text format:

```rust
let prometheus_output = metrics.to_prometheus();
```

Output format:

```prometheus
# HELP realizar_lambda_requests_total Total Lambda requests
# TYPE realizar_lambda_requests_total counter
realizar_lambda_requests_total 1000

# HELP realizar_lambda_requests_success Successful requests
# TYPE realizar_lambda_requests_success counter
realizar_lambda_requests_success 995

# HELP realizar_lambda_requests_failed Failed requests
# TYPE realizar_lambda_requests_failed counter
realizar_lambda_requests_failed 5

# HELP realizar_lambda_latency_avg_ms Average inference latency
# TYPE realizar_lambda_latency_avg_ms gauge
realizar_lambda_latency_avg_ms 0.045

# HELP realizar_lambda_cold_starts Cold start count
# TYPE realizar_lambda_cold_starts counter
realizar_lambda_cold_starts 1

# HELP realizar_lambda_batch_requests Batch requests processed
# TYPE realizar_lambda_batch_requests counter
realizar_lambda_batch_requests 50

# HELP realizar_lambda_batch_instances Total batch instances
# TYPE realizar_lambda_batch_instances counter
realizar_lambda_batch_instances 500
```

## Computed Metrics

### Average Latency

```rust
impl LambdaMetrics {
    pub fn avg_latency_ms(&self) -> f64 {
        if self.requests_success == 0 {
            0.0
        } else {
            self.latency_total_ms / self.requests_success as f64
        }
    }
}
```

### Success Rate

```rust
let success_rate = metrics.requests_success as f64 / metrics.requests_total as f64;
```

## CloudWatch Integration

Export metrics to CloudWatch via Lambda extension:

```rust
// In Lambda handler
let metrics_output = metrics.to_prometheus();

// Send to CloudWatch via EMF (Embedded Metric Format)
println!("{}", serde_json::json!({
    "_aws": {
        "Timestamp": chrono::Utc::now().timestamp_millis(),
        "CloudWatchMetrics": [{
            "Namespace": "Realizar/Lambda",
            "Dimensions": [["FunctionName"]],
            "Metrics": [
                {"Name": "RequestsTotal", "Unit": "Count"},
                {"Name": "LatencyAvgMs", "Unit": "Milliseconds"},
            ]
        }]
    },
    "FunctionName": std::env::var("AWS_LAMBDA_FUNCTION_NAME").unwrap_or_default(),
    "RequestsTotal": metrics.requests_total,
    "LatencyAvgMs": metrics.avg_latency_ms(),
}));
```
