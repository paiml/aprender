# Case Study: Model Serving

This case study demonstrates serving ML models with APR's built-in HTTP server. The server supports multiple model formats with automatic format detection, Prometheus metrics, and graceful shutdown.

## Overview

APR serve provides:

- **Multi-format support** - APR, GGUF, and SafeTensors
- **Automatic format detection** - Detect model type from magic bytes
- **REST API** - Standard endpoints for inference
- **Prometheus metrics** - Production-ready observability
- **Memory-mapped loading** - Efficient handling of large models
- **Graceful shutdown** - Clean termination on Ctrl+C

## Running the Server

```bash
# Serve an APR model
apr serve model.apr

# Custom port and host
apr serve model.apr --port 3000 --host 0.0.0.0

# Disable GPU acceleration
apr serve model.apr --no-gpu

# Disable metrics endpoint
apr serve model.apr --no-metrics
```

## Server Configuration

```rust
use apr_cli::commands::serve::{ServerConfig, ServerState};

let config = ServerConfig {
    port: 8080,
    host: "127.0.0.1".to_string(),
    cors: true,
    timeout_secs: 30,
    max_concurrent: 10,
    metrics: true,
    no_gpu: false,
};

// Builder pattern
let config = ServerConfig::default()
    .with_port(3000)
    .with_host("0.0.0.0");

println!("Binding to: {}", config.bind_addr());
// Output: Binding to: 0.0.0.0:3000
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `port` | 8080 | HTTP port |
| `host` | 127.0.0.1 | Bind address |
| `cors` | true | Enable CORS headers |
| `timeout_secs` | 30 | Request timeout |
| `max_concurrent` | 10 | Max concurrent requests |
| `metrics` | true | Enable /metrics endpoint |
| `no_gpu` | false | Disable GPU acceleration |

## API Endpoints

### APR Models

```
POST /predict        - Single prediction
POST /predict/batch  - Batch prediction
GET  /health         - Health check
GET  /ready          - Readiness check
GET  /models         - List loaded models
GET  /metrics        - Prometheus metrics
```

### GGUF Models

```
GET /health          - Health check
GET /model           - Model information (tensors, metadata)
```

### SafeTensors Models

```
GET /health          - Health check
GET /tensors         - List tensor names
```

## Health Checks

```rust
use apr_cli::commands::serve::{health_check, ServerState, HealthResponse};

let state = ServerState::new(model_path, config)?;
let health = health_check(&state, uptime_secs);

// HealthResponse {
//     status: "healthy",
//     model: "/path/to/model.apr",
//     uptime_secs: 3600,
// }
```

### Health Endpoint Response

```json
{
  "status": "healthy",
  "model": "/models/whisper-large.apr",
  "uptime_secs": 3600
}
```

## Prometheus Metrics

The `/metrics` endpoint exposes Prometheus-format metrics:

```rust
use apr_cli::commands::serve::ServerMetrics;
use std::sync::Arc;

let metrics = ServerMetrics::new();

// Record requests
metrics.record_request(true, 100, 150);   // success, tokens, duration_ms
metrics.record_request(false, 0, 50);     // error

// Get Prometheus output
let output = metrics.prometheus_output();
```

### Available Metrics

```
# HELP apr_requests_total Total number of requests
# TYPE apr_requests_total counter
apr_requests_total 1500

# HELP apr_requests_success Successful requests
# TYPE apr_requests_success counter
apr_requests_success 1450

# HELP apr_requests_error Failed requests
# TYPE apr_requests_error counter
apr_requests_error 50

# HELP apr_tokens_generated_total Total tokens generated
# TYPE apr_tokens_generated_total counter
apr_tokens_generated_total 150000

# HELP apr_inference_duration_seconds_total Total inference time
# TYPE apr_inference_duration_seconds_total counter
apr_inference_duration_seconds_total 450.250
```

## Memory-Mapped Loading

Large models (>50MB) are automatically memory-mapped:

```rust
use apr_cli::commands::serve::ServerState;

let state = ServerState::new(model_path, config)?;

if state.uses_mmap {
    println!("Using memory-mapped loading");
} else {
    println!("Loading full model into memory");
}
```

### Benefits

- **Reduced memory pressure** - OS manages memory
- **Faster startup** - No full file read required
- **Efficient for large models** - 70B parameter models become feasible

## Format Detection

Models are automatically identified by magic bytes:

```rust
use realizar::format::{detect_format, ModelFormat};

let data = std::fs::read(&model_path)?;
let format = detect_format(&data[..8])?;

match format {
    ModelFormat::Apr => println!("APR model"),
    ModelFormat::Gguf => println!("GGUF model"),
    ModelFormat::SafeTensors => println!("SafeTensors model"),
}
```

### Magic Bytes

| Format | Magic | Description |
|--------|-------|-------------|
| APR | `APR1` | APR native format |
| GGUF | `GGUF` | GGML Unified Format |
| SafeTensors | `{` | JSON header |

## Example: Prediction Request

```bash
# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1.0, 2.0, 3.0, 4.0]}'

# Batch prediction
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}'
```

### Response Format

```json
{
  "output": [0.95, 0.03, 0.02],
  "latency_ms": 45,
  "tokens": 100
}
```

## Graceful Shutdown

The server handles Ctrl+C gracefully:

```rust
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
}

// In server startup
axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())
    .await?;
```

### Shutdown Behavior

1. Stop accepting new connections
2. Complete in-flight requests
3. Clean up resources
4. Exit cleanly

## Thread-Safe Metrics

Metrics are safe for concurrent access:

```rust
use std::sync::Arc;
use std::thread;
use apr_cli::commands::serve::ServerMetrics;

let metrics = ServerMetrics::new();

// Spawn multiple threads
let handles: Vec<_> = (0..10)
    .map(|_| {
        let m = Arc::clone(&metrics);
        thread::spawn(move || {
            for _ in 0..100 {
                m.record_request(true, 1, 1);
            }
        })
    })
    .collect();

for handle in handles {
    handle.join().unwrap();
}

// Metrics are correctly accumulated
assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1000);
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
        assert!(config.cors);
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_metrics_accumulation() {
        let metrics = ServerMetrics::new();
        metrics.record_request(true, 10, 100);
        metrics.record_request(true, 20, 200);
        metrics.record_request(false, 0, 50);

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.requests_error.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 30);
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = ServerMetrics::new();
        metrics.record_request(true, 100, 1000);

        let output = metrics.prometheus_output();
        assert!(output.contains("apr_requests_total 1"));
        assert!(output.contains("# TYPE apr_requests_total counter"));
    }
}
```

## Integration with Federation

Model serving integrates with the Federation Gateway:

```rust
use apr_cli::federation::{
    GatewayBuilder, ModelCatalog, ModelCatalogTrait,
    ModelId, NodeId, RegionId, Capability,
};

// Register served models with federation
catalog.register(
    ModelId("whisper-large-v3".to_string()),
    NodeId("us-west-serve-01".to_string()),
    RegionId("us-west-2".to_string()),
    vec![Capability::Transcribe],
).await?;

// Health checks report to federation
health.report_success(
    &NodeId("us-west-serve-01".to_string()),
    Duration::from_millis(45),
);

// Gateway routes to this server
let response = gateway.infer(&request).await?;
```

## Best Practices

1. **Use memory mapping** for models >50MB
2. **Enable metrics** in production
3. **Set appropriate timeouts** for your workload
4. **Monitor with Prometheus** - Scrape `/metrics` regularly
5. **Use health checks** - `/health` for liveness, `/ready` for readiness
6. **Handle shutdown gracefully** - Don't kill in-flight requests

## Further Reading

- [Federation Gateway](./federation-gateway.md)
- [Federation Routing Policies](./federation-routing.md)
- APR Model Format (see `docs/specifications/APR-SPEC.md`)
