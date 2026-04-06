# HTTP API for Aprender

When the `aprender-serve` feature is enabled, Realizar exposes HTTP endpoints for aprender model inference.

## Enable the Feature

```toml
[dependencies]
realizar = { version = "0.2", features = ["aprender-serve"] }
```

## Endpoints

### POST /predict

Single prediction endpoint.

**Request:**
```json
{
    "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response:**
```json
{
    "prediction": 0,
    "probabilities": [0.97, 0.02, 0.01],
    "latency_ms": 0.045
}
```

### POST /predict/batch

Batch prediction for multiple instances.

**Request:**
```json
{
    "instances": [
        {"features": [5.1, 3.5, 1.4, 0.2]},
        {"features": [6.2, 2.9, 4.3, 1.3]},
        {"features": [7.1, 3.0, 5.9, 2.1]}
    ]
}
```

**Response:**
```json
{
    "predictions": [
        {"prediction": 0, "probabilities": [0.97, 0.02, 0.01]},
        {"prediction": 1, "probabilities": [0.05, 0.90, 0.05]},
        {"prediction": 2, "probabilities": [0.01, 0.04, 0.95]}
    ],
    "total_latency_ms": 0.12,
    "success_count": 3,
    "error_count": 0
}
```

### GET /model/info

Model metadata and capabilities.

**Response:**
```json
{
    "model_type": "logistic_regression",
    "version": "0.9.0",
    "features": {
        "count": 4,
        "names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    },
    "classes": ["setosa", "versicolor", "virginica"],
    "created_at": "2025-11-26T12:00:00Z"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "uptime_seconds": 3600
}
```

### GET /metrics

Prometheus-compatible metrics.

**Response:**
```prometheus
# HELP realizar_requests_total Total requests processed
# TYPE realizar_requests_total counter
realizar_requests_total 10000

# HELP realizar_latency_avg_ms Average inference latency
# TYPE realizar_latency_avg_ms gauge
realizar_latency_avg_ms 0.045
```

## Request Types

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub features: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub prediction: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<Vec<f32>>,
    pub latency_ms: f64,
}
```

## Error Handling

All errors return JSON with appropriate HTTP status codes:

```json
{
    "error": "InvalidFeatures",
    "message": "Expected 4 features, got 3",
    "status": 400
}
```

| Status | Error Type | Description |
|--------|------------|-------------|
| 400 | InvalidFeatures | Wrong feature count or format |
| 404 | ModelNotFound | Model not loaded |
| 500 | InferenceError | Internal prediction error |
| 503 | ServiceUnavailable | Server starting up |

## Starting the Server

```rust
use realizar::serve::start_aprender_server;

#[tokio::main]
async fn main() {
    let model_path = "models/classifier.apr";
    start_aprender_server(model_path, "0.0.0.0:3000").await?;
}
```

Or via CLI:

```bash
realizar serve --model models/classifier.apr --bind 0.0.0.0:3000
```
