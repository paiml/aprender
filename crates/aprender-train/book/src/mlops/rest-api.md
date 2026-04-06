# REST API Server

HTTP API for remote experiment tracking, enabling distributed training workflows and web dashboards.

## Toyota Principle: Jidoka

Built-in quality through structured API responses and validation at every endpoint.

## Quick Start

```rust
use entrenar::server::{TrackingServer, ServerConfig};

// Configure server
let config = ServerConfig::default()
    .with_host("0.0.0.0")
    .with_port(5000);

// Create and run server
let server = TrackingServer::new(config);
server.run().await?;
```

## API Endpoints

### Health Check

```bash
GET /health

# Response
{
  "status": "healthy",
  "version": "0.2.3",
  "uptime_secs": 3600
}
```

### Experiments

```bash
# Create experiment
POST /api/v1/experiments
Content-Type: application/json

{
  "name": "gpt2-finetune",
  "config": {
    "model": "gpt2",
    "dataset": "wikitext"
  }
}

# Response
{
  "id": "exp-abc123",
  "name": "gpt2-finetune",
  "created_at": "2024-12-05T10:30:00Z"
}

# List experiments
GET /api/v1/experiments

# Get experiment
GET /api/v1/experiments/{id}
```

### Runs

```bash
# Create run
POST /api/v1/runs
Content-Type: application/json

{
  "experiment_id": "exp-abc123"
}

# Response
{
  "id": "run-xyz789",
  "experiment_id": "exp-abc123",
  "status": "pending",
  "created_at": "2024-12-05T10:31:00Z"
}

# Start run
POST /api/v1/runs/{id}/start

# Complete run
POST /api/v1/runs/{id}/complete
Content-Type: application/json

{
  "status": "success"
}

# Get run
GET /api/v1/runs/{id}

# List runs for experiment
GET /api/v1/experiments/{id}/runs
```

### Metrics

```bash
# Log metric
POST /api/v1/runs/{id}/metrics
Content-Type: application/json

{
  "key": "loss",
  "step": 100,
  "value": 0.5
}

# Log batch of metrics
POST /api/v1/runs/{id}/metrics/batch
Content-Type: application/json

{
  "metrics": [
    {"key": "loss", "step": 100, "value": 0.5},
    {"key": "accuracy", "step": 100, "value": 0.85}
  ]
}

# Get metrics
GET /api/v1/runs/{id}/metrics?key=loss
```

### Parameters

```bash
# Log parameter
POST /api/v1/runs/{id}/params
Content-Type: application/json

{
  "key": "learning_rate",
  "value": 0.001
}

# Get parameters
GET /api/v1/runs/{id}/params
```

### Artifacts

```bash
# Upload artifact
POST /api/v1/runs/{id}/artifacts
Content-Type: multipart/form-data

# Response
{
  "key": "model.safetensors",
  "hash": "sha256:abc123...",
  "size_bytes": 1048576
}

# Download artifact
GET /api/v1/runs/{id}/artifacts/{key}
```

## Client SDK

```rust
use entrenar::server::client::TrackingClient;

let client = TrackingClient::new("http://localhost:5000");

// Create experiment
let exp = client.create_experiment("my-experiment", None).await?;

// Create and start run
let run = client.create_run(&exp.id).await?;
client.start_run(&run.id).await?;

// Log metrics
client.log_metric(&run.id, "loss", 0, 0.5).await?;

// Complete run
client.complete_run(&run.id, "success").await?;
```

## Server Configuration

```rust
use entrenar::server::ServerConfig;

let config = ServerConfig::default()
    .with_host("0.0.0.0")
    .with_port(5000)
    .with_cors(true)
    .with_max_body_size(100 * 1024 * 1024)  // 100MB
    .with_request_timeout_secs(300);
```

## Authentication

```rust
use entrenar::server::ServerConfig;

// API key authentication
let config = ServerConfig::default()
    .with_auth_type("api_key")
    .with_api_keys(vec!["key1", "key2"]);

// Request with API key
// Authorization: Bearer key1
```

## Cargo Run Example

```bash
# Start server
cargo run --features server -- server

# With custom port
cargo run --features server -- server --port 8080

# With API key auth
cargo run --features server -- server --auth-type api_key --api-key mysecretkey
```

## Docker Deployment

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features server

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/entrenar /usr/local/bin/
EXPOSE 5000
CMD ["entrenar", "server", "--host", "0.0.0.0", "--port", "5000"]
```

```bash
docker build -t entrenar-server .
docker run -p 5000:5000 entrenar-server
```

## Integration with Training

```rust
use entrenar::train::Trainer;
use entrenar::server::client::TrackingClient;

let client = TrackingClient::new("http://localhost:5000");
let exp = client.create_experiment("training-job", None).await?;
let run = client.create_run(&exp.id).await?;

let trainer = Trainer::new(config)
    .with_tracking_client(client, &run.id);

trainer.fit(&model, &dataset)?;
```

## Error Responses

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Run not found: run-xyz789"
  }
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NOT_FOUND` | 404 | Resource not found |
| `INVALID_REQUEST` | 400 | Malformed request |
| `UNAUTHORIZED` | 401 | Missing/invalid auth |
| `INTERNAL_ERROR` | 500 | Server error |

## Best Practices

1. **Use batch endpoints** - Reduce HTTP overhead
2. **Enable CORS for web dashboards** - Required for browser access
3. **Set appropriate timeouts** - Long uploads need more time
4. **Use API key auth in production** - Never expose unauthenticated
5. **Deploy behind reverse proxy** - nginx/traefik for TLS

## See Also

- [MLOps Overview](./overview.md)
- [Experiment Tracking](./experiment-tracking.md)
- [Cloud Storage](./cloud-storage.md)
