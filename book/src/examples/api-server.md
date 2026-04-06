# API Server

Production-ready HTTP API server built with Axum for serving transformer models.

## Features

- **REST API**: Text generation, tokenization, and batch processing endpoints
- **Multi-Model Serving**: Host and serve multiple models simultaneously with model selection via API
- **Streaming**: Server-Sent Events (SSE) for real-time token streaming
- **Production Monitoring**: Prometheus-compatible metrics endpoint
- **Backward Compatible**: Single-model mode still supported for simple deployments

## Quick Start

### Single Model Mode

Basic deployment with one model:

```rust
use realizar::api::{create_router, AppState};
use realizar::layers::{Model, ModelConfig};
use realizar::tokenizer::BPETokenizer;

// Create model and tokenizer
let config = ModelConfig {
    vocab_size: 1000,
    hidden_dim: 512,
    num_heads: 8,
    num_layers: 6,
    intermediate_dim: 2048,
    eps: 1e-5,
};
let model = Model::new(config)?;
let tokenizer = BPETokenizer::new(vocab, merges, "<unk>")?;

// Create server
let state = AppState::new(model, tokenizer);
let app = create_router(state);

// Serve on port 3000
let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
axum::serve(listener, app).await?;
```

### Multi-Model Mode

Serve multiple models with model selection:

```rust
use realizar::api::{create_router, AppState};
use realizar::registry::{ModelRegistry, ModelInfo};

// Create registry
let registry = ModelRegistry::new(5); // Cache up to 5 models

// Register models
let (llama_model, llama_tokenizer) = load_llama_7b()?;
let (mistral_model, mistral_tokenizer) = load_mistral_7b()?;

registry.register("llama-7b", llama_model, llama_tokenizer)?;

let mistral_info = ModelInfo {
    id: "mistral-7b".to_string(),
    name: "Mistral 7B".to_string(),
    description: "Mistral 7B instruction model".to_string(),
    format: "GGUF".to_string(),
    loaded: true,
};
registry.register_with_info(mistral_info, mistral_model, mistral_tokenizer)?;

// Create server with registry and default model
let state = AppState::with_registry(registry, "llama-7b")?;
let app = create_router(state);

// Serve
let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
axum::serve(listener, app).await?;
```

## API Endpoints

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### List Models

```bash
GET /models
```

**Response (Multi-Model Mode):**
```json
{
  "models": [
    {
      "id": "llama-7b",
      "name": "Llama 7B",
      "description": "7B parameter Llama model",
      "format": "GGUF",
      "loaded": true
    },
    {
      "id": "mistral-7b",
      "name": "Mistral 7B",
      "description": "Mistral 7B instruction model",
      "format": "GGUF",
      "loaded": true
    }
  ]
}
```

**Response (Single-Model Mode):**
```json
{
  "models": [
    {
      "id": "default",
      "name": "Default Model",
      "description": "Single model deployment",
      "format": "unknown",
      "loaded": true
    }
  ]
}
```

### Tokenize Text

```bash
POST /tokenize
Content-Type: application/json

{
  "text": "Hello, world!",
  "model_id": "llama-7b"  // Optional, uses default if omitted
}
```

**Response:**
```json
{
  "token_ids": [15496, 11, 1917, 0],
  "num_tokens": 4
}
```

### Generate Text

```bash
POST /generate
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "temperature": 0.8,
  "strategy": "top_p",
  "top_k": 50,
  "top_p": 0.9,
  "seed": 42,
  "model_id": "mistral-7b"  // Optional, uses default if omitted
}
```

**Sampling Strategies:**
- `"greedy"`: Always pick highest probability token (deterministic)
- `"top_k"`: Sample from top-k most likely tokens
- `"top_p"`: Nucleus sampling with cumulative probability threshold

**Response:**
```json
{
  "token_ids": [345, 678, 901, ...],
  "text": " there was a brave knight who lived in a castle...",
  "num_generated": 50
}
```

### Stream Generation

Server-Sent Events endpoint for real-time token streaming:

```bash
POST /stream/generate
Content-Type: application/json

{
  "prompt": "Tell me a story",
  "max_tokens": 100,
  "temperature": 0.8,
  "strategy": "greedy",
  "top_k": 50,
  "top_p": 0.9,
  "seed": 42,
  "model_id": "llama-7b"  // Optional
}
```

**Response (SSE Stream):**
```
data: {"token":"Once"}

data: {"token":" upon"}

data: {"token":" a"}

data: {"token":" time"}

data: {"done":true}
```

### Batch Tokenize

```bash
POST /batch/tokenize
Content-Type: application/json

{
  "texts": [
    "First text",
    "Second text",
    "Third text"
  ]
}
```

**Response:**
```json
{
  "results": [
    {"token_ids": [123, 456], "num_tokens": 2},
    {"token_ids": [789, 012], "num_tokens": 2},
    {"token_ids": [345, 678], "num_tokens": 2}
  ]
}
```

### Batch Generate

```bash
POST /batch/generate
Content-Type: application/json

{
  "prompts": ["Hello", "Goodbye"],
  "max_tokens": 10,
  "temperature": 0.8,
  "strategy": "greedy",
  "top_k": 50,
  "top_p": 0.9,
  "seed": 42
}
```

**Response:**
```json
{
  "results": [
    {
      "token_ids": [1, 2, 3],
      "text": " world, how are you?",
      "num_generated": 5
    },
    {
      "token_ids": [4, 5, 6],
      "text": " friend, see you later!",
      "num_generated": 6
    }
  ]
}
```

### Metrics

Prometheus-compatible metrics for monitoring:

```bash
GET /metrics
```

**Response:**
```
# HELP realizar_total_requests Total number of inference requests
# TYPE realizar_total_requests counter
realizar_total_requests 1234

# HELP realizar_successful_requests Number of successful requests
# TYPE realizar_successful_requests counter
realizar_successful_requests 1200

# HELP realizar_failed_requests Number of failed requests
# TYPE realizar_failed_requests counter
realizar_failed_requests 34

# HELP realizar_total_tokens Total tokens generated
# TYPE realizar_total_tokens counter
realizar_total_tokens 56789

# HELP realizar_total_latency_ms Total inference latency in milliseconds
# TYPE realizar_total_latency_ms counter
realizar_total_latency_ms 45678.5

# HELP realizar_avg_latency_ms Average inference latency in milliseconds
# TYPE realizar_avg_latency_ms gauge
realizar_avg_latency_ms 38.1

# HELP realizar_tokens_per_second Throughput in tokens per second
# TYPE realizar_tokens_per_second gauge
realizar_tokens_per_second 1243.5
```

## Multi-Model Selection

When running in multi-model mode, specify the model to use via the optional `model_id` field:

```bash
# Use specific model
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10,
    "model_id": "mistral-7b"
  }'

# Use default model (omit model_id)
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10
  }'
```

**Error Handling:**
```bash
# Non-existent model returns 404
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10,
    "model_id": "does-not-exist"
  }'

# Response: 404 Not Found
{
  "error": "Model 'does-not-exist' not found"
}
```

## Concurrent Access

The registry supports thread-safe concurrent access to multiple models:

```rust
// Multiple requests can access different models simultaneously
// Each model serves requests independently with no blocking
tokio::spawn(async {
    client.post("/generate")
        .json(&json!({"prompt": "...", "model_id": "llama-7b"}))
        .send().await
});

tokio::spawn(async {
    client.post("/generate")
        .json(&json!({"prompt": "...", "model_id": "mistral-7b"}))
        .send().await
});
```

## Production Deployment

### Docker

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/realizar /usr/local/bin/
EXPOSE 3000
CMD ["realizar", "serve", "--host", "0.0.0.0", "--port", "3000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realizar-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: realizar
  template:
    metadata:
      labels:
        app: realizar
    spec:
      containers:
      - name: realizar
        image: realizar:latest
        ports:
        - containerPort: 3000
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: realizar-service
spec:
  selector:
    app: realizar
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

## Testing

The API has comprehensive test coverage:

- **26 unit tests** in `src/api.rs` covering all endpoints and error cases
- **10 integration tests** in `tests/integration_multi_model_api.rs` for multi-model functionality
- **100% mutation score** on mutation testing (18/18 mutants caught)

Run tests:
```bash
cargo test --lib --features server  # Unit tests
cargo test --test integration_multi_model_api --features server  # Integration tests
```

## Performance

Benchmarks on AMD64 with SIMD acceleration:

- **Single request latency**: <100ms p50, <200ms p95
- **Throughput**: 1000+ tokens/second sustained
- **Batch processing**: 3x speedup for batch size 10
- **Memory overhead**: <512MB per model beyond model weights

## Security

- **Input validation**: All requests validated for proper types and ranges
- **Error messages**: Safe error messages without internal details
- **Rate limiting**: Implement via reverse proxy (nginx, envoy)
- **Authentication**: Implement via middleware or API gateway

## Related Examples

- See [`examples/api_server.rs`](https://github.com/paiml/realizar/blob/main/examples/api_server.rs) for complete working example
- See [`tests/integration_multi_model_api.rs`](https://github.com/paiml/realizar/blob/main/tests/integration_multi_model_api.rs) for usage patterns
