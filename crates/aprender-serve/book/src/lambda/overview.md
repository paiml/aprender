# AWS Lambda Serving

Realizar supports serverless deployment via AWS Lambda with optimized cold start performance and ARM64 Graviton support.

## Architecture

Per `docs/specifications/serve-deploy-apr.md` Section 7:

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Lambda Runtime                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LambdaHandler                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Model Bytes │  │  OnceLock   │  │  Metrics   │  │   │
│  │  │ (embedded)  │  │ (lazy init) │  │ (Prom)     │  │   │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Target: <50ms cold start, <10ms warm inference (p50)       │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Single Binary** | Model embedded via `include_bytes!()` |
| **Lazy Init** | `OnceLock` amortizes cold start across requests |
| **ARM64 Native** | Optimized for AWS Graviton processors |
| **Batch Support** | Process multiple predictions in one invocation |
| **Prometheus Metrics** | Production-grade observability |

## Performance Targets

From benchmark results:

- **Cold Start**: <50ms target (achieved: ~100µs handler creation)
- **Warm Inference**: <10ms p50 (achieved: 35-676ns depending on input size)
- **Batch Throughput**: Linear scaling with batch size

## Usage

```rust
use realizar::lambda::{LambdaHandler, LambdaRequest, LambdaResponse};

// Model bytes embedded at compile time
static MODEL_BYTES: &[u8] = include_bytes!("../models/model.apr");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;

    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0],
        model_id: None,
    };

    let response = handler.handle(&request)?;
    println!("Prediction: {}", response.prediction);
    println!("Latency: {}ms", response.latency_ms);
    println!("Cold start: {}", response.cold_start);

    Ok(())
}
```

## Feature Flag

Enable Lambda support in `Cargo.toml`:

```toml
[dependencies]
realizar = { version = "0.2", features = ["lambda"] }
```

The `lambda` feature is intentionally lightweight - it doesn't pull in HTTP server dependencies, making the binary smaller for Lambda deployment.
