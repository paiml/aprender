# Aprender Model Serving

Realizar provides native support for serving [Aprender](https://github.com/paiml/aprender) ML models via the `.apr` format.

## What is Aprender?

Aprender is a pure Rust ML library built from scratch, providing:

- **Linear Regression** with L1/L2 regularization
- **Logistic Regression** for classification
- **Decision Trees** and Random Forests
- **K-Means Clustering**
- **Naive Bayes** classifiers
- **Neural Networks** (MLP)

## The .apr Format

The `.apr` format is Aprender's native model serialization:

```
┌────────────────────────────────────────┐
│  Magic: "APR\0" (4 bytes)              │
├────────────────────────────────────────┤
│  Version: u32 (4 bytes)                │
├────────────────────────────────────────┤
│  Model Type: u32 (4 bytes)             │
├────────────────────────────────────────┤
│  Metadata: JSON (variable)             │
├────────────────────────────────────────┤
│  Weights: f32[] (variable)             │
└────────────────────────────────────────┘
```

### Magic Bytes Validation

```rust
// Valid .apr files start with "APR\0"
const APR_MAGIC: [u8; 4] = [b'A', b'P', b'R', 0];

fn validate_apr(bytes: &[u8]) -> Result<(), LambdaError> {
    if bytes.len() < 4 || &bytes[0..4] != &APR_MAGIC {
        return Err(LambdaError::InvalidMagic { ... });
    }
    Ok(())
}
```

## Feature Flag

Enable aprender serving:

```toml
[dependencies]
realizar = { version = "0.2", features = ["aprender-serve"] }
```

## HTTP API

When `aprender-serve` is enabled, the HTTP server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/health` | GET | Health check |

### Predict Request

```json
{
    "features": [1.0, 2.0, 3.0, 4.0]
}
```

### Predict Response

```json
{
    "prediction": 0.85,
    "probabilities": [0.15, 0.85],
    "latency_ms": 0.045
}
```

## Performance Targets

Per `docs/specifications/serve-deploy-apr.md`:

| Metric | Target | Achieved |
|--------|--------|----------|
| Cold Start | <50ms | ~100µs |
| Warm Inference (p50) | <10ms | 35-676ns |
| Throughput | >1000 req/s | ✅ |

## Integration with Lambda

```rust
use realizar::lambda::{LambdaHandler, LambdaRequest};

// Embed .apr model at compile time
static MODEL: &[u8] = include_bytes!("../models/classifier.apr");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handler = LambdaHandler::from_bytes(MODEL)?;

    let request = LambdaRequest {
        features: vec![5.1, 3.5, 1.4, 0.2],  // Iris features
        model_id: None,
    };

    let response = handler.handle(&request)?;
    println!("Class: {}", response.prediction);

    Ok(())
}
```

## Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                     Aprender Ecosystem                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Aprender   │───▶│   .apr      │───▶│  Realizar   │     │
│  │  (training) │    │   file      │    │  (serving)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                                     │             │
│         ▼                                     ▼             │
│  ┌─────────────┐                      ┌─────────────┐      │
│  │   Trueno    │                      │   Lambda    │      │
│  │  (compute)  │                      │   Docker    │      │
│  └─────────────┘                      │   WASM      │      │
│                                       └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```
