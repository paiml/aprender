# The .apr Format

The `.apr` format is Aprender's native model serialization format, designed for fast loading and minimal parsing overhead.

## Binary Structure

```
┌────────────────────────────────────────┐
│  Magic: "APR\0" (4 bytes)              │  ← Identifies .apr files
├────────────────────────────────────────┤
│  Version: u32 (4 bytes, little-endian) │  ← Format version
├────────────────────────────────────────┤
│  Model Type: u32 (4 bytes)             │  ← Algorithm identifier
├────────────────────────────────────────┤
│  Metadata Length: u64 (8 bytes)        │  ← JSON metadata size
├────────────────────────────────────────┤
│  Metadata: JSON (variable)             │  ← Hyperparameters, features
├────────────────────────────────────────┤
│  Weights: f32[] (variable)             │  ← Model parameters
└────────────────────────────────────────┘
```

## Magic Bytes

All valid `.apr` files start with `APR\0` (0x41, 0x50, 0x52, 0x00):

```rust
const APR_MAGIC: [u8; 4] = [b'A', b'P', b'R', 0x00];

fn validate_magic(bytes: &[u8]) -> Result<(), AprError> {
    if bytes.len() < 4 {
        return Err(AprError::TooSmall);
    }
    if &bytes[0..4] != &APR_MAGIC {
        return Err(AprError::InvalidMagic {
            expected: APR_MAGIC,
            actual: [bytes[0], bytes[1], bytes[2], bytes[3]],
        });
    }
    Ok(())
}
```

## Model Types

| Type ID | Model | Description |
|---------|-------|-------------|
| 0x01 | Linear Regression | y = Xw + b |
| 0x02 | Logistic Regression | Binary classification |
| 0x03 | Decision Tree | Tree-based classifier |
| 0x04 | Random Forest | Ensemble of trees |
| 0x05 | K-Means | Clustering |
| 0x06 | Naive Bayes | Probabilistic classifier |
| 0x10 | MLP | Multi-layer perceptron |

## Metadata JSON

```json
{
  "model_type": "logistic_regression",
  "version": "0.9.0",
  "created_at": "2025-11-26T12:00:00Z",
  "features": {
    "count": 4,
    "names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  },
  "hyperparameters": {
    "learning_rate": 0.01,
    "regularization": "l2",
    "lambda": 0.001
  },
  "training": {
    "epochs": 100,
    "final_loss": 0.0234,
    "samples": 150
  }
}
```

## Reading .apr Files

```rust
use std::io::Read;

fn read_apr(path: &str) -> Result<Model, AprError> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    // Validate magic
    if &bytes[0..4] != b"APR\0" {
        return Err(AprError::InvalidMagic);
    }

    // Parse version
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

    // Parse model type
    let model_type = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

    // Parse metadata length
    let meta_len = u64::from_le_bytes(
        bytes[12..20].try_into().unwrap()
    ) as usize;

    // Parse metadata JSON
    let metadata: Metadata = serde_json::from_slice(&bytes[20..20+meta_len])?;

    // Load weights
    let weights_start = 20 + meta_len;
    let weights = load_f32_array(&bytes[weights_start..]);

    Ok(Model { version, model_type, metadata, weights })
}
```

## Embedding in Lambda

```rust
// Compile model into binary
static MODEL_BYTES: &[u8] = include_bytes!("../models/iris_classifier.apr");

// Load at runtime
let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;
```

## Comparison with Other Formats

| Format | Parse Time | Size | Features |
|--------|------------|------|----------|
| .apr | ~10µs | Small | Native aprender |
| ONNX | ~1ms | Medium | Universal, complex |
| Pickle | ~100ms | Large | Python-only |
| Safetensors | ~100µs | Medium | Zero-copy, any framework |

## Best Practices

1. **Validate magic bytes** before parsing
2. **Check version** for compatibility
3. **Use `include_bytes!()`** for Lambda deployment
4. **Store metadata** for model provenance
5. **Keep models small** (<10MB) for fast Lambda cold start
