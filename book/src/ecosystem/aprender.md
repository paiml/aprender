# Aprender

Machine learning framework for the Sovereign AI Stack.

## Overview

| Feature | Description |
|---------|-------------|
| Format | `.apr` model files |
| Runtime | WASM-compatible inference |
| Backend | trueno for SIMD operations |
| Zero-copy | Mmap for large models |

## Model Format

```
.apr file structure:
┌─────────────────────┐
│ Header (64 bytes)   │
│ - Magic: "APR\0"    │
│ - Version: u32      │
│ - Architecture      │
├─────────────────────┤
│ Metadata            │
│ - Input shape       │
│ - Output shape      │
│ - Hyperparameters   │
├─────────────────────┤
│ Weights             │
│ - Layer data        │
│ - Quantization      │
└─────────────────────┘
```

## Inference in Presentar

```rust
// Load model
let model = aprender::load("classifier.apr")?;

// Run inference
let input = trueno::Tensor::from_slice(&[1.0, 2.0, 3.0]);
let output = model.forward(&input);
```

## Model Card

Every `.apr` requires a model card:

| Field | Required |
|-------|----------|
| Name | Yes |
| Version | Yes |
| Task | Yes |
| Metrics | Yes |
| Limitations | Yes |

## WASM Compatibility

```rust
// Models run entirely in browser
#[cfg(target_arch = "wasm32")]
let model = aprender::load_wasm(bytes)?;
```

## Verified Test

```rust
#[test]
fn test_aprender_model_card_fields() {
    // Model card validation
    struct ModelCard {
        name: String,
        version: String,
        task: String,
        accuracy: f32,
        limitations: Vec<String>,
    }

    let card = ModelCard {
        name: "digit-classifier".to_string(),
        version: "1.0.0".to_string(),
        task: "classification".to_string(),
        accuracy: 0.98,
        limitations: vec!["28x28 grayscale only".to_string()],
    };

    assert!(!card.name.is_empty());
    assert!(!card.version.is_empty());
    assert!(!card.task.is_empty());
    assert!(card.accuracy > 0.0 && card.accuracy <= 1.0);
    assert!(!card.limitations.is_empty());
}
```
