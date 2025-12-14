# Case Study: APR with JSON Metadata

This case study demonstrates embedding arbitrary JSON metadata (vocabulary, tokenizer config, model settings) alongside tensor data in a single `.apr` file for WASM-ready deployment.

## The Problem

Modern ML models need more than just weights:

| Data Type | Traditional Approach | Problem |
|-----------|---------------------|---------|
| Vocabulary | Separate `vocab.json` | Multiple files to manage |
| Tokenizer | Separate `tokenizer.json` | Version mismatches |
| Config | Separate `config.yaml` | Deployment complexity |
| Custom | Application-specific files | N+1 file problem |

## The Solution: Embedded Metadata

The `.apr` format supports arbitrary JSON metadata embedded directly in the model file:

```rust,ignore
use aprender::serialization::apr::{AprWriter, AprReader};
use serde_json::json;

let mut writer = AprWriter::new();

// Embed any JSON metadata
writer.set_metadata("model_type", json!("whisper-tiny"));
writer.set_metadata("n_vocab", json!(51865));
writer.set_metadata("tokenizer", json!({
    "tokens": ["<|endoftext|>", "<|startoftranscript|>"],
    "merges": [["t", "h"], ["th", "e"]],
    "special_tokens": {"eot": 50256, "sot": 50257}
}));

// Add tensors
writer.add_tensor_f32("encoder.weight", vec![384, 80], &weights);

// Single file contains everything
writer.write("model.apr")?;
```

## Complete Example

Run: `cargo run --example apr_with_metadata`

```rust,ignore
{{#include ../../../examples/apr_with_metadata.rs}}
```

## Key Features

### 1. Arbitrary JSON Metadata

Any JSON-serializable data can be embedded:

```rust,ignore
// Strings
writer.set_metadata("model_name", json!("my-model"));

// Numbers
writer.set_metadata("n_layers", json!(12));

// Arrays
writer.set_metadata("supported_languages", json!(["en", "es", "fr"]));

// Objects
writer.set_metadata("config", json!({
    "hidden_size": 768,
    "num_attention_heads": 12
}));
```

### 2. Type-Safe Tensor Storage

Tensors are stored with shape information:

```rust,ignore
writer.add_tensor_f32("layer.0.weight", vec![768, 768], &weights);
writer.add_tensor_f32("layer.0.bias", vec![768], &bias);
```

### 3. Single-File Deployment

Perfect for WASM:

```rust,ignore
// Embed at compile time
const MODEL: &[u8] = include_bytes!("model.apr");

fn inference(input: &[f32]) -> Vec<f32> {
    let reader = AprReader::from_bytes(MODEL.to_vec()).unwrap();

    // Access metadata
    let vocab = reader.get_metadata("tokenizer").unwrap();

    // Access tensors
    let weights = reader.read_tensor_f32("encoder.weight").unwrap();

    // ... inference logic
}
```

## Use Cases

### Speech Recognition (Whisper-style)

```rust,ignore
writer.set_metadata("tokenizer", json!({
    "tokens": vocab_tokens,
    "merges": bpe_merges,
    "special_tokens": {
        "eot": 50256,
        "sot": 50257,
        "transcribe": 50358,
        "translate": 50359
    }
}));
```

### Language Models

```rust,ignore
writer.set_metadata("tokenizer", json!({
    "type": "BPE",
    "vocab_size": 32000,
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>"
}));
```

### Custom Models

```rust,ignore
writer.set_metadata("preprocessing", json!({
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "input_size": [224, 224]
}));
```

## Benefits

| Benefit | Description |
|---------|-------------|
| Single file | No more managing multiple files |
| Version-locked | Metadata travels with weights |
| WASM-ready | Embed entire model in binary |
| Type-safe | CRC32 checksum for integrity |
| Flexible | Any JSON structure supported |

## Related Resources

- [The .apr Format: A Five Whys Deep Dive](./apr-format-deep-dive.md)
- [APR Loading Modes](./apr-loading-modes.md)
- [APR Model Inspection](./apr-inspection.md)
