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

### Audio Models (Mel Filterbank)

For speech recognition models like Whisper, embedding the exact mel filterbank used during training is **critical** for correct transcription. Computing filterbanks at runtime produces different values due to normalization differences.

```rust,ignore
// Store filterbank as a named tensor (most efficient for 64KB+ data)
writer.add_tensor_f32(
    "audio.mel_filterbank",
    vec![80, 201],  // n_mels x n_freqs
    &filterbank_data,
);

// Store audio preprocessing config in metadata
writer.set_metadata("audio", json!({
    "sample_rate": 16000,
    "n_fft": 400,
    "hop_length": 160,
    "n_mels": 80
}));
```

Reading back:

```rust,ignore
let reader = AprReader::from_bytes(model_bytes)?;

// Read filterbank tensor
let filterbank = reader.read_tensor_f32("audio.mel_filterbank")?;

// Get audio config
let audio_config = reader.get_metadata("audio").unwrap();
let n_mels = audio_config["n_mels"].as_u64().unwrap() as usize;
let n_freqs = filterbank.len() / n_mels;

// Use filterbank for mel spectrogram computation
let mel_spectrogram = compute_mel(&audio_samples, &filterbank, n_mels, n_freqs);
```

**Why this matters:** Whisper was trained with librosa's slaney-normalized filterbank where row sums are ~0.025. Computing from scratch produces peak-normalized filterbanks with row sums of ~1.0+. This mismatch causes the "rererer" hallucination bug.

## Benefits

| Benefit | Description |
|---------|-------------|
| Single file | No more managing multiple files |
| Version-locked | Metadata travels with weights |
| WASM-ready | Embed entire model in binary |
| Type-safe | CRC32 checksum for integrity |
| Flexible | Any JSON structure supported |

## Binary Data: Metadata vs Tensor

When storing binary data (filterbanks, embeddings), choose the right approach:

| Data Size | JSON Metadata | Named Tensor |
|-----------|---------------|--------------|
| < 100KB | Preferred | Overkill |
| 100KB - 1MB | Acceptable | Recommended |
| > 1MB | Avoid (slow JSON parsing) | Required |

**Mel filterbank (64KB):** Both work; tensor is more efficient.

**Vocabulary (1-5MB):** Use JSON for string arrays, tensor for embedding matrices.

**Large embeddings (>10MB):** Always use tensors.

## Related Resources

- [The .apr Format: A Five Whys Deep Dive](./apr-format-deep-dive.md)
- [APR Loading Modes](./apr-loading-modes.md)
- [APR Model Inspection](./apr-inspection.md)
