# Auxiliary Data Patterns in .apr Format

**Status:** Reference
**Author:** Claude Code
**Created:** 2025-12-16

## Overview

This document describes patterns for storing auxiliary (non-tensor) data in `.apr` model files. Examples include vocabulary, tokenizer configuration, mel filterbanks, and other model-specific data.

## Pattern: JSON Metadata

The primary pattern for auxiliary data is the JSON metadata section.

### Format Location

```
[APR1 magic (4 bytes)]
[metadata_len (4 bytes)]
[JSON metadata] ← Auxiliary data goes here
[tensor count]
[tensor index]
[tensor data]
[CRC32]
```

### API

```rust
// Writing
let mut writer = AprWriter::new();
writer.set_metadata("key", JsonValue::...);

// Reading
let reader = AprReader::open(path)?;
let value = reader.get_metadata("key");
```

## Common Auxiliary Data Types

### 1. Vocabulary (NLP Models)

```json
{
  "vocab": ["<pad>", "<unk>", "<s>", "</s>", "the", "a", ...],
  "vocab_size": 51865
}
```

**Use case:** BPE/WordPiece tokenizers, language models

### 2. BPE Merges (NLP Models)

```json
{
  "bpe_merges": [
    ["t", "h"],
    ["th", "e"],
    ["the", " "],
    ...
  ]
}
```

**Use case:** GPT-style BPE tokenization

### 3. Mel Filterbank (Audio Models)

```json
{
  "mel_filterbank": [0.0, 0.0, ..., 0.0234, ...],
  "mel_filterbank_shape": [80, 201]
}
```

**Use case:** Whisper, speech recognition models

**Note:** Filterbank weights MUST match the training implementation exactly. Do not compute at runtime.

### 4. Tokenizer Config (NLP Models)

```json
{
  "tokenizer_config": {
    "type": "bpe",
    "unk_token": "<|unk|>",
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
    "add_prefix_space": false
  }
}
```

### 5. Model Architecture (All Models)

```json
{
  "architecture": {
    "model_type": "whisper",
    "n_layers": 4,
    "n_heads": 6,
    "d_model": 384,
    "vocab_size": 51865
  }
}
```

### 6. Image Preprocessing (Vision Models)

```json
{
  "image_config": {
    "image_size": 224,
    "patch_size": 16,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

### 7. Label Mapping (Classification Models)

```json
{
  "labels": {
    "0": "cat",
    "1": "dog",
    "2": "bird"
  },
  "num_labels": 3
}
```

## Alternative: Store as Tensor

For large auxiliary data (> 1MB), consider storing as a named tensor instead:

```rust
// Writing
writer.add_tensor_f32("audio.mel_filterbank", vec![80, 201], &filterbank_data);

// Reading
let filterbank = reader.read_tensor_f32("audio.mel_filterbank")?;
```

### Decision Matrix

| Data Size | JSON Metadata | Tensor |
|-----------|---------------|--------|
| < 100KB | Preferred | Overkill |
| 100KB - 1MB | Acceptable | Good |
| > 1MB | Avoid | Preferred |

### Naming Convention for Auxiliary Tensors

Use dot-separated namespaces:

```
audio.mel_filterbank
text.token_embedding
image.patch_embedding
```

## GGUF Comparison

GGUF uses typed key-value metadata:

| GGUF Type | APR Equivalent |
|-----------|----------------|
| `string` | `JsonValue::String` |
| `uint32` | `JsonValue::Number` |
| `float32` | `JsonValue::Number` |
| `array[string]` | `JsonValue::Array` |
| `array[float32]` | `JsonValue::Array` or tensor |

## SafeTensors Comparison

SafeTensors stores metadata as JSON header:

```json
{
  "__metadata__": {
    "format": "pt",
    "model_type": "whisper"
  },
  "encoder.conv1.weight": {"dtype": "F32", "shape": [384, 80, 3], "data_offsets": [0, 92160]}
}
```

APR separates metadata from tensor index for clarity.

## Best Practices

1. **Use standard keys**: Follow conventions from HuggingFace/GGUF where applicable
2. **Include shape info**: Always store shape alongside flattened arrays
3. **Version metadata**: Include `format_version` for future compatibility
4. **Document units**: Specify if values are normalized, in Hz, etc.
5. **Validate on load**: Check array lengths match expected shapes

## Example: Complete Whisper Metadata

```json
{
  "format_version": "1.0",
  "model_type": "whisper",
  "model_size": "tiny",

  "architecture": {
    "n_vocab": 51865,
    "n_audio_ctx": 1500,
    "n_text_ctx": 448,
    "n_mels": 80,
    "n_audio_layer": 4,
    "n_text_layer": 4,
    "n_audio_head": 6,
    "n_text_head": 6,
    "n_audio_state": 384,
    "n_text_state": 384
  },

  "audio": {
    "sample_rate": 16000,
    "n_fft": 400,
    "hop_length": 160,
    "chunk_length": 30
  },

  "mel_filterbank": [/* 16080 floats */],
  "mel_filterbank_shape": [80, 201],

  "vocab": [/* 51865 tokens */],

  "special_tokens": {
    "sot": 50258,
    "eot": 50257,
    "translate": 50358,
    "transcribe": 50359,
    "no_timestamps": 50363
  }
}
```

## References

- APR format: `src/serialization/apr.rs`
- GGUF metadata: `src/format/gguf.rs`
- HuggingFace config: https://huggingface.co/docs/transformers/main_classes/configuration
