# APR Complete Specification

**Version**: 2.0.0-draft
**Status**: Draft
**Created**: 2025-12-16
**GitHub Issue**: https://github.com/paiml/aprender/issues/119

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Design Principles](#2-design-principles)
3. [APR v2 Format](#3-apr-v2-format)
   - [3.1 Format Overview](#31-format-overview)
   - [3.2 Header](#32-header-32-bytes)
   - [3.3 Feature Flags](#33-feature-flags)
   - [3.4 Metadata Section](#34-metadata-section)
   - [3.5 Tensor Index](#35-tensor-index-binary)
   - [3.6 Tensor Data Section](#36-tensor-data-section)
   - [3.7 Footer](#37-footer-16-bytes)
   - [3.8 Sharding](#38-sharding-multi-file)
   - [3.9 WASM Considerations](#39-wasm-considerations)
4. [CLI Operations](#4-cli-operations)
   - [4.1 Command Overview](#41-command-overview)
   - [4.2 Inspect Command](#42-inspect-command)
   - [4.3 Debug Command](#43-debug-command-drama-mode)
   - [4.4 Validate Command](#44-validate-command)
   - [4.5 Diff Command](#45-diff-command)
   - [4.6 Export Command](#46-export-command)
   - [4.7 Import Command](#47-import-command)
   - [4.8 Convert Command](#48-convert-command)
   - [4.9 Merge Command](#49-merge-command)
   - [4.10 Trace Command](#410-trace-command)
5. [Auxiliary Data Patterns](#5-auxiliary-data-patterns)
   - [5.1 JSON Metadata Pattern](#51-json-metadata-pattern)
   - [5.2 Common Auxiliary Data Types](#52-common-auxiliary-data-types)
   - [5.3 Tensor Storage for Large Data](#53-tensor-storage-for-large-data)
   - [5.4 Best Practices](#54-best-practices)
6. [Format Comparison](#6-format-comparison)
7. [Error Handling](#7-error-handling)
8. [Configuration](#8-configuration)
9. [Quality Gates](#9-quality-gates)
10. [Falsification QA Checklist](#10-falsification-qa-checklist)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [References](#12-references)
13. [Appendices](#13-appendices)

---

## 1. Abstract

APR (Aprender Portable Representation) is a WASM-first model serialization format for machine learning models. This specification covers:

- **APR v2 Format**: Binary format supporting web-scale models (10B+ parameters) with tensor alignment, LZ4 streaming compression, and multi-file sharding
- **CLI Operations**: Comprehensive tooling for inspect, debug, trace, export, convert, import, merge, diff, and validate operations
- **Auxiliary Data**: Patterns for storing vocabulary, tokenizer config, mel filterbanks, and other model-specific data

---

## 2. Design Principles

### 2.1 WASM-First Design

1. **WASM-first**: Must work in `wasm32-unknown-unknown` without Emscripten
2. **Progressive enhancement**: Features degrade gracefully (mmap → heap, compression → raw)
3. **Backward compatibility**: APR1 files remain readable
4. **Zero-copy where possible**: Alignment enables direct tensor access
5. **Streaming**: Support chunked loading for large models

### 2.2 Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Go and see the actual model data, not abstractions |
| **Visualization** | Make model internals visible for debugging |
| **Jidoka** | Stop on quality issues (corrupted models, NaN weights) |
| **Kaizen** | Continuous improvement via diff and merge operations |
| **Standardization** | Consistent CLI interface across all operations |

---

## 3. APR v2 Format

### 3.1 Format Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Header (32 bytes, aligned)                                  │
├─────────────────────────────────────────────────────────────┤
│ Metadata Section (JSON, variable length)                    │
├─────────────────────────────────────────────────────────────┤
│ Tensor Index (binary, variable length)                      │
├─────────────────────────────────────────────────────────────┤
│ [Padding to 64-byte alignment]                              │
├─────────────────────────────────────────────────────────────┤
│ Tensor Data Section (aligned tensors)                       │
│   ├── Tensor 0 (64-byte aligned)                           │
│   ├── Tensor 1 (64-byte aligned)                           │
│   └── ...                                                   │
├─────────────────────────────────────────────────────────────┤
│ Footer (16 bytes)                                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Header (32 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | `APR2` (0x41505232) |
| 4 | 2 | version_major | Format major version (2) |
| 6 | 2 | version_minor | Format minor version (0) |
| 8 | 4 | flags | Feature flags (see below) |
| 12 | 4 | metadata_offset | Offset to metadata section |
| 16 | 4 | metadata_size | Size of metadata section |
| 20 | 4 | index_offset | Offset to tensor index |
| 24 | 4 | index_size | Size of tensor index |
| 28 | 4 | data_offset | Offset to tensor data section |

### 3.3 Feature Flags

```rust
bitflags! {
    pub struct AprFlags: u32 {
        const COMPRESSED     = 0b0000_0001;  // LZ4 compression enabled
        const ALIGNED_64     = 0b0000_0010;  // 64-byte tensor alignment
        const ALIGNED_32     = 0b0000_0100;  // 32-byte tensor alignment (GGUF compat)
        const SHARDED        = 0b0000_1000;  // Multi-file model
        const ENCRYPTED      = 0b0001_0000;  // AES-256-GCM encryption
        const SIGNED         = 0b0010_0000;  // Ed25519 signature present
        const QUANTIZED      = 0b0100_0000;  // Contains quantized tensors
        const STREAMING      = 0b1000_0000;  // Streaming-optimized layout
    }
}
```

### 3.4 Metadata Section

JSON object containing model configuration and auxiliary data.

#### Required Keys

```json
{
  "apr_version": "2.0.0",
  "model_type": "whisper",
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
  }
}
```

#### Optional Keys

```json
{
  "vocab": ["<|endoftext|>", "<|startoftranscript|>", "..."],
  "mel_filterbank": [0.0, 0.0, "..."],
  "mel_filterbank_shape": [80, 201],
  "tokenizer_config": { "..." },
  "model_card": { "..." },
  "quantization": {
    "method": "Q8_0",
    "bits_per_weight": 8.5
  }
}
```

### 3.5 Tensor Index (Binary)

#### Index Header (8 bytes)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | tensor_count |
| 4 | 4 | reserved |

#### Tensor Entry (variable, ~40+ bytes each)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 2 | name_len | Length of tensor name |
| 2 | name_len | name | UTF-8 tensor name |
| +0 | 1 | dtype | Data type enum |
| +1 | 1 | n_dims | Number of dimensions (1-8) |
| +2 | 8×n_dims | dims | Dimension sizes (u64 each) |
| +n | 8 | offset | Byte offset in data section |
| +n+8 | 8 | size | Compressed size (or raw size) |
| +n+16 | 8 | raw_size | Uncompressed size (0 if not compressed) |
| +n+24 | 4 | flags | Per-tensor flags |

#### Data Type Enum

```rust
#[repr(u8)]
pub enum DType {
    F32 = 0, F16 = 1, BF16 = 2, I8 = 3, I16 = 4, I32 = 5, I64 = 6, U8 = 7,
    Q8_0 = 16, Q4_0 = 17, Q4_1 = 18, Q5_0 = 19, Q5_1 = 20,
}
```

### 3.6 Tensor Data Section

Tensors stored contiguously with alignment padding.

- **Default**: 64-byte alignment (cache-line optimal)
- **GGUF-compatible**: 32-byte alignment
- **Compression**: Per-tensor LZ4 block compression (64KB blocks)

### 3.7 Footer (16 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | crc32 | CRC32 of all preceding bytes |
| 4 | 4 | magic_end | `2RPA` (reverse magic) |
| 8 | 8 | file_size | Total file size for validation |

### 3.8 Sharding (Multi-File)

For models > 2GB, use manifest + shard files.

```json
{
  "apr_version": "2.0.0",
  "sharded": true,
  "shard_count": 4,
  "shards": [
    {"file": "model-00001-of-00004.apr", "size": 2147483648, "crc32": "..."},
    {"file": "model-00002-of-00004.apr", "size": 2147483648, "crc32": "..."}
  ],
  "tensor_shard_map": {
    "encoder.conv1.weight": 0,
    "decoder.token_embedding.weight": 1
  }
}
```

### 3.9 WASM Considerations

```rust
pub trait StreamingLoader {
    fn load_metadata(&mut self) -> Result<AprMetadata>;
    fn load_index(&mut self) -> Result<Vec<TensorDescriptor>>;
    fn load_tensor(&mut self, name: &str) -> Result<Tensor>;
    fn prefetch(&mut self, names: &[&str]);
}
```

---

## 4. CLI Operations

### 4.1 Command Overview

```
apr - APR Model Operations Tool

COMMANDS:
    inspect     Inspect model metadata, vocab, and structure
    debug       Simple debugging output ("drama" mode)
    validate    Validate model integrity
    diff        Compare two models
    tensors     List tensor information
    export      Export model to other formats
    import      Import from external formats
    convert     Convert between model types
    merge       Merge multiple models
    trace       Trace model operations with renacer
```

### 4.2 Inspect Command

```bash
$ apr inspect whisper.apr

=== whisper.apr ===
Type:        NeuralCustom (Whisper ASR)
Version:     1.0
Size:        1.5 GB (compressed: 890 MB)
Parameters:  39,000,000
Vocab Size:  51,865
Flags:       COMPRESSED | SIGNED
Checksum:    0xA1B2C3D4 (valid)
```

Options: `--vocab`, `--filters`, `--json`, `--full`

### 4.3 Debug Command ("Drama" Mode)

```bash
$ apr debug whisper.apr --drama

====[ DRAMA: whisper.apr ]====

ACT I: THE HEADER
  Scene 1: Magic bytes... APRN (applause!)
  Scene 2: Version check... 1.0 (standing ovation!)

ACT II: THE METADATA
  Scene 1: Parameters... 39,000,000 (a cast of millions!)

ACT III: THE VERDICT
  CURTAIN CALL: Model is PRODUCTION READY!
```

Options: `--hex`, `--strings`, `--limit`

### 4.4 Validate Command

```bash
$ apr validate model.apr --quality

=== 100-Point Quality Assessment ===

Structure (25 pts):     24/25
Security (25 pts):      20/25
Weights (25 pts):       25/25
Metadata (25 pts):      22/25

TOTAL: 91/100 (EXCELLENT)
```

### 4.5 Diff Command

```bash
$ apr diff model_v1.apr model_v2.apr

Similarity: 94.2%
Weight Changes: Max delta 0.0234, L2 distance 1.234
Vocab Changes: Added 42 tokens, Removed 3 tokens
```

### 4.6 Export Command

| Format | Extension | Use Case |
|--------|-----------|----------|
| ONNX | `.onnx` | Cross-framework inference |
| SafeTensors | `.safetensors` | HuggingFace ecosystem |
| GGUF | `.gguf` | llama.cpp / local inference |
| TorchScript | `.pt` | PyTorch deployment |

```bash
apr export model.apr --format gguf --quantize q4_0 --output model.gguf
```

### 4.7 Import Command

```bash
apr import hf://openai/whisper-tiny --output whisper.apr
apr import model.safetensors --from safetensors --output model.apr
```

### 4.8 Convert Command

```bash
apr convert model.apr --quantize q8_0 --output model_q8.apr
apr convert model.apr --precision fp16 --output model_fp16.apr
```

### 4.9 Merge Command

| Strategy | Description |
|----------|-------------|
| `average` | Average weights (ensemble) |
| `weighted` | Weighted average by performance |
| `ties` | TIES merging (trim, elect, sign) |
| `dare` | DARE merging (drop and rescale) |
| `slerp` | Spherical linear interpolation |

```bash
apr merge model1.apr model2.apr --strategy ties --output merged.apr
```

### 4.10 Trace Command

```bash
$ apr trace model.apr --input sample.wav

Layer                          Time (ms)   Memory (MB)
encoder.conv1                      12.3         45.2
decoder.attention.0                15.4         12.3
TOTAL                             142.5        312.4
```

---

## 5. Auxiliary Data Patterns

### 5.1 JSON Metadata Pattern

```
[APR magic] → [metadata_len] → [JSON metadata] → [tensors] → [CRC32]
                                     ↑
                            Auxiliary data here
```

### 5.2 Common Auxiliary Data Types

#### Vocabulary (NLP)
```json
{"vocab": ["<pad>", "<unk>", "the", "..."], "vocab_size": 51865}
```

#### Mel Filterbank (Audio)
```json
{"mel_filterbank": [0.0, "..."], "mel_filterbank_shape": [80, 201]}
```

#### Tokenizer Config
```json
{"tokenizer_config": {"type": "bpe", "unk_token": "<|unk|>", "eos_token": "<|endoftext|>"}}
```

#### Image Preprocessing (Vision)
```json
{"image_config": {"image_size": 224, "mean": [0.485, 0.456, 0.406]}}
```

#### Label Mapping (Classification)
```json
{"labels": {"0": "cat", "1": "dog"}, "num_labels": 2}
```

### 5.3 Tensor Storage for Large Data

| Data Size | JSON Metadata | Tensor |
|-----------|---------------|--------|
| < 100KB | Preferred | Overkill |
| 100KB - 1MB | Acceptable | Good |
| > 1MB | Avoid | Preferred |

Naming convention: `audio.mel_filterbank`, `text.token_embedding`

### 5.4 Best Practices

1. **Use standard keys**: Follow HuggingFace/GGUF conventions
2. **Include shape info**: Always store shape alongside flattened arrays
3. **Version metadata**: Include `format_version` for compatibility
4. **Document units**: Specify if values are normalized, in Hz, etc.
5. **Validate on load**: Check array lengths match expected shapes

---

## 6. Format Comparison

| Feature | APR1 | APR2 | GGUF | SafeTensors |
|---------|------|------|------|-------------|
| WASM-first | Yes | Yes | No | Yes |
| Tensor alignment | No | Yes (64B) | Yes (32B) | Yes |
| Compression | No | LZ4 | No | No |
| Quantization | Metadata | Native | Native | No |
| Sharding | No | Yes | No | Yes |
| Streaming | No | Yes | No | No |
| JSON metadata | Yes | Yes | Typed KV | JSON |
| CRC32 | Yes | Yes | No | No |

---

## 7. Error Handling

| Code | Category | Description |
|------|----------|-------------|
| E001 | FORMAT | Invalid file format |
| E002 | CORRUPT | Corrupted data |
| E003 | VERSION | Unsupported version |
| E004 | CHECKSUM | Checksum mismatch |
| E005 | DECRYPT | Decryption failed |
| E006 | SIGNATURE | Signature invalid |
| E007 | IO | File I/O error |
| E008 | MEMORY | Out of memory |

---

## 8. Configuration

```toml
# ~/.config/apr/config.toml

[defaults]
output_format = "text"
color = true

[inspect]
show_vocab = true
max_tokens_display = 20

[debug]
drama_mode = false
hex_limit = 256

[validate]
strict = true
require_signature = false
```

---

## 9. Quality Gates

```toml
# .pmat-gates.toml
[apr-ops]
test_coverage_minimum = 95.0
max_cyclomatic_complexity = 10
satd_maximum = 0
mutation_score_minimum = 85.0
max_inspect_latency_ms = 100
```

---

## 10. Falsification QA Checklist

### Header & Format (5 points)
| # | Claim | Falsification |
|---|-------|---------------|
| 1 | Magic bytes are always "APRN" | Find valid .apr without APRN magic |
| 2 | Version (1,0) is always supported | Find v1.0 file that fails to load |
| 3 | Header is exactly 32 bytes | Find valid .apr with different header size |
| 4 | Checksum detects single-bit errors | Find bit flip that passes checksum |
| 5 | Compressed size <= uncompressed size | Find file where compressed > uncompressed |

### Inspection (5 points)
| # | Claim | Falsification |
|---|-------|---------------|
| 6 | Inspect never loads full payload | Find inspect that loads full weights |
| 7 | Vocab size matches actual tokens | Find vocab size mismatch |
| 8 | All metadata fields are optional | Find required metadata field |
| 9 | JSON output is valid JSON | Find JSON output that fails parsing |
| 10 | Inspect completes in < 100ms | Find model where inspect > 100ms |

### Export/Import (5 points)
| # | Claim | Falsification |
|---|-------|---------------|
| 16 | Export→Import roundtrip preserves weights | Find weight drift > 1e-6 |
| 17 | ONNX export produces valid ONNX | Find invalid ONNX output |
| 18 | SafeTensors export is pickle-free | Find pickle in safetensors |
| 19 | Import rejects malicious pickles | Find accepted malicious pickle |
| 20 | Quantization reduces file size | Find quantization that increases size |

### Merge/Diff (5 points)
| # | Claim | Falsification |
|---|-------|---------------|
| 21 | Diff of identical models shows 100% | Find identical models with < 100% |
| 22 | Average merge is commutative | Find non-commutative merge |
| 23 | Merge preserves model type | Find type change after merge |
| 24 | Vocab merge union contains both | Find missing token in union |
| 25 | Weighted merge [1.0, 0.0] equals first | Find difference with weight=1.0 |

---

## 11. Implementation Roadmap

### Phase 1: Alignment (v2.0)
- 64-byte tensor alignment
- Binary tensor index
- Backward-compatible reader

### Phase 2: Compression (v2.1)
- LZ4 block compression
- Per-tensor compression flag
- Streaming decompression

### Phase 3: Sharding (v2.2)
- Manifest file format
- Multi-file loader
- Tensor-level demand loading

---

## 12. References

1. Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*
2. Amershi, S., et al. (2019). "Software Engineering for Machine Learning." *ICSE 2019*
3. Vartak, M., et al. (2016). "ModelDB: A System for ML Model Management." *SIGMOD 2016*
4. Baylor, D., et al. (2017). "TFX: A TensorFlow-Based Production-Scale ML Platform." *KDD 2017*
5. Zaharia, M., et al. (2018). "Accelerating the ML Lifecycle with MLflow." *IEEE Data Eng. Bull.*

**Code References:**
- APR v1: `src/serialization/apr.rs`
- GGUF: `src/format/gguf.rs`
- Bundle system: `src/bundle/`
- SafeTensors: `src/serialization/safetensors.rs`

---

## 13. Appendices

### A. Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Format error |
| 5 | Validation failed |

### B. Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APR_CONFIG` | Config file path | `~/.config/apr/config.toml` |
| `APR_CACHE` | Cache directory | `~/.cache/apr` |
| `APR_LOG_LEVEL` | Log level | `info` |
| `APR_COLOR` | Enable colors | `auto` |

---

*Document generated following Toyota Way principles and PMAT quality standards.*
