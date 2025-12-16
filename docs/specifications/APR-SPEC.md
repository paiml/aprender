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
   - [4.11 Lint Command](#411-lint-command)
   - [4.12 Explain Command](#412-explain-command)
   - [4.13 TUI Command](#413-tui-command)
   - [4.14 Canary Command](#414-canary-command)
   - [4.15 Run Command](#415-run-command)
   - [4.16 Compile Command](#416-compile-command)
5. [Auxiliary Data Patterns](#5-auxiliary-data-patterns)
   - [5.1 JSON Metadata Pattern](#51-json-metadata-pattern)
   - [5.2 Common Auxiliary Data Types](#52-common-auxiliary-data-types)
   - [5.3 Tensor Storage for Large Data](#53-tensor-storage-for-large-data)
   - [5.4 Best Practices](#54-best-practices)
6. [Format Comparison](#6-format-comparison)
7. [Error Handling](#7-error-handling)
8. [Configuration](#8-configuration)
9. [Quality Gates](#9-quality-gates)
10. [Multi-Format Conversion Specification](#10-multi-format-conversion-specification)
    - [10.1 Supported Input Formats](#101-supported-input-formats)
    - [10.2 SafeTensors (HuggingFace)](#102-safetensors-huggingface)
    - [10.3 PyTorch (.pt, .pth, .bin)](#103-pytorch-pt-pth-bin)
    - [10.4 GGUF (llama.cpp)](#104-gguf-llamacpp)
    - [10.5 GGML (Legacy)](#105-ggml-legacy)
    - [10.6 ONNX](#106-onnx)
    - [10.7 TensorFlow/Keras](#107-tensorflowkeras)
    - [10.8 Tensor Name Mapping](#108-tensor-name-mapping)
    - [10.9 Expected Tensor Statistics](#109-expected-tensor-statistics)
    - [10.10 Conversion Validation Requirements](#1010-conversion-validation-requirements)
    - [10.11 Known Failure Modes](#1011-known-failure-modes)
11. [Conversion QA Checklist (25 Points)](#11-conversion-qa-checklist-25-points)
    - [A. Structural Integrity](#a-structural-integrity-5-points)
    - [B. Layer Norm Validation](#b-layer-norm-validation-5-points)
    - [C. Attention/Linear Validation](#c-attentionlinear-validation-5-points)
    - [D. Embedding Validation](#d-embedding-validation-5-points)
    - [E. Functional Validation](#e-functional-validation-5-points)
12. [Automated Conversion Validation](#12-automated-conversion-validation)
13. [Falsification QA Checklist (Legacy)](#13-falsification-qa-checklist-legacy)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [References](#15-references)
16. [Appendices](#16-appendices)

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
    run         Run model directly (auto-download, cache, execute)
    compile     Build standalone executable with embedded model
    inspect     Inspect model metadata, vocab, and structure
    debug       Simple debugging output ("drama" mode)
    validate    Validate model integrity
    diff        Compare two models
    tensors     List tensor information
    export      Export model to other formats (SafeTensors, GGUF)
    import      Import from external formats (HuggingFace, SafeTensors)
    convert     Convert/optimize model (quantization: int8, int4, fp16)
    merge       Merge multiple models (average, weighted strategies)
    trace       Trace model operations with renacer
    lint        Check for best practices and conventions
    explain     Explain errors, architecture, and tensors
    canary      Regression testing via tensor statistics
    tui         Interactive terminal UI for exploration
    probar      Export for visual testing
    tasks       List available tasks from apr.toml
    lock        Generate/verify apr.lock for reproducibility

EXAMPLES:
    apr run hf://openai/whisper-tiny --input audio.wav
    apr run whisper.apr < audio.wav > transcript.txt
    apr compile whisper.apr --quantize int8 -o whisper-cli
    apr transcribe recording.wav    # Run task from apr.toml
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

### 4.2.1 Visual Inspection
For suspect tensors, generate an in-terminal histogram to visualize distributions (e.g., detecting shifted means):

```bash
$ apr tensors model.apr --hist encoder.layer_norm.weight

Distribution: encoder.layer_norm.weight (shape: [384])
Min: 10.4  Max: 12.1  Mean: 11.2  Std: 0.2

       |          *
       |         ***
  50%  |        *****
       |       *******
       |      *********
       +------------------
       10.0      11.2      12.5
```

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

#### Diff vs Reference
Compare an APR model against a raw `.safetensors` reference to detect translation drift:

```bash
$ apr diff model.apr source.safetensors --tensor-mapping mapping.json

# Output:
# encoder.conv1.weight: MATCH (delta < 1e-6)
# encoder.layer_norm.weight: DRIFT (delta = 10.2) !!!
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

Model optimization and size reduction operations.

```bash
apr convert model.apr --quantize q8_0 --output model_q8.apr
apr convert model.apr --precision fp16 --output model_fp16.apr
```

#### 4.8.1 Size Reduction Techniques

| Technique | Flag | Reduction | Quality | Reversible |
|-----------|------|-----------|---------|------------|
| **Quantization** | `--quantize` | 2-8x | Low loss | No |
| **Compression** | `--compress` | 1.2-2x | Lossless | Yes |
| **Pruning** | `--prune` | 2-10x | Medium | No |
| **Distillation** | `--distill` | 2-10x | Medium | No |
| **Low-rank (SVD)** | `--lowrank` | 2-4x | Low loss | No |
| **Sparsity** | `--sparse` | 2-5x | Low loss | Yes |

##### Quantization

Reduce precision of weights:

```bash
# Integer quantization
apr convert model.apr --quantize int8 -o model-int8.apr      # 4x smaller
apr convert model.apr --quantize int4 -o model-int4.apr      # 8x smaller

# Float quantization
apr convert model.apr --quantize fp16 -o model-fp16.apr      # 2x smaller
apr convert model.apr --quantize bf16 -o model-bf16.apr      # 2x smaller

# GGUF-style quantization
apr convert model.apr --quantize q4_k_m -o model-q4km.apr    # 4.5 bits/weight
apr convert model.apr --quantize q8_0 -o model-q8.apr        # 8 bits/weight
```

##### Compression

Lossless compression of tensor data:

```bash
# LZ4 (fast, default)
apr convert model.apr --compress lz4 -o model-lz4.apr

# Zstd (better ratio)
apr convert model.apr --compress zstd -o model-zstd.apr
apr convert model.apr --compress zstd:19 -o model-zstd19.apr  # Max compression

# Combine with quantization
apr convert model.apr --quantize int8 --compress zstd -o model-int8-zstd.apr
```

##### Pruning

Remove low-magnitude weights:

```bash
# Unstructured pruning (sparse tensors)
apr convert model.apr --prune 0.5 -o model-pruned.apr        # 50% sparsity

# Structured pruning (remove entire neurons/heads)
apr convert model.apr --prune-heads 2 -o model-pruned.apr    # Remove 2 attention heads
apr convert model.apr --prune-layers 1 -o model-pruned.apr   # Remove 1 layer

# Magnitude-based with threshold
apr convert model.apr --prune-threshold 0.01 -o model-pruned.apr
```

##### Distillation

Train smaller model from larger (requires reference data):

```bash
# Distill to smaller architecture
apr convert model-large.apr --distill tiny --data train.jsonl -o model-tiny.apr

# Layer reduction
apr convert model.apr --distill-layers 4 --data train.jsonl -o model-4layer.apr

# Knowledge distillation with temperature
apr convert model.apr --distill small --temperature 2.0 --data train.jsonl -o model-small.apr
```

**Note**: Distillation requires training data and compute. Use `--epochs` and `--lr` to control.

##### Low-Rank Factorization

Decompose weight matrices using SVD/LoRA:

```bash
# SVD decomposition
apr convert model.apr --lowrank svd --rank 64 -o model-svd.apr

# LoRA-style decomposition
apr convert model.apr --lowrank lora --rank 16 -o model-lora.apr

# Target specific layers
apr convert model.apr --lowrank svd --rank 32 --target "*.fc1.weight" -o model-svd.apr
```

##### Sparsity Encoding

Efficient storage for sparse tensors:

```bash
# CSR format for sparse tensors
apr convert model.apr --sparse csr --threshold 0.001 -o model-sparse.apr

# Block sparsity (GPU-friendly)
apr convert model.apr --sparse block:4 -o model-block-sparse.apr
```

#### 4.8.2 Combination Examples

```bash
# Maximum compression pipeline
apr convert model.apr \
  --quantize int4 \
  --prune 0.3 \
  --compress zstd:19 \
  -o model-optimized.apr
# Result: ~20x smaller than original

# WASM-optimized (fast decode, small size)
apr convert model.apr \
  --quantize int8 \
  --compress lz4 \
  -o model-wasm.apr
# Result: ~5x smaller, fast streaming decode

# Quality-preserving compression
apr convert model.apr \
  --quantize fp16 \
  --lowrank svd --rank 128 \
  --compress zstd \
  -o model-quality.apr
# Result: ~3x smaller, minimal quality loss
```

#### 4.8.3 Size Comparison Table

| Technique | Whisper Tiny | Whisper Base | LLaMA 7B |
|-----------|--------------|--------------|----------|
| Original (f32) | 145 MB | 290 MB | 26 GB |
| fp16 | 73 MB | 145 MB | 13 GB |
| int8 | 37 MB | 73 MB | 6.5 GB |
| int4 | 19 MB | 37 MB | 3.3 GB |
| int4 + zstd | 15 MB | 29 MB | 2.6 GB |
| int4 + prune50% | 10 MB | 19 MB | 1.7 GB |

#### 4.8.4 Quality Validation (Pre vs Post)

Compare model quality before and after optimization:

```bash
# Compare outputs between original and optimized
apr validate model.apr model-optimized.apr --quality

Quality Comparison: model.apr vs model-optimized.apr
═══════════════════════════════════════════════════════════════
                          Original    Optimized    Δ
Tensor count              167         167          0
Total params              39.0M       39.0M        0
Non-zero params           39.0M       19.5M        -50%
Size                      145 MB      15 MB        -89%

Output Comparison (10 test inputs):
  Mean L2 distance:       0.0234      (threshold: 0.1)  ✓ PASS
  Max L2 distance:        0.0891      (threshold: 0.5)  ✓ PASS
  Cosine similarity:      0.9987      (threshold: 0.99) ✓ PASS

Layer-by-layer drift:
  encoder.conv1:          0.001       ✓
  encoder.layer_norm:     0.002       ✓
  decoder.layer_norm:     0.089       ⚠ (highest drift)

VERDICT: ✓ PASS - Optimized model within quality tolerance
═══════════════════════════════════════════════════════════════
```

##### Canary Inputs

Define reference inputs with expected outputs for regression testing:

```bash
# Create canary test suite
apr canary create model.apr --input test.wav --output canary.json

# Validate optimized model against canary
apr canary check model-optimized.apr --canary canary.json

Canary Test Results:
  Input: test.wav
  Expected: "The quick brown fox jumps over the lazy dog"
  Original:  "The quick brown fox jumps over the lazy dog"  ✓
  Optimized: "The quick brown fox jumps over the lazy dog"  ✓

  Token-level accuracy: 100%
  Character error rate: 0.0%
```

##### Automatic Quality Gates

```bash
# Fail optimization if quality degrades beyond threshold
apr convert model.apr --quantize int4 --prune 0.5 \
  --quality-check \
  --max-drift 0.1 \
  --canary canary.json \
  -o model-optimized.apr

# If quality check fails:
# ERROR: Quality gate failed
#   - L2 drift: 0.24 (max: 0.1)
#   - Canary "test.wav" failed: expected "fox" got "box"
# Use --force to ignore quality gates
```

#### 4.8.5 Payload Tracing (Radioactive Tracer)

Trace a payload through the model step-by-step, like a radioactive tracer in medicine:

```bash
apr trace model.apr --input test.wav --trace-payload

Payload Trace: test.wav → model.apr
═══════════════════════════════════════════════════════════════

Step 1: Audio Input
  Shape: [1, 480000]  (30s @ 16kHz)
  Stats: mean=0.002, std=0.15, range=[-0.98, 0.97]

Step 2: Mel Spectrogram
  Shape: [1, 80, 3000]
  Stats: mean=-4.2, std=2.1
  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁  (frequency distribution)

Step 3: encoder.conv1
  Shape: [1, 384, 3000]
  Stats: mean=0.12, std=0.34
  Time: 2.3ms
  ⚠ Activation spike at position 1247 (value: 12.4)

Step 4: encoder.conv2
  Shape: [1, 384, 1500]
  Stats: mean=0.08, std=0.29
  Time: 1.8ms

Step 5: encoder.positional_embedding
  Shape: [1, 1500, 384]
  Stats: mean=0.08, std=0.31

Step 6: encoder.layers.0.self_attn
  Shape: [1, 1500, 384]
  Attention pattern:
  ░░░░░░░░░░░░░░░░░░░░
  ░░░░████░░░░░░░░░░░░  ← attending to positions 40-80
  ░░░░░░░░░░░░████░░░░

  ... (layers 1-3) ...

Step 10: encoder.layer_norm
  Shape: [1, 1500, 384]
  Stats: mean=0.00, std=1.02  ✓ (properly normalized)

Step 11: decoder.token_embedding (SOT token)
  Shape: [1, 1, 384]
  Token: <|startoftranscript|> (50258)

  ... (decoder steps) ...

Step 47: Output Logits
  Shape: [1, 12, 51865]
  Top predictions:
    1. "The" (0.94)
    2. "A" (0.03)
    3. "This" (0.01)

═══════════════════════════════════════════════════════════════
Total time: 142ms | Peak memory: 312MB | Tokens generated: 12
```

##### Comparing Traces (Diff Mode)

Compare payload path between two models:

```bash
apr trace model.apr model-optimized.apr --input test.wav --diff

Trace Diff: model.apr vs model-optimized.apr
═══════════════════════════════════════════════════════════════

Step    Layer                    Original     Optimized    Drift
─────   ─────                    ────────     ─────────    ─────
1       audio_input              ████████     ████████     0.000
2       mel_spectrogram          ████████     ████████     0.000
3       encoder.conv1            ████████     ███████░     0.012
4       encoder.conv2            ████████     ███████░     0.018
...
10      encoder.layer_norm       ████████     ██████░░     0.089 ⚠
11      decoder.token_embed      ████████     ████████     0.001
...
47      output_logits            ████████     ███████░     0.023

Divergence detected at: encoder.layer_norm (step 10)
  Original mean:  0.0023
  Optimized mean: 0.0892

Recommendation: Check layer norm weight quantization
```

##### Anomaly Detection

Automatically detect unusual activations:

```bash
apr trace model.apr --input test.wav --detect-anomalies

Anomaly Report:
═══════════════════════════════════════════════════════════════

⚠ ANOMALY at encoder.layers.2.self_attn (step 8)
  - Activation explosion: max=847.3 (expected <10)
  - Possible cause: NaN propagation or weight corruption
  - Affected tokens: positions 120-135

⚠ ANOMALY at decoder.layer_norm (step 15)
  - Dead neurons: 12% of outputs are exactly 0
  - Possible cause: Aggressive pruning or ReLU saturation

✓ No anomalies in remaining 45 layers
```

##### Interactive Trace Mode (TUI)

```bash
apr trace model.apr --input test.wav --interactive
```

```
┌─────────────────────────────────────────────────────────────────┐
│  Payload Trace: test.wav                        [Interactive]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Pipeline ───────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  [Audio] ──▶ [Mel] ──▶ [Conv1] ──▶ [Conv2] ──▶ ...      │  │
│  │     ✓         ✓         ✓          ✓                     │  │
│  │                                    ▲                      │  │
│  │                                    │ YOU ARE HERE         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Current Layer: encoder.conv2 ───────────────────────────┐  │
│  │ Input:  [1, 384, 3000]   Output: [1, 384, 1500]          │  │
│  │ Params: 589,824          Time: 1.8ms                     │  │
│  │                                                          │  │
│  │ Activation Distribution:                                 │  │
│  │     ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁                                      │  │
│  │   -2.0            0            2.0                       │  │
│  │                                                          │  │
│  │ Weight Stats: mean=0.002, std=0.04                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Payload Snapshot ───────────────────────────────────────┐  │
│  │ [0.12, 0.34, -0.21, 0.08, 0.45, -0.11, 0.02, ...]       │  │
│  │ mean=0.08  std=0.29  min=-1.2  max=2.1                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [←/→] step  [Enter] inspect  [d]iff  [e]xport  [q]uit   4/47   │
└─────────────────────────────────────────────────────────────────┘
```

##### Export Trace for Analysis

```bash
# Export full trace to JSON
apr trace model.apr --input test.wav --export trace.json

# Export to Chrome trace format (for chrome://tracing)
apr trace model.apr --input test.wav --export trace.perfetto

# Export intermediate activations for debugging
apr trace model.apr --input test.wav --dump-activations ./activations/
```

#### 4.8.6 Debugging Conversion

```bash
# Analyze source tensor stats without converting
apr convert model.safetensors --analyze-source --arch whisper

# Output:
# [PASS] encoder.conv1.weight: mean=0.003 (expected ~0.0)
# [FAIL] encoder.layer_norm.weight: mean=11.2 (expected ~1.0) -> SOURCE ALREADY CORRUPT?
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

### 4.11 Lint Command

Static analysis for best practices, conventions, and "soft" requirements. Unlike `validate` (which checks for corruption/invalidity), `lint` checks for *quality* and *standardization*.

```bash
$ apr lint model.apr

[WARN] Metadata: Missing 'license' field
[WARN] Metadata: Missing 'model_card'
[INFO] Tensor Naming: 'encoder.w' should be 'encoder.weight' for auto-mapping
[INFO] Efficiency: 12 tensors could be aligned to 64 bytes (currently 32)
```

**Falsifiable Guarantees (Must Fail If):**
- **Naming**: Any tensor name not matching canonical schema (Section 10.8) raises INFO/WARN.
- **Metadata**: Missing `license`, `model_card`, or `provenance` raises WARN.
- **Efficiency**: Tensors unaligned to 64 bytes raise INFO.
- **Compression**: Uncompressed tensors >1MB raise INFO.

### 4.12 Explain Command

Provides human-readable context, architectural explanations, and error troubleshooting.

#### Explain Model Architecture
```bash
$ apr explain model.apr

This is a **Whisper (Tiny)** model.
- **Purpose**: Automatic Speech Recognition (ASR)
- **Architecture**: Encoder-Decoder Transformer
- **Input**: 80-channel Mel spectrograms
- **Output**: Text tokens (multilingual)
```

#### Explain Specific Tensor
```bash
$ apr explain model.apr --tensor encoder.conv1.weight

**encoder.conv1.weight**
- **Role**: Initial feature extraction (Audio -> Latent)
- **Shape**: [384, 80, 3] (Filters, Input Channels, Kernel Size)
- **Stats**: Mean 0.002, Std 0.04 (Healthy)
```

#### Explain Error Codes
```bash
$ apr explain E002

**E002: Corrupted Data**
The payload checksum does not match the header.
- **Common Causes**: Interrupted download, bit rot, disk error.
- **Troubleshooting**:
  1. Run `apr validate --checksum` to verify.
  2. Check source file integrity (MD5/SHA256).
```

**Falsifiable Guarantees:**
- **Unknown Error**: `apr explain E999` must return "Unknown Error Code" (not crash).
- **Unknown Tensor**: `apr explain --tensor nonexistent` must list fuzzy matches.
- **Architecture**: Must correctly identify all supported architectures (Section 10).

### 4.13 TUI Command

Interactive terminal UI for model exploration, statistics visualization, and comparison. Built with `ratatui` and `trueno-viz`.

```bash
$ apr tui model.apr
$ apr tui model1.apr model2.apr --compare
```

#### 4.13.1 Graph View

ASCII/Unicode graph visualization of model architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│  Model: whisper-tiny.apr                          [Graph View]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │  Audio  │───▶│  Conv1  │───▶│  Conv2  │                    │
│   │ [80,3000]│    │[384,80,3]│   │[384,384]│                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                                      │                          │
│                                      ▼                          │
│   ┌──────────────────────────────────────────────────────┐     │
│   │              Encoder Layers (×4)                      │     │
│   │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   │     │
│   │  │Self-Attn│──▶│  LN   │──▶│  FFN   │──▶│  LN    │   │     │
│   │  └────────┘   └────────┘   └────────┘   └────────┘   │     │
│   └──────────────────────────────────────────────────────┘     │
│                           │                                     │
│                           ▼                                     │
│   ┌──────────────────────────────────────────────────────┐     │
│   │              Decoder Layers (×4)                      │     │
│   │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   │     │
│   │  │Self-Attn│──▶│Cross-Attn│─▶│  FFN   │──▶│  LN    │   │     │
│   │  └────────┘   └────────┘   └────────┘   └────────┘   │     │
│   └──────────────────────────────────────────────────────┘     │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │   Output    │                              │
│                    │  [51865]    │                              │
│                    └─────────────┘                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [g]raph [s]tats [c]ompare [t]ensors [h]ist [q]uit    Page 1/3  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.13.2 Descriptive Statistics View

Live-updating tensor statistics dashboard:

```
┌─────────────────────────────────────────────────────────────────┐
│  Model: whisper-tiny.apr                          [Stats View]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Overview ───────────────────────────────────────────────┐  │
│  │ Total Params: 39,000,000    Tensors: 167    Size: 145MB  │  │
│  │ Quantization: f32           Vocab: 51,865   Arch: Whisper│  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Layer Norm Health ──────────────────────────────────────┐  │
│  │ Tensor                        Mean    Std    Status      │  │
│  │ encoder.layer_norm.weight     1.48    0.32   ✓ OK        │  │
│  │ decoder.layer_norm.weight    11.10    0.21   ✗ BAD       │  │
│  │ encoder.layers.0.ln.weight    1.22    0.28   ✓ OK        │  │
│  │ encoder.layers.1.ln.weight    1.35    0.31   ✓ OK        │  │
│  │ encoder.layers.2.ln.weight    1.41    0.29   ✓ OK        │  │
│  │ encoder.layers.3.ln.weight   10.94    0.18   ✗ BAD       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Weight Distribution ────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Attention:  ████████████████████  Mean: 0.002  ✓        │  │
│  │  FFN:        ███████████████████   Mean: 0.001  ✓        │  │
│  │  Embedding:  █████████████████     Mean: 0.015  ✓        │  │
│  │  LayerNorm:  ██████████████████████████████████  ✗       │  │
│  │              ↑ outlier: decoder.layer_norm.weight        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Validation Score ───────────────────────────────────────┐  │
│  │ ████████████████████░░░░  21/25 FAIL                     │  │
│  │ Critical: 2 Layer Norm weights outside [0.5, 3.0]        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [g]raph [s]tats [c]ompare [t]ensors [h]ist [q]uit    Page 1/1  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.13.3 Comparison View

Side-by-side model comparison with diff highlighting:

```
┌─────────────────────────────────────────────────────────────────┐
│  Comparing: model_v1.apr vs model_v2.apr         [Compare View] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Summary ────────────────────────────────────────────────┐  │
│  │ Similarity: 94.2%    Changed: 12 tensors    New: 0       │  │
│  │ Max Δ: 0.0234        L2 Dist: 1.234         Removed: 0   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Tensor Comparison ──────────────────────────────────────┐  │
│  │ Tensor                    v1 Mean   v2 Mean   Δ          │  │
│  │ encoder.conv1.weight      0.0023    0.0025    +0.0002    │  │
│  │ encoder.layer_norm.wt     1.4832    1.4901    +0.0069    │  │
│  │ decoder.layer_norm.wt    11.0983    1.0521   -10.0462 !! │  │
│  │ decoder.layers.0.fc1.wt   0.0012    0.0014    +0.0002    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Distribution Comparison ────────────────────────────────┐  │
│  │                                                          │  │
│  │  decoder.layer_norm.weight:                              │  │
│  │                                                          │  │
│  │  v1: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████  (mean=11.1)   │  │
│  │  v2: ░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░░  (mean=1.05)   │  │
│  │      ──────────────────────────────────────              │  │
│  │      0         5         10        15                    │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Validation Score Comparison ────────────────────────────┐  │
│  │ v1: ████████████████████░░░░  21/25 FAIL                 │  │
│  │ v2: ████████████████████████  25/25 PASS  ← IMPROVED     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [g]raph [s]tats [c]ompare [t]ensors [h]ist [q]uit    Page 1/1  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.13.4 Histogram View

Per-tensor distribution visualization with sparklines:

```
┌─────────────────────────────────────────────────────────────────┐
│  Tensor: decoder.layer_norm.weight               [Histogram]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Shape: [384]    dtype: f32    Size: 1.5 KB                    │
│  Mean: 11.0983   Std: 0.2134   Min: 10.42   Max: 12.01         │
│                                                                 │
│  Distribution:                                                  │
│                                                                 │
│   150 │                    ▄▄▄▄                                 │
│       │                  ▄██████▄                               │
│   100 │                ▄██████████▄                             │
│       │              ▄██████████████▄                           │
│    50 │            ▄██████████████████▄                         │
│       │          ▄██████████████████████▄                       │
│     0 ├──────────────────────────────────────────────           │
│       10.0      10.5      11.0      11.5      12.0              │
│                                                                 │
│  ⚠ ANOMALY DETECTED:                                           │
│  Expected mean ≈ 1.0 for LayerNorm weight                       │
│  Actual mean = 11.0983 (10x higher than expected)               │
│                                                                 │
│  Possible causes:                                               │
│  • Incorrect tensor scaling during conversion                   │
│  • Wrong tensor mapped to this name                             │
│  • Source model corruption                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [←/→] prev/next tensor  [Enter] select  [q] back    12/167     │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.13.5 Keybindings

| Key | Action |
|-----|--------|
| `g` | Switch to Graph view |
| `s` | Switch to Stats view |
| `c` | Switch to Compare view (if 2 models) |
| `t` | Switch to Tensor list |
| `h` | Switch to Histogram view |
| `Enter` | Select/drill down |
| `Esc` | Back/cancel |
| `↑/↓` | Navigate list |
| `←/→` | Previous/next page or tensor |
| `/` | Search tensors |
| `?` | Help |
| `q` | Quit |

#### 4.13.6 Implementation

**Crates**:
- `ratatui = "0.28"` - Terminal UI framework
- `crossterm = "0.28"` - Cross-platform terminal handling
- `trueno-viz` - Tensor visualization utilities (optional)

**Feature Flag**:
```toml
[features]
tui = ["ratatui", "crossterm"]
```

### 4.14 Canary Command

Regression testing via tensor statistics. Creates reference snapshots of model weights and validates that optimized/converted models stay within tolerance.

#### 4.14.1 Create Canary

```bash
$ apr canary create model.apr --input reference.wav --output canary.json

=== APR Canary Create ===

Model: model.apr
Output: canary.json

Loading model...
Computing tensor statistics...

=== Canary Created ===
Tensors captured: 167
Output file: canary.json

Canary test created successfully
```

**Output Format (canary.json)**:
```json
{
  "model_name": "model.apr",
  "tensor_count": 167,
  "created_at": "2025-12-16T12:00:00Z",
  "tensors": {
    "encoder.conv1.weight": {
      "shape": [384, 80, 3],
      "count": 92160,
      "mean": 0.0023,
      "std": 0.0412,
      "min": -0.1234,
      "max": 0.1567
    }
  }
}
```

#### 4.14.2 Check Canary

```bash
$ apr canary check model-optimized.apr --canary canary.json

=== APR Canary Check ===

Model: model-optimized.apr
Canary: canary.json

Loading model...
Comparing tensors...

=== Canary Check Results ===

[PASS] encoder.conv1.weight
[PASS] encoder.conv2.weight
[FAIL] decoder.layer_norm.weight
       Mean drift 15.2% exceeds threshold 10.0%

Results: 165 passed, 2 failed out of 167 tensors

Canary check FAILED - model drifted beyond tolerance
```

#### 4.14.3 Falsifiable Guarantees

| Check | Threshold | Failure Condition |
|-------|-----------|-------------------|
| Mean drift | 10% | `abs(new_mean - old_mean) / old_mean > 0.10` |
| Std drift | 20% | `abs(new_std - old_std) / old_std > 0.20` |
| Shape match | exact | `new_shape != old_shape` |
| Tensor exists | required | Tensor missing from model |

#### 4.14.4 Integration with Convert

```bash
# Fail optimization if canary check fails
apr convert model.apr --quantize int4 \
  --canary canary.json \
  -o model-optimized.apr

# Output on failure:
# ERROR: Canary check failed
#   - decoder.layer_norm.weight: Mean drift 15.2% (max: 10.0%)
# Use --force to ignore canary check
```

### 4.15 Run Command

The `apr run` command provides a modern, zero-friction experience for running models directly. Inspired by contemporary CLI tools that prioritize developer ergonomics and "just works" behavior.

#### 4.15.1 Design Philosophy

Modern CLI tools have established new standards for user experience:

> "The best CLI tools get out of your way. They should do what you mean, not what you say." — **uv documentation** [1]

| Inspiration | Tool | Key Innovation |
|-------------|------|----------------|
| Zero-config execution | uv [1] | `uv run script.py` auto-resolves dependencies |
| Task-based workflows | just [2] | Simple task definitions without Makefile complexity |
| URL-based imports | Deno [3] | `deno run https://...` runs remote code directly |
| Content-addressed caching | Nix [4] | Reproducible builds via hash-based storage |
| Progressive disclosure | cargo [5] | Simple defaults, powerful options when needed |

#### 4.15.2 Basic Usage

```bash
# Run model directly - auto-downloads, caches, executes
apr run hf://openai/whisper-tiny --input audio.wav

# Run local model
apr run whisper.apr --input audio.wav

# Smart inference - detects input type from extension
apr run model.apr audio.wav

# Pipeline style (Unix philosophy)
apr run hf://openai/whisper-tiny < audio.wav > transcript.txt

# Interactive mode - REPL for audio input
apr run whisper.apr --interactive
> [Recording... speak now]
> "Hello world"
```

#### 4.15.3 Remote Model Resolution

Like `uv` and `deno`, `apr run` supports URL-based model references with automatic caching:

```bash
# HuggingFace Hub
apr run hf://openai/whisper-tiny

# Direct URL (like deno)
apr run https://models.example.com/whisper-tiny.apr

# Git repository
apr run git+https://github.com/org/models#whisper-tiny.apr

# Local path (explicit)
apr run ./models/whisper.apr
apr run /absolute/path/model.apr
```

**Cache Location**: `~/.apr/cache/` (content-addressed, like Nix)

```
~/.apr/cache/
├── hf/
│   └── openai/
│       └── whisper-tiny/
│           └── v1.0.0/
│               ├── model.apr
│               └── config.json
└── urls/
    └── sha256-abc123.../
        └── model.apr
```

#### 4.15.4 Task Configuration (`apr.toml`)

Inspired by `just` [2] and `pyproject.toml`, define reusable tasks:

```toml
# apr.toml - Project-level model configuration

[model]
source = "hf://openai/whisper-tiny"
quantize = "int8"
cache = true

[model.options]
language = "en"
task = "transcribe"

# Named tasks (like justfile recipes)
[tasks.transcribe]
description = "Transcribe audio files to text"
input = "*.wav"
output = "transcripts/{stem}.txt"

[tasks.translate]
description = "Translate audio to English"
input = "*.wav"
output = "translations/{stem}.txt"
options = { task = "translate", language = "en" }

[tasks.batch]
description = "Process all recordings"
input = "recordings/**/*.wav"
output = "output/{relative_dir}/{stem}.json"
parallel = true
```

**Usage**:
```bash
# Run named task
apr transcribe recording.wav
apr translate meeting.wav

# List available tasks
apr tasks
  transcribe  Transcribe audio files to text
  translate   Translate audio to English
  batch       Process all recordings

# Run with overrides
apr transcribe --language fr audio.wav
```

#### 4.15.5 Lockfile (`apr.lock`)

For reproducible model versions (inspired by `uv.lock` [1] and `Cargo.lock` [5]):

```toml
# apr.lock - Auto-generated, do not edit

[[models]]
name = "whisper-tiny"
source = "hf://openai/whisper-tiny"
version = "1.0.0"
revision = "abc123def456"
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
downloaded = "2025-12-16T12:00:00Z"

[[models.tensors]]
name = "encoder.conv1.weight"
shape = [384, 80, 3]
dtype = "f32"
checksum = "crc32:deadbeef"
```

**Commands**:
```bash
# Generate lockfile from apr.toml
apr lock

# Verify models match lockfile
apr lock --check

# Update to latest versions
apr lock --upgrade
```

#### 4.15.6 Environment Variables

```bash
# Cache directory (default: ~/.apr/cache)
APR_CACHE_DIR=/path/to/cache

# Offline mode (fail if model not cached)
APR_OFFLINE=1

# HuggingFace token for private models
HF_TOKEN=hf_xxx

# Parallelism for batch processing
APR_PARALLEL=4

# Quiet mode (no progress bars)
APR_QUIET=1
```

#### 4.15.7 Progress and Output

Modern progress indication (like cargo/uv):

```bash
$ apr run hf://openai/whisper-tiny --input audio.wav

Resolving hf://openai/whisper-tiny...
  Found: whisper-tiny v1.0.0 (145 MB)

Downloading ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 145/145 MB

Cached: ~/.apr/cache/hf/openai/whisper-tiny/v1.0.0/model.apr

Loading model... done (1.2s)
Processing audio.wav...

Output:
  "The quick brown fox jumps over the lazy dog."

Completed in 3.4s (RTF: 0.34x)
```

#### 4.15.8 Falsifiable Guarantees

| Behavior | Guarantee |
|----------|-----------|
| Cache hit | If `apr.lock` matches cached model, no network request |
| Offline mode | `APR_OFFLINE=1` must fail immediately if model not cached |
| Reproducibility | Same `apr.lock` + same input = identical output |
| Progress | Downloads >1MB must show progress bar |
| Exit codes | 0=success, 1=model error, 2=input error, 3=network error |

#### 4.15.9 References

[1] **uv** - "An extremely fast Python package installer and resolver, written in Rust."
    https://github.com/astral-sh/uv
    Key influence: Zero-config execution, lockfiles, content-addressed caching

[2] **just** - "A handy way to save and run project-specific commands."
    https://github.com/casey/just
    Key influence: Simple task definitions, recipe-based workflows

[3] **Deno** - "A modern runtime for JavaScript and TypeScript."
    https://deno.land/
    Key influence: URL-based imports, secure-by-default execution

[4] **Nix** - "Reproducible builds and deployments."
    https://nixos.org/
    Key influence: Content-addressed storage, reproducibility guarantees

[5] **Cargo** - "The Rust package manager."
    https://doc.rust-lang.org/cargo/
    Key influence: Progressive disclosure, lockfiles, workspace management

---

### 4.16 Compile Command

The `apr compile` command builds optimized standalone executables with embedded models. This enables zero-dependency deployment where the model weights are bundled directly into the binary.

#### 4.16.1 Design Philosophy

Many production use cases require deploying ML models as self-contained executables:

- **Edge deployment**: Single binary for IoT devices, embedded systems
- **Air-gapped environments**: No network access to download models
- **Distribution simplicity**: Ship one file, not model + runtime
- **Startup optimization**: No model loading overhead at runtime
- **IP protection**: Model weights embedded in binary (harder to extract)

> "The best deployment is no deployment. Just copy one file." — **Single-binary philosophy**

| Inspiration | Tool | Key Innovation |
|-------------|------|----------------|
| Static linking | Go [6] | Single static binary with no dependencies |
| Embedded resources | Rust `include_bytes!` | Compile-time asset embedding |
| AOT compilation | GraalVM Native Image [7] | Ahead-of-time compilation for instant startup |
| WASM components | wasm-tools [8] | Self-contained WASM modules |
| Tree-shaking | esbuild [9] | Dead code elimination for smaller bundles |

#### 4.16.2 Basic Usage

```bash
# Compile model into standalone executable
apr compile whisper.apr -o whisper-cli

# Compile with specific target
apr compile whisper.apr --target x86_64-unknown-linux-gnu -o whisper-cli

# Compile to WASM module
apr compile whisper.apr --target wasm32-unknown-unknown -o whisper.wasm

# Compile with quantization for smaller binary
apr compile whisper.apr --quantize int8 -o whisper-cli

# Compile with custom entrypoint
apr compile whisper.apr --entrypoint transcribe -o transcribe
```

#### 4.16.3 Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| Native executable | (default) | Platform-specific binary (ELF, Mach-O, PE) |
| WASM module | `--target wasm32-*` | Self-contained `.wasm` file |
| Static library | `--lib` | `.a`/`.lib` for embedding in other projects |
| Shared library | `--shared` | `.so`/`.dylib`/`.dll` |
| WASM component | `--component` | WASI component model format |

#### 4.16.4 Target Platforms

```bash
# List available targets
apr compile --list-targets

Targets:
  Native:
    x86_64-unknown-linux-gnu      Linux x86_64 (glibc)
    x86_64-unknown-linux-musl     Linux x86_64 (musl, fully static)
    aarch64-unknown-linux-gnu     Linux ARM64
    x86_64-apple-darwin           macOS x86_64
    aarch64-apple-darwin          macOS ARM64 (Apple Silicon)
    x86_64-pc-windows-msvc        Windows x86_64

  WebAssembly:
    wasm32-unknown-unknown        Pure WASM (browser)
    wasm32-wasi                   WASM + WASI (server-side)
    wasm32-wasip1                 WASM + WASI Preview 1
    wasm32-wasip2                 WASM + WASI Preview 2 (component model)

# Cross-compile for different platform
apr compile whisper.apr --target aarch64-unknown-linux-gnu -o whisper-arm64
```

#### 4.16.5 Size Optimization

Multiple strategies to minimize binary size:

```bash
# Aggressive optimization (slower compile, smaller binary)
apr compile whisper.apr --optimize size -o whisper-small

# Strip debug symbols
apr compile whisper.apr --strip -o whisper-stripped

# Quantize weights (significant size reduction)
apr compile whisper.apr --quantize int4 -o whisper-tiny
#   fp32: 145 MB → int8: 39 MB → int4: 20 MB

# Enable LTO (Link-Time Optimization)
apr compile whisper.apr --lto -o whisper-lto

# All optimizations combined
apr compile whisper.apr --release --quantize int8 --strip --lto -o whisper-prod
```

**Size breakdown example**:
```bash
$ apr compile whisper-tiny.apr --quantize int8 --strip --lto -o whisper

Binary composition:
  Model weights:    38.2 MB (94.1%)
  Runtime code:      2.1 MB (5.2%)
  Metadata:          0.3 MB (0.7%)
  ─────────────────────────────
  Total:            40.6 MB

Optimization savings:
  Original fp32:    145.0 MB
  After int8:        39.0 MB  (-73%)
  After strip:       38.5 MB  (-1%)
  After LTO:         38.2 MB  (-1%)
```

#### 4.16.6 Runtime Configuration

Compiled binaries accept runtime arguments:

```bash
# Generated CLI interface
$ ./whisper --help
whisper-cli 1.0.0
Compiled from: whisper-tiny.apr
Model: openai/whisper-tiny (39M parameters, int8)

USAGE:
    whisper [OPTIONS] <INPUT>

ARGS:
    <INPUT>    Audio file to transcribe

OPTIONS:
    -l, --language <LANG>     Language code (default: auto)
    -t, --task <TASK>         Task: transcribe|translate (default: transcribe)
    -o, --output <FILE>       Output file (default: stdout)
    -f, --format <FMT>        Output format: text|json|srt|vtt (default: text)
    --timestamps              Include word-level timestamps
    --no-gpu                  Disable GPU acceleration
    -v, --verbose             Verbose output
    -h, --help                Print help

# Example usage
$ ./whisper audio.wav
Hello world, this is a test.

$ ./whisper --format json --timestamps audio.wav > result.json
```

#### 4.16.7 Embedding Configuration

Control what gets embedded in the binary:

```toml
# apr-compile.toml - Build configuration

[model]
path = "whisper-tiny.apr"
quantize = "int8"

[embed]
# What to include in binary
vocabulary = true           # Required for tokenization
mel_filterbank = true       # Required for audio processing
language_tokens = ["en", "es", "fr"]  # Subset of supported languages

[exclude]
# Remove unused components
translation_head = true     # If only transcription needed
language_detection = false  # Keep this

[runtime]
# Runtime defaults (overridable via CLI)
default_language = "en"
default_task = "transcribe"
max_audio_length = 30       # Seconds

[binary]
name = "whisper"
version = "1.0.0"
entrypoint = "transcribe"   # Main function
```

#### 4.16.8 Build Pipeline

```bash
# Development build (fast compile, larger binary)
apr compile whisper.apr -o whisper-dev

# Release build (optimized)
apr compile whisper.apr --release -o whisper

# Production build (all optimizations + signing)
apr compile whisper.apr --release --sign key.pem -o whisper-prod

# Reproducible build (deterministic output)
apr compile whisper.apr --reproducible -o whisper
sha256sum whisper  # Always same hash for same inputs
```

#### 4.16.9 WASM-Specific Options

```bash
# Browser-optimized WASM
apr compile whisper.apr --target wasm32-unknown-unknown \
    --wasm-opt \
    --no-threads \
    -o whisper.wasm

# WASI with filesystem access
apr compile whisper.apr --target wasm32-wasi \
    --wasi-snapshot preview1 \
    -o whisper.wasm

# WASM component (composable)
apr compile whisper.apr --target wasm32-wasip2 \
    --component \
    --wit interface.wit \
    -o whisper.component.wasm
```

**Generated WASM interface**:
```typescript
// TypeScript bindings auto-generated
export interface Whisper {
  transcribe(audio: Float32Array): Promise<TranscriptionResult>;
  translate(audio: Float32Array, targetLang: string): Promise<TranslationResult>;
  detectLanguage(audio: Float32Array): Promise<LanguageDetection>;
}
```

#### 4.16.10 Library Mode

Generate libraries for embedding in other applications:

```bash
# Static library
apr compile whisper.apr --lib --target x86_64-unknown-linux-gnu -o libwhisper.a

# Shared library with C ABI
apr compile whisper.apr --shared --cabi -o libwhisper.so

# Generate header file
apr compile whisper.apr --header -o whisper.h
```

**Generated C header**:
```c
// whisper.h - Auto-generated C bindings
#ifndef WHISPER_H
#define WHISPER_H

#include <stdint.h>

typedef struct WhisperContext WhisperContext;

// Initialize (model weights already embedded)
WhisperContext* whisper_init(void);
void whisper_free(WhisperContext* ctx);

// Transcription
typedef struct {
    const char* text;
    float confidence;
    // ...
} WhisperResult;

WhisperResult* whisper_transcribe(
    WhisperContext* ctx,
    const float* audio,
    size_t audio_len
);

void whisper_result_free(WhisperResult* result);

#endif
```

#### 4.16.11 Runtime Acceleration (trueno Integration)

Compiled binaries automatically leverage optimal SIMD/GPU acceleration via **trueno** [10] — no configuration required.

**"Just Works" Philosophy**:

```rust
// Same code, optimal execution everywhere
let result = matrix.matmul(&weights);  // Auto-selects best backend
```

At runtime, trueno automatically detects and uses the best available backend:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Runtime Backend Selection                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CPU Feature Detection (x86_64):                                │
│    AVX-512 → AVX2 → AVX → SSE2 → Scalar                        │
│                                                                  │
│  CPU Feature Detection (ARM):                                   │
│    NEON (AArch64) → NEON (ARMv7) → Scalar                       │
│                                                                  │
│  GPU Detection (via wgpu):                                      │
│    Vulkan → Metal → DX12 → WebGPU → CPU fallback               │
│                                                                  │
│  WASM (browser):                                                │
│    SIMD128 (all modern browsers) → Scalar fallback             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Operation-Specific Dispatch**:

| Operation | CPU SIMD | GPU | Notes |
|-----------|----------|-----|-------|
| Matrix multiply | ✅ AVX2/NEON | ✅ Beneficial | GPU for 500×500+ |
| Dot product | ✅ 3-11x speedup | ❌ Overhead | CPU always faster |
| Element-wise | ✅ Memory-bound | ❌ Transfer cost | CPU preferred |
| Convolution | ✅ Small kernels | ✅ Large images | Threshold-based |
| Attention | ✅ Cache-blocked | ✅ Large batches | Adaptive |

**Compile-Time Options**:

```bash
# Force specific backend (override auto-selection)
apr compile whisper.apr --backend avx2 -o whisper-avx2
apr compile whisper.apr --backend gpu -o whisper-gpu

# Disable GPU (smaller binary, CPU only)
apr compile whisper.apr --no-gpu -o whisper-cpu-only

# WASM with SIMD128 (browser deployment)
apr compile whisper.apr --target wasm32-unknown-unknown -o whisper.wasm
# Note: WASM SIMD128 enabled by default, supported in all modern browsers
```

**Runtime Override**:

```bash
# Override backend at runtime (compiled binary)
./whisper --backend scalar audio.wav    # Force scalar (debugging)
./whisper --backend avx2 audio.wav      # Force AVX2
./whisper --no-gpu audio.wav            # Disable GPU acceleration

# Environment variable override
TRUENO_BACKEND=avx2 ./whisper audio.wav
TRUENO_DISABLE_GPU=1 ./whisper audio.wav
```

**Performance Characteristics**:

| Target | Backend | Typical Speedup | Memory |
|--------|---------|-----------------|--------|
| x86_64 server | AVX-512 | 8-16x vs scalar | Same |
| x86_64 laptop | AVX2 | 4-8x vs scalar | Same |
| Apple Silicon | NEON | 4-8x vs scalar | Same |
| GPU (discrete) | wgpu | 2-10x for matmul | +VRAM |
| WASM browser | SIMD128 | 2-4x vs scalar | Same |

**Zero Configuration Required**:

The "just works" guarantee means:
1. No `--enable-simd` flags needed
2. No runtime configuration files
3. No environment variables required
4. Graceful fallback if features unavailable
5. Identical results across all backends (bit-exact for deterministic ops)

#### 4.16.12 Falsifiable Guarantees

| Behavior | Guarantee |
|----------|-----------|
| Self-contained | Binary runs without external model files |
| Reproducible | `--reproducible` flag produces identical binary for identical inputs |
| Cross-compile | All listed targets produce working binaries |
| Size bounds | `--quantize int8` reduces model size by ≥70% |
| Startup time | Compiled binary starts in <100ms (model already loaded) |
| Runtime parity | Compiled binary produces identical output to `apr run` |
| SIMD auto-select | Correct backend selected without user configuration |
| GPU threshold | GPU only used when beneficial (>500×500 matrices) |

#### 4.16.13 References

[6] **Go static binaries** - "Go programs compile to a single binary with no external dependencies."
    https://go.dev/doc/install/source
    Key influence: Zero-dependency deployment philosophy

[7] **GraalVM Native Image** - "Ahead-of-time compilation for Java applications."
    https://www.graalvm.org/native-image/
    Key influence: AOT compilation for instant startup

[8] **wasm-tools** - "Low-level tooling for WebAssembly."
    https://github.com/bytecodealliance/wasm-tools
    Key influence: WASM component model, composability

[9] **esbuild** - "An extremely fast bundler for the web."
    https://esbuild.github.io/
    Key influence: Tree-shaking, dead code elimination, size optimization

[10] **trueno** - "Multi-target high-performance compute library."
    https://github.com/paiml/trueno
    Key influence: Runtime SIMD auto-selection (AVX-512/AVX2/NEON/WASM), GPU dispatch via wgpu,
    "just works" acceleration with zero configuration

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

## 10. Multi-Format Conversion Specification

### 10.1 Supported Input Formats

APR supports conversion from all major ML model formats:

| Format | Extensions | Source | Priority | Status |
|--------|------------|--------|----------|--------|
| **SafeTensors** | `.safetensors` | HuggingFace | P0 | ✅ Implemented |
| **PyTorch** | `.pt`, `.pth`, `.bin` | PyTorch | P0 | 🔲 Planned |
| **GGUF** | `.gguf` | llama.cpp | P1 | 🔲 Planned |
| **GGML** | `.bin` | Legacy llama.cpp | P2 | 🔲 Planned |
| **ONNX** | `.onnx` | ONNX Runtime | P1 | 🔲 Planned |
| **TensorFlow** | `.pb`, `.h5`, SavedModel | TensorFlow/Keras | P2 | 🔲 Planned |
| **Core ML** | `.mlmodel`, `.mlpackage` | Apple | P3 | 🔲 Future |
| **TensorRT** | `.engine`, `.plan` | NVIDIA | P3 | 🔲 Future |

**Critical Lesson Learned**: A single incorrect tensor conversion (e.g., `decoder.layer_norm.weight` with mean=11 instead of ~1) can cause complete model failure while passing basic structural checks.

---

### 10.2 SafeTensors (HuggingFace)

**Status**: ✅ Primary implementation

**File Structure**:
```
model.safetensors
├── Header (8 bytes): JSON length (u64 LE)
├── JSON Metadata: tensor names, shapes, dtypes, offsets
└── Tensor Data: contiguous f32/f16/bf16 arrays
```

**CLI Usage**:
```bash
apr convert model.safetensors -o model.apr
apr convert model.safetensors --quantize int8 -o model-int8.apr

# From HuggingFace Hub
apr convert hf://openai/whisper-tiny -o whisper-tiny.apr
```

**Data Types**:
| SafeTensors Type | APR Conversion |
|------------------|----------------|
| F32 | Direct copy |
| F16 | Convert to f32 or keep as f16 |
| BF16 | Convert to f32 |
| I8 | Keep as int8 (quantized) |

**Crate**: `safetensors = "0.4"`

---

### 10.3 PyTorch (.pt, .pth, .bin)

**Status**: 🔲 Planned (P0)

**File Structure**:
```
model.pt (ZIP archive)
├── data.pkl          # Python pickle with tensor metadata
├── data/0            # Raw tensor bytes
├── data/1
└── ...
```

**Security Warning**: PyTorch files use Python pickle, which can execute arbitrary code. APR conversion MUST:
1. Use `pickle` in restricted mode (no arbitrary imports)
2. Validate tensor shapes before allocation
3. Reject files with suspicious pickle opcodes

**CLI Usage**:
```bash
apr convert model.pt -o model.apr --arch whisper
apr convert model.pth -o model.apr --arch llama

# With state_dict key prefix
apr convert model.pt -o model.apr --prefix "model."
```

**Implementation Notes**:
- Use `zip` crate for archive extraction
- Implement minimal pickle parser (BINGET, MARK, TUPLE, etc.)
- Map `torch.float32` → f32, `torch.float16` → f16
- Handle both full checkpoints and state_dict-only files

**Crate**: Custom pickle parser (no Python dependency)

---

### 10.4 GGUF (llama.cpp)

**Status**: 🔲 Planned (P1)

**File Structure**:
```
model.gguf
├── Magic (4 bytes): "GGUF"
├── Version (4 bytes): u32
├── Tensor Count (8 bytes): u64
├── Metadata KV Count (8 bytes): u64
├── Metadata KV Pairs: typed key-value store
├── Tensor Infos: name, dims, type, offset
└── Tensor Data: aligned, possibly quantized
```

**CLI Usage**:
```bash
apr convert model.gguf -o model.apr
apr convert model-q4_k_m.gguf -o model.apr --dequantize f32
apr convert model.gguf -o model.apr --keep-quantization
```

**Quantization Types**:
| GGUF Type | Bits | APR Handling |
|-----------|------|--------------|
| F32 | 32 | Direct copy |
| F16 | 16 | Convert or keep |
| Q8_0 | 8 | Dequantize or convert to APR int8 |
| Q4_0 | 4 | Dequantize to f32 |
| Q4_K_M | 4.5 | Dequantize to f32 |
| Q5_K_M | 5.5 | Dequantize to f32 |
| Q6_K | 6 | Dequantize to f32 |

**Metadata Mapping**:
| GGUF Key | APR Metadata |
|----------|--------------|
| `general.architecture` | `model_type` |
| `general.name` | `model_name` |
| `llama.context_length` | `context_length` |
| `llama.embedding_length` | `hidden_size` |
| `tokenizer.ggml.tokens` | Vocabulary |

**Crate**: Custom GGUF parser

---

### 10.5 GGML (Legacy)

**Status**: 🔲 Planned (P2)

**File Structure**:
```
model.bin
├── Magic (4 bytes): "lmgg" or "tjgg"
├── Hyperparameters: model-specific struct
├── Vocabulary: token strings
└── Tensors: name + dims + data (unaligned)
```

**CLI Usage**:
```bash
apr convert model.bin -o model.apr --format ggml --arch llama
```

**Notes**:
- Legacy format, prefer GGUF for new conversions
- No standardized metadata format
- Architecture must be specified manually

---

### 10.6 ONNX

**Status**: 🔲 Planned (P1)

**File Structure**:
```
model.onnx (Protobuf)
├── ModelProto
│   ├── graph: GraphProto
│   │   ├── node[]: operators
│   │   ├── input[]: model inputs
│   │   ├── output[]: model outputs
│   │   └── initializer[]: weight tensors
│   └── metadata_props: key-value pairs
```

**CLI Usage**:
```bash
apr convert model.onnx -o model.apr
apr convert model.onnx -o model.apr --opset 17
```

**Data Types**:
| ONNX Type | APR Conversion |
|-----------|----------------|
| FLOAT | f32 |
| FLOAT16 | f16 |
| BFLOAT16 | f32 (convert) |
| INT8 | int8 |
| UINT8 | int8 (reinterpret) |

**Crate**: `onnx-pb = "0.1"` or custom protobuf parser

---

### 10.7 TensorFlow/Keras

**Status**: 🔲 Planned (P2)

**Supported Formats**:

| Format | Description | CLI Flag |
|--------|-------------|----------|
| SavedModel | Directory with `saved_model.pb` | `--format savedmodel` |
| HDF5 | Keras `.h5` files | `--format h5` |
| Frozen Graph | Single `.pb` file | `--format frozen` |
| TFLite | `.tflite` mobile format | `--format tflite` |

**CLI Usage**:
```bash
apr convert saved_model/ -o model.apr --format savedmodel
apr convert model.h5 -o model.apr --format h5
apr convert model.tflite -o model.apr --format tflite
```

**Notes**:
- HDF5 requires `hdf5` crate
- SavedModel requires protobuf parsing
- TFLite uses FlatBuffers

---

### 10.8 Tensor Name Mapping

Each source format uses different naming conventions. APR standardizes to a canonical form:

#### Whisper Model Mapping

| Source Format | Source Name | APR Name |
|---------------|-------------|----------|
| SafeTensors | `model.encoder.conv1.weight` | `encoder.conv1.weight` |
| SafeTensors | `model.encoder.embed_positions.weight` | `encoder.positional_embedding` |
| SafeTensors | `model.decoder.embed_tokens.weight` | `decoder.token_embedding` |
| PyTorch | `encoder.conv1.weight` | `encoder.conv1.weight` |
| GGUF | `encoder.conv1.weight` | `encoder.conv1.weight` |
| ONNX | `/encoder/conv1/weight` | `encoder.conv1.weight` |

#### LLaMA Model Mapping

| Source Format | Source Name | APR Name |
|---------------|-------------|----------|
| SafeTensors | `model.embed_tokens.weight` | `token_embedding` |
| SafeTensors | `model.layers.0.self_attn.q_proj.weight` | `layers.0.attn.q_proj.weight` |
| GGUF | `token_embd.weight` | `token_embedding` |
| GGUF | `blk.0.attn_q.weight` | `layers.0.attn.q_proj.weight` |

#### Full HuggingFace Whisper Mapping

| HuggingFace Name | APR Name |
|------------------|----------|
| `model.encoder.conv1.weight` | `encoder.conv1.weight` |
| `model.encoder.conv1.bias` | `encoder.conv1.bias` |
| `model.encoder.conv2.weight` | `encoder.conv2.weight` |
| `model.encoder.conv2.bias` | `encoder.conv2.bias` |
| `model.encoder.embed_positions.weight` | `encoder.positional_embedding` |
| `model.encoder.layer_norm.weight` | `encoder.layer_norm.weight` |
| `model.encoder.layer_norm.bias` | `encoder.layer_norm.bias` |
| `model.encoder.layers.N.self_attn_layer_norm.weight` | `encoder.layers.N.self_attn_layer_norm.weight` |
| `model.encoder.layers.N.self_attn.q_proj.weight` | `encoder.layers.N.self_attn.q_proj.weight` |
| `model.decoder.embed_tokens.weight` | `decoder.token_embedding` |
| `model.decoder.embed_positions.weight` | `decoder.positional_embedding` |
| `model.decoder.layer_norm.weight` | `decoder.layer_norm.weight` |
| `model.decoder.layer_norm.bias` | `decoder.layer_norm.bias` |
| `model.decoder.layers.N.self_attn_layer_norm.weight` | `decoder.layers.N.self_attn_layer_norm.weight` |
| `model.decoder.layers.N.encoder_attn_layer_norm.weight` | `decoder.layers.N.encoder_attn_layer_norm.weight` |
| `model.decoder.layers.N.final_layer_norm.weight` | `decoder.layers.N.final_layer_norm.weight` |

---

### 10.9 Expected Tensor Statistics

**Layer Norm Weights (gamma)** - MUST have mean ≈ 1.0:
```
Tensor                                   Expected Mean   Acceptable Range
encoder.layer_norm.weight                1.0 - 2.0       [0.5, 3.0]
decoder.layer_norm.weight                1.0 - 2.0       [0.5, 3.0]
*.self_attn_layer_norm.weight            1.0 - 2.0       [0.5, 3.0]
*.encoder_attn_layer_norm.weight         1.0 - 2.0       [0.5, 3.0]
*.final_layer_norm.weight                1.0 - 2.0       [0.5, 3.0]
```

**Layer Norm Bias (beta)** - MUST have mean ≈ 0.0:
```
Tensor                                   Expected Mean   Acceptable Range
*.layer_norm.bias                        0.0             [-0.5, 0.5]
```

**Attention/Linear Weights** - Should have mean ≈ 0.0:
```
Tensor                                   Expected Mean   Expected Std
*.q_proj.weight                          ~0.0            0.02 - 0.10
*.k_proj.weight                          ~0.0            0.02 - 0.10
*.v_proj.weight                          ~0.0            0.02 - 0.10
*.out_proj.weight                        ~0.0            0.02 - 0.10
*.fc1.weight                             ~0.0            0.02 - 0.05
*.fc2.weight                             ~0.0            0.02 - 0.05
```

**Embeddings**:
```
Tensor                                   Expected Mean   Expected Std
token_embedding                          ~0.0            0.02 - 0.05
positional_embedding                     ~0.0            0.01 - 0.02
```

### 10.10 Conversion Validation Requirements

1. **Shape Validation**: Every tensor must match expected shape for model architecture
2. **Value Validation**: Every tensor must have statistics within expected ranges
3. **Reference Comparison**: Converted model must produce outputs within tolerance of HF reference
4. **Inline Validation (Strict Mode)**: The `apr convert` tool MUST run the statistical checks (Section 10.9) *as tensors are being written*.
   - **Default Behavior**: If a tensor violates the "Acceptable Range" (e.g., LayerNorm mean > 3.0), the conversion **aborts** with an error.
   - **Override**: Use `--force` or `--relaxed` to bypass this check.
   - **Justification**: Better to fail early than produce a "zombie" model.

### 10.11 Known Failure Modes

| Failure | Symptom | Root Cause | Troubleshooting |
|---------|---------|------------|-----------------|
| LN weight mean=11 | Repetitive token output (e.g., "...") | Incorrect tensor scaling or name mapping | Use `apr tensors --hist` to visualize distribution |
| Missing conv bias | Zero encoder output | Conv layer not loaded | Check `--analyze-source` |
| Transposed weights | Garbage output | Row-major vs column-major confusion | Run `apr diff` vs reference |
| Truncated tensors | Partial outputs | Size mismatch during copy | Verify header vs file size |

---

## 11. Master Falsification QA Checklist (100 Points)

This checklist unifies structural, physical, operational, and conversion requirements into a single 100-point quality gate. **Every point must be testable and falsifiable.**

### A. Format & Structural Integrity (25 Points)

| # | Claim | Test Command | Falsification (How to Fail) |
|---|-------|--------------|-----------------------------|
| 1 | **Magic bytes valid** | `head -c4 m.apr \| grep APR2` | Edit file to start with "APR1" or random bytes |
| 2 | **Header size fixed** | `apr inspect m.apr --header` | Insert 1 byte before data offset |
| 3 | **Version supported** | Load v2.0 file | Load v3.0 file (should fail E003) |
| 4 | **Checksum valid** | `apr validate m.apr --checksum` | Flip 1 bit in payload (should fail E004) |
| 5 | **JSON Metadata** | `apr inspect m.apr --json` | Corrupt JSON syntax in editor |
| 6 | **Tensor Alignment** | `apr lint m.apr` checks 64B | Create file with 1-byte alignment (should warn) |
| 7 | **Index Sorted** | Validate index sort order | Swap two entries in binary index |
| 8 | **Compression** | `apr info` shows `lz4` | Compress with unsupported algo (should fail) |
| 9 | **Sharding Manifest** | Load sharded model | Delete one shard file (should fail E007) |
| 10 | **Endianness** | Read on Big Endian system | (Simulate BE) Read LE floats incorrectly |
| 11 | **Flags Parsed** | Check specific flag bits | Set undefined flag bit (should warn/ignore) |
| 12 | **Footer Magic** | Check `2RPA` at EOF | Truncate last 16 bytes (should fail) |
| 13 | **File Size** | Header size == `ls -l` | Append garbage to EOF (should warn) |
| 14 | **Tensor Offsets** | Read last tensor | Set offset beyond EOF (should fail E002) |
| 15 | **Empty Model** | Load model with 0 tensors | Create valid header, 0 tensors (should pass) |
| 16 | **Huge Header** | Metadata > 100MB | Create 200MB JSON header (should stream/fail gracefully) |
| 17 | **UTF-8 Names** | Tensor names are UTF-8 | Insert invalid UTF-8 in name (should fail) |
| 18 | **Duplicate Names** | Index has unique names | Duplicate "tensor.a" in index (should fail) |
| 19 | **Dimension Limit** | Support 8 dims | Create 9-dim tensor (should fail) |
| 20 | **Zero Dims** | Support scalar (0-dim) | Create 0-dim tensor (should pass) |
| 21 | **Datatypes** | Support all `DType` enums | Use invalid enum id 255 (should fail) |
| 22 | **Padding Bytes** | Padding is zeroed | Fill padding with 0xFF (should warn in lint) |
| 23 | **Signature** | Verify Ed25519 (if signed) | Modify 1 byte of signature (should fail E006) |
| 24 | **Encryption** | Decrypt AES-256-GCM | Provide wrong key (should fail E005) |
| 25 | **WASM Load** | Load in `wasm32` env | Run in browser (must work) |

### B. Tensor Physics & Statistics (25 Points)

| # | Claim | Test Command | Falsification (How to Fail) |
|---|-------|--------------|-----------------------------|
| 26 | **No NaNs** | `apr validate --nan-check` | Manually inject `0x7FC00000` (NaN) into f32 tensor |
| 27 | **No Infs** | `apr validate --nan-check` | Inject `0x7F800000` (+Inf) |
| 28 | **LayerNorm Mean** | `apr tensors --stats` in [0.5, 3] | Set LN weights to 11.0 (should fail/warn) |
| 29 | **LayerNorm Bias** | `apr tensors --stats` in [-0.5, 0.5] | Set LN bias to 5.0 (should fail/warn) |
| 30 | **Embedding Std** | `apr tensors --stats` < 0.2 | Set embedding std to 1.0 (should warn) |
| 31 | **Zero Tensors** | `apr validate --zero-check` | Set entire tensor to 0.0 (should warn) |
| 32 | **Shape Match** | `apr validate --shapes` | Resize tensor [384]->[383] (should fail) |
| 33 | **Vocab Match** | Metadata `n_vocab` == tensor dim | Change metadata `n_vocab` to mismatch (should fail) |
| 34 | **Quantization Range** | q8_0 values in [-127, 127] | Manually set byte -128 (if using symm quant) |
| 35 | **Attn/Linear Mean** | Mean approx 0.0 | Set Linear weight mean to 1.0 (should warn) |
| 36 | **Softmax Valid** | (If traceable) Output sums to 1.0 | (Hard to fuzz statically, use trace) |
| 37 | **Mel Filters** | Values >= 0.0 | Set negative filter bank value (should warn) |
| 38 | **Pos Embeddings** | Correct shape for ctx len | Truncate pos embedding (should fail shape) |
| 39 | **Token IDs** | (Trace) Output tokens < vocab | (Trace) Force output token > vocab_max |
| 40 | **Audio Range** | (Trace) Input in [-1, 1] | Feed audio with amp 10.0 (trace should warn) |
| 41 | **FP16 Range** | Values within FP16 limits | value > 65504 in FP16 tensor (should become Inf) |
| 42 | **Sparsity** | (If sparse) Check non-zero % | Claim sparse but 100% dense (lint warning) |
| 43 | **Dead Neurons** | (Trace) Activations never > 0 | (Trace) Detect 0-activation neuron across 100 inputs |
| 44 | **Exploding Grads** | (Trace) Values > 1e6 | (Trace) Detect activation spike |
| 45 | **Repeat Tokens** | (Trace) Repetition > 5x | (Trace) Feed silence, check for hallucination |
| 46 | **Silence Input** | (Trace) Output is empty/silence | Feed silence, check non-empty output |
| 47 | **White Noise** | (Trace) Output is garbage | Feed noise, check for confident output (bad) |
| 48 | **Mel Shape** | Filterbank matches audio/mels | Mismatch n_mels 80 vs 128 (should fail) |
| 49 | **Text Context** | Pos embed covers text ctx | Input text > max context (should truncate/fail) |
| 50 | **L2 Distance** | `apr diff` vs ref < 1.0 | Compare against random tensor (should fail L2) |

### C. Tooling & Operations (25 Points)

| # | Claim | Test Command | Falsification (How to Fail) |
|---|-------|--------------|-----------------------------|
| 51 | **Inspect Speed** | `inspect` < 100ms | (Perf) Load 100GB model (should be fast) |
| 52 | **Lint Defaults** | `apr lint` runs default checks | Create file with no license (must warn) |
| 53 | **Drama Mode** | `apr debug --drama` | Run on CI (no tty) - should output text |
| 54 | **TUI Graph** | `apr tui` renders graph | Create cyclic graph (should handle/error) |
| 55 | **TUI Stats** | `apr tui` stats match CLI | (Manual) Compare TUI number vs CLI number |
| 56 | **Diff Identity** | `apr diff a.apr a.apr` | Diff same file (must show 100% match) |
| 57 | **Diff Detection** | `apr diff a.apr b.apr` | Diff modified file (must show mismatch) |
| 58 | **Merge Average** | `apr merge` averages weights | Merge [1.0] and [3.0] -> expect [2.0] |
| 59 | **Merge TIES** | `apr merge --strategy ties` | (Complex) Verify TIES masking logic |
| 60 | **Export ONNX** | `apr export --format onnx` | Validate output with `onnx.checker` |
| 61 | **Export GGUF** | `apr export --format gguf` | Load output in `llama.cpp` |
| 62 | **Convert Quant** | `apr convert --quantize int8` | Check output size < 25% of input |
| 63 | **Convert Prune** | `apr convert --prune 0.5` | Check non-zero count is 50% |
| 64 | **Trace Output** | `apr trace` produces JSON | Corrupt input audio (should err/warn) |
| 65 | **Explain Error** | `apr explain E001` | Ask for E999 (should say unknown) |
| 66 | **Explain Tensor** | `apr explain --tensor` | Ask for random name (should fuzzy match) |
| 67 | **Analyze Source** | `convert --analyze-source` | Run on corrupt safetensors (must fail) |
| 68 | **Inline Valid** | `convert` fails on bad stat | Force bad mean in source, run convert (must abort) |
| 69 | **Force Override** | `convert --force` | Same as 68, but use --force (must pass) |
| 70 | **Cache Dir** | Uses `APR_CACHE` | Set APR_CACHE=/tmp/x (check files there) |
| 71 | **Config Load** | Uses `config.toml` | Set output_format=json in config (check output) |
| 72 | **Canary Check** | `apr canary check` | Modify weights to cause regression (should fail canary) |
| 73 | **JSON Output** | `apr inspect --json` | Pipe to `jq` (must parse) |
| 74 | **Trace Payload** | `apr trace --payload` | Corrupt tensor, check for anomaly in trace output |
| 75 | **Trace Diff** | `apr trace --diff` | Diff identical models (should show 0 drift) |

### D. Conversion & Interoperability (25 Points)

| # | Claim | Test Command | Falsification (How to Fail) |
|---|-------|--------------|-----------------------------|
| 76 | **SafeTensors** | Import `.safetensors` | Import renamed .txt file (should fail) |
| 77 | **PyTorch** | Import `.pt` (pickle) | Import malicious pickle (should fail/block) |
| 78 | **GGUF Import** | Import `.gguf` | Import GGUF with unknown arch (should fail) |
| 79 | **Roundtrip** | APR->ONNX->APR | Compare tensor values (drift < 1e-5) |
| 80 | **HF Mapping** | Maps `model.layers.0` correctly | Rename layer in source (should fail map) |
| 81 | **Q-DeepCopy** | Preserves quantization | Convert q8->apr (should stay q8 if supported) |
| 82 | **F32->BF16** | `convert --precision bf16` | Check dtype is BF16 |
| 83 | **BF16->F32** | `convert --precision f32` | Check dtype is F32 |
| 84 | **Vocab Import** | Imports full vocab | Truncate vocab in source (check count) |
| 85 | **Special Tokens** | Preserves BOS/EOS/UNK | Check metadata for token IDs |
| 86 | **Metadata Copy** | Copies model card/license | Remove metadata from source (check warnings) |
| 87 | **Tensor Name Norm** | Normalizes to `encoder.x` | Check for "model.encoder.x" (bad) |
| 88 | **Permutation** | Transposes weights if needed | Disable transpose (check output garbage) |
| 89 | **Scale Factors** | Applies rescaling (e.g. div 2) | Disable scaling (check mean drift) |
| 90 | **Sharded Import** | Imports `model-0001...` | Missing shard 2 (should fail) |
| 91 | **Remote Import** | `apr import hf://...` | Network down (should fail gracefully) |
| 92 | **Cache Hit** | Second import is fast | Clear cache, time it; run again, time it |
| 93 | **Checksum Verify** | Verify source SHA256 | Modify source file (should fail checksum) |
| 94 | **License Warning** | Warns on non-commercial | Import CC-BY-NC model (check warning) |
| 95 | **Arch Detect** | Auto-detects Whisper/LLaMA | Import unknown arch (should ask user) |
| 96 | **Output Path** | Honors `--output` | Check file exists at path |
| 97 | **Overwrite** | Fails if exists (no -f) | Create file, run export (should fail) |
| 98 | **Disk Full** | Handle ENOSPC | Simulate small disk (should fail clean) |
| 99 | **Memory Limit** | Respect `APR_RAM_LIMIT` | Set low limit, load big model (should error/mmap) |
| 100| **Golden Trace** | Passes canonical trace | Run against `golden_traces/` (must pass) |

---

## 12. Automated Validation Script

The `apr-qa` tool runs this 100-point checklist automatically.

```bash
# Run the full suite
apr-qa verify model.apr --score

# Run specific category
apr-qa verify model.apr --category physics

# CI/CD usage (fail if score < 95)
apr-qa verify model.apr --min-score 95
```

---

## 13. Import/Convert Pipeline

The complete pipeline for downloading, converting, validating, and optimizing models.

### 13.1 Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │───▶│   Import    │───▶│  Validate   │───▶│   Output    │
│ (HF/Local)  │    │ (Converter) │    │ (100-Point) │    │   (.apr)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  hf://openai/     SafeTensors→APR    Inline checks      Quantized/
  whisper-tiny     Name mapping       Tensor stats       Compressed
```

### 13.2 CLI Interface

```bash
# Full pipeline: download → convert → validate
apr import hf://openai/whisper-tiny -o whisper.apr

# With quantization
apr import hf://openai/whisper-tiny -o whisper-int8.apr --quantize int8

# Local file conversion
apr import model.safetensors -o model.apr

# Validate after import (automatic, but can run standalone)
apr validate whisper.apr --quality --min-score 95

# Post-import optimization
apr convert whisper.apr --quantize int8 --compress lz4 -o whisper-optimized.apr
```

### 13.3 SDK Interface

```rust
use aprender::format::{AprConverter, ImportOptions, ValidationConfig};

// Full pipeline with builder pattern
let apr_bytes = AprConverter::new()
    .source("hf://openai/whisper-tiny")
    .architecture("whisper")
    .validate(ValidationConfig::strict())  // Inline validation
    .quantize(Quantization::Int8)
    .compress(Compression::Lz4)
    .convert()?;

// Save to file
std::fs::write("whisper.apr", apr_bytes)?;

// Or use the high-level API
apr_import("hf://openai/whisper-tiny", "whisper.apr", ImportOptions::default())?;
```

### 13.4 Source Types

| Source | Format | Example |
|--------|--------|---------|
| HuggingFace Hub | `hf://org/repo` | `hf://openai/whisper-tiny` |
| HuggingFace File | `hf://org/repo/file` | `hf://openai/whisper-tiny/model.safetensors` |
| Local SafeTensors | Path | `./model.safetensors` |
| Local PyTorch | Path | `./model.pt` |
| Local GGUF | Path | `./model.gguf` |
| URL | `https://` | `https://example.com/model.safetensors` |

### 13.5 Tensor Name Mapping

During import, tensor names are normalized from source format to APR canonical form:

```rust
/// Tensor name mapper trait
pub trait TensorNameMapper {
    /// Map source tensor name to APR name
    fn map_name(&self, source_name: &str) -> Option<String>;

    /// Get expected tensor statistics for validation
    fn expected_stats(&self, apr_name: &str) -> Option<TensorExpectation>;
}

/// Built-in mappers
pub enum Architecture {
    Whisper,  // HuggingFace Whisper → APR Whisper
    Llama,    // HuggingFace LLaMA → APR LLaMA
    Bert,     // HuggingFace BERT → APR BERT
    Custom(Box<dyn TensorNameMapper>),
}
```

**Whisper Mapping Example:**
```
HuggingFace                           → APR
model.encoder.conv1.weight            → encoder.conv1.weight
model.decoder.layer_norm.weight       → decoder.layer_norm.weight
model.decoder.layers.0.self_attn...   → decoder.layers.0.self_attn...
```

### 13.6 Inline Validation

**Critical**: Validation runs DURING conversion, not after. If a tensor fails validation, conversion aborts immediately.

```rust
/// Validation that runs inline during conversion
pub struct InlineValidator {
    config: ValidationConfig,
    report: ValidationReport,
}

impl InlineValidator {
    /// Called for each tensor during conversion
    pub fn validate_tensor(&mut self, name: &str, data: &[f32]) -> Result<(), ValidationError> {
        let stats = TensorStats::compute(name, data);

        // Check for NaN/Inf
        if stats.nan_count > 0 {
            return Err(ValidationError::NanDetected { name: name.to_string(), count: stats.nan_count });
        }

        // Check LayerNorm weights (mean should be ~1.0)
        if name.contains("layer_norm") && name.ends_with(".weight") {
            if stats.mean < 0.5 || stats.mean > 3.0 {
                return Err(ValidationError::LayerNormMean {
                    name: name.to_string(),
                    mean: stats.mean,
                    expected: (0.5, 3.0),
                });
            }
        }

        Ok(())
    }
}
```

### 13.7 Import Options

```rust
/// Options for the import pipeline
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// Target architecture for name mapping
    pub architecture: Architecture,

    /// Validation configuration
    pub validation: ValidationConfig,

    /// Quantization (None = keep original precision)
    pub quantize: Option<Quantization>,

    /// Compression algorithm
    pub compress: Option<Compression>,

    /// Force import even if validation fails
    pub force: bool,

    /// Cache downloaded files
    pub cache: bool,

    /// HuggingFace token (from env HF_TOKEN if None)
    pub hf_token: Option<String>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            architecture: Architecture::Auto,  // Auto-detect
            validation: ValidationConfig::strict(),
            quantize: None,
            compress: None,
            force: false,
            cache: true,
            hf_token: None,
        }
    }
}
```

### 13.8 Error Handling

Import errors are specific and actionable:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("Download failed: {source} - {reason}")]
    DownloadFailed { source: String, reason: String },

    #[error("Unsupported format: {extension}")]
    UnsupportedFormat { extension: String },

    #[error("Tensor validation failed: {name} - {reason}")]
    ValidationFailed { name: String, reason: String },

    #[error("Name mapping failed: unknown tensor '{source_name}'")]
    UnknownTensor { source_name: String },

    #[error("Architecture mismatch: expected {expected}, found {found}")]
    ArchitectureMismatch { expected: String, found: String },

    #[error("Missing required tensor: {name}")]
    MissingTensor { name: String },
}
```

### 13.9 Caching

Downloaded models are cached to avoid re-downloading:

```
~/.cache/apr/
├── hf/
│   └── openai/
│       └── whisper-tiny/
│           ├── model.safetensors
│           └── config.json
└── checksum.json
```

```bash
# Clear cache
apr cache clear

# Show cache usage
apr cache info

# Pre-download without converting
apr download hf://openai/whisper-tiny
```

### 13.10 Testing Requirements

Every import path must have:

1. **Unit Test**: Test name mapping and validation logic
2. **Integration Test**: Download real model, convert, validate
3. **Golden Test**: Compare output against known-good .apr file
4. **Regression Test**: Ensure tensor statistics match expected values

```rust
#[test]
fn test_whisper_tiny_import() {
    let result = apr_import(
        "hf://openai/whisper-tiny",
        "/tmp/test.apr",
        ImportOptions::default(),
    );

    assert!(result.is_ok());

    // Validate the output
    let validator = AprValidator::new();
    let report = validator.validate(&std::fs::read("/tmp/test.apr").unwrap());

    assert!(report.passed(95), "Score: {}/100", report.total_score);

    // Check specific tensor that was previously buggy
    let reader = AprReader::new(&std::fs::read("/tmp/test.apr").unwrap()).unwrap();
    let ln_weight = reader.load_tensor("decoder.layer_norm.weight").unwrap();
    let stats = TensorStats::compute("decoder.layer_norm.weight", &ln_weight);

    assert!(stats.mean >= 0.5 && stats.mean <= 3.0,
        "decoder.layer_norm.weight mean={} should be in [0.5, 3.0]", stats.mean);
}
```

---

## 14. Implementation Roadmap

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

## 15. References

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

## 16. Appendices

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
