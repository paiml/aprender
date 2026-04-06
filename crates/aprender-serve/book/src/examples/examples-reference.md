# Examples Reference

This page provides a complete reference for all `cargo run --example` commands available in Realizar.

## Quick Reference

| Example | Description | Features |
|---------|-------------|----------|
| `quick_generate` | Real GGUF model inference | - |
| `inference` | Demo model with sampling strategies | - |
| `gguf_loading` | GGUF format parsing | - |
| `safetensors_loading` | SafeTensors format parsing | - |
| `tokenization` | BPE and SentencePiece tokenizers | - |
| `api_server` | HTTP API server | `server` |
| `model_cache` | LRU model caching | - |
| `observability_demo` | Metrics and A/B testing | `server` |
| `check_token_scores_v2` | PAR-001 parity verification | - |
| `par_001_trace_token` | Layer-by-layer tracing | - |
| `test_q4k_cuda` | Q4_K CUDA kernel test | `cuda` |
| `test_tc_attention` | Tensor Core attention | `cuda` |
| `bench_forward` | Forward pass benchmark | - |
| `bench_simd_dot` | SIMD dot product benchmark | - |

## Core Examples

### quick_generate - Real Model Inference

Load and run inference with actual GGUF models (TinyLlama, Qwen2, etc.):

```bash
# Use default model path
cargo run --example quick_generate --release

# Specify custom model
cargo run --example quick_generate --release -- /path/to/model.gguf
```

**Features demonstrated:**
- Memory-mapped GGUF loading
- Quantized tensor dequantization (Q4_0, Q8_0)
- Greedy token generation
- Both GPT-2 and SentencePiece tokenizer support

**Example output:**
```
Prompt: 'Once upon a time'
Tokens: [9038, 2501, 263, 931]
, a group of friends decided to go on a trip to a nearby city. They packed their bags

Full tokens: [9038, 2501, 263, 931, 29892, 263, 2318, ...]
```

### inference - Sampling Strategies Demo

Demonstrates the complete text generation pipeline with a demo model:

```bash
cargo run --example inference --release
```

**Features demonstrated:**
- Model configuration and initialization
- Forward pass through transformer blocks
- Greedy sampling (deterministic)
- Top-k sampling (controlled randomness)
- Top-p/nucleus sampling (dynamic vocabulary)
- Temperature control
- Progress indicators and formatted output

**Example output:**
```
╔════════════════════════════════════════╗
║     Realizar Inference Example         ║
╚════════════════════════════════════════╝

Model Configuration
┌───────────────────┬────────┐
│ Parameter         │ Value  │
├───────────────────┼────────┤
│ Vocabulary Size   │ 100    │
│ Hidden Dimension  │ 32     │
│ Number of Heads   │ 1      │
│ Number of Layers  │ 2      │
└───────────────────┴────────┘

▶ Greedy Generation
  Prompt: [1, 5, 10]
  Generated: [1, 5, 10, 42, 17, 23, 8, 91]
  Latency: 1.23ms
```

### gguf_loading - GGUF Format Parsing

Learn how to parse GGUF files (llama.cpp/Ollama format):

```bash
cargo run --example gguf_loading
```

**Features demonstrated:**
- Binary format parsing (magic number, version)
- Metadata extraction (key-value pairs)
- Tensor information (shapes, quantization types, offsets)
- F32 tensor data extraction
- Q4_0 dequantization

**Example output:**
```
=== GGUF Loading Example ===

--- GGUF Header ---
Magic: 0x46554747 (valid ✓)
Version: 3
Tensor count: 2
Metadata count: 3

--- Tensor Information ---
Tensor: embedding.weight
  - Dimensions: [1000, 256]
  - Quantization type: 0 (F32 unquantized)
  - Total elements: 256000
```

### safetensors_loading - SafeTensors Format

Load models in SafeTensors format (HuggingFace/Aprender compatible):

```bash
cargo run --example safetensors_loading
```

**Features demonstrated:**
- JSON header parsing
- Zero-copy tensor access
- F32 data extraction
- Linear regression inference example

**Example output:**
```
=== SafeTensors Loading Example ===

--- Tensor Metadata ---
Tensor: coefficients
  - dtype: F32
  - shape: [3]
  - data_offsets: [0, 12]

--- Linear Regression Inference ---
Input features: [1.0, 2.0, 3.0]
Model equation: y = 2.0*x1 + 3.0*x2 + 1.5*x3 + 0.5
Prediction: 13.00
```

### tokenization - Tokenizer Comparison

Compare Basic, BPE, and SentencePiece tokenizers:

```bash
cargo run --example tokenization
```

**Features demonstrated:**
- Basic character-level tokenization
- BPE (Byte Pair Encoding) with merge rules
- SentencePiece unigram model with scores
- Encode/decode roundtrip verification

**Example output:**
```
=== Tokenization Example ===

Input text: "hello world"

--- Basic Tokenizer ---
  Encoded: [1, 2, 3, 3, 4, 5, 6, 4, 7, 3, 8]
  Decoded: "hello world"
  Vocab size: 9

--- BPE Tokenizer ---
  Encoded: [10, 9, 4, 5, 11, 7, 8]
  Decoded: "hello world"
  Vocab size: 12

--- SentencePiece Tokenizer ---
  Encoded: [9, 5, 10]
  Decoded: "hello world"
  Vocab size: 11
```

## Server Examples

### api_server - HTTP API Server

Run the full HTTP API server with OpenAI-compatible endpoints:

```bash
cargo run --example api_server --features server
```

**Endpoints:**
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /v1/completions` - OpenAI completions
- `POST /v1/chat/completions` - OpenAI chat
- `POST /v1/embeddings` - OpenAI embeddings
- `POST /realize/generate` - Native streaming
- `POST /realize/batch` - Batch inference

**Example requests:**
```bash
# Health check
curl http://127.0.0.1:3000/health

# OpenAI chat completions
curl -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Native API - Model metadata
curl http://127.0.0.1:3000/realize/model
```

### observability_demo - Metrics and A/B Testing

Demonstrates the observability stack:

```bash
cargo run --example observability_demo --features server
```

**Features demonstrated:**
- Configuration (tracing, sampling rates)
- Metric points with labels
- Trueno-DB line protocol
- Distributed tracing with spans
- A/B testing with deterministic variant selection
- Prometheus metrics export

**Example output:**
```
--- Metrics ---
Metric: inference_latency_ms
  Value: 45.5
  Labels: {"model": "llama3-8b", "quantization": "q4_k_m"}
  Line protocol: inference_latency_ms,model=llama3-8b,quantization=q4_k_m value=45.5

--- A/B Testing ---
A/B Test: model_comparison
Variants:
  - control (llama3-8b-q4): weight 50%
  - treatment (llama3-8b-q8): weight 50%

Variant selection (deterministic by user ID):
  user-001 -> control (llama3-8b-q4)
  user-002 -> treatment (llama3-8b-q8)
```

## Caching Examples

### model_cache - LRU Model Caching

Demonstrates efficient model reuse with caching:

```bash
cargo run --example model_cache
```

**Features demonstrated:**
- Cache creation with capacity limits
- Cache hits vs misses
- LRU eviction behavior
- Cache metrics (hit rate, evictions)
- Config-based cache keys

**Example output:**
```
=== Model Cache Example ===

Created cache with capacity: 3

--- Example 1: First Access (Cache Miss) ---
  Loading model_50 (vocab=50)...
  ✓ Model loaded
  Cache metrics:
    Hits: 0
    Misses: 1
    Hit rate: 0.0%
    Cache size: 1

--- Example 2: Second Access (Cache Hit) ---
  ✓ Model retrieved from cache
  Cache metrics:
    Hits: 1
    Misses: 1
    Hit rate: 50.0%

--- Example 4: LRU Eviction ---
  Cache capacity: 3 (full)
  Loading 4th model will evict least recently used...
  ✓ LRU model evicted
  Cache metrics:
    Evictions: 1
```

## Benchmarking Examples

### bench_forward - Forward Pass Benchmarks

Profile forward pass performance:

```bash
cargo run --example bench_forward --release -- /path/to/model.gguf
```

### bench_simd_dot - SIMD Dot Product

Benchmark SIMD vs scalar dot product:

```bash
cargo run --example bench_simd_dot --release
```

### convert_and_bench_apr - APR Format Benchmarks

Convert GGUF to APR and benchmark:

```bash
cargo run --example convert_and_bench_apr --release -- /path/to/model.gguf
```

## PAR-001 Parity Testing Examples

These examples were created for verifying parity with llama.cpp inference. They trace through the forward pass layer-by-layer to identify numerical discrepancies.

### check_token_scores_v2 - Complete Inference with RoPE

The canonical parity test that correctly predicts "W" (token 399) matching llama.cpp:

```bash
cargo run --example check_token_scores_v2 --release
```

**Features demonstrated:**
- BOS token handling (token 1)
- RoPE positional encoding
- Causal attention with KV cache
- Multi-token inference

### par_001_trace_token - Layer-by-Layer Tracing

Traces hidden states through all transformer layers:

```bash
cargo run --example par_001_trace_token --release
```

### par_001_qkv_parity - Q/K/V Projection Verification

Verifies Q, K, V projections match HuggingFace reference:

```bash
cargo run --example par_001_qkv_parity --release
```

### par_001_verify_q4k_matvec - Q4_K Matrix-Vector Verification

Verifies quantized matrix-vector multiplication:

```bash
cargo run --example par_001_verify_q4k_matvec --release
```

## CUDA/GPU Examples

These examples require `--features cuda` and NVIDIA GPU hardware.

### test_q4k_cuda - Q4_K CUDA Kernel

```bash
cargo run --example test_q4k_cuda --release --features cuda
```

### test_tc_attention - Tensor Core Attention

```bash
cargo run --example test_tc_attention --release --features cuda
```

### cuda_debug - CUDA Debugging

```bash
cargo run --example cuda_debug --release --features cuda
```

### debug_ptx - PTX Generation Debug

```bash
cargo run --example debug_ptx --release --features cuda
```

## Debugging Examples

### check_tensors - Inspect Tensor Dimensions

```bash
cargo run --example check_tensors -- /path/to/model.gguf
```

### check_gguf_metadata - Inspect GGUF Metadata

```bash
cargo run --example check_gguf_metadata -- /path/to/model.gguf
```

### check_tokenizer - Verify Tokenizer Behavior

```bash
cargo run --example check_tokenizer -- /path/to/model.gguf
```

### debug_forward - Forward Pass Debugging

```bash
cargo run --example debug_forward --release -- /path/to/model.gguf
```

### debug_layer0_trace - Layer 0 Detailed Trace

```bash
cargo run --example debug_layer0_trace --release
```

### debug_v_weight_layout - V Weight Layout Analysis

```bash
cargo run --example debug_v_weight_layout --release
```

## Running Examples

### Basic Usage

```bash
# Run with default settings
cargo run --example <name>

# Run in release mode (10-20x faster)
cargo run --example <name> --release

# With feature flags
cargo run --example <name> --features server
cargo run --example <name> --features cuda

# With arguments
cargo run --example <name> -- arg1 arg2
```

### Common Patterns

```bash
# Quick test with demo model
cargo run --example inference --release

# Real model inference
cargo run --example quick_generate --release -- ~/models/tinyllama.Q4_0.gguf

# Start API server
cargo run --example api_server --features server --release

# Profile performance
cargo run --example bench_forward --release -- ~/models/model.gguf
```

## See Also

- [Inference Demo](./inference.md) - Detailed inference walkthrough
- [API Server Demo](./api-server.md) - HTTP API details
- [Tokenization Comparison](./tokenization.md) - Tokenizer deep dive
- [CLI Reference](../cli/command-structure.md) - CLI commands
