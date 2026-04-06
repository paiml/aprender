# Realizar Examples

This directory contains example programs demonstrating the capabilities of realizar, a pure Rust ML inference engine.

## Available Examples (414 total)

> **32 documented examples** below + **382 development/debugging scripts** (bench_*, check_*, compare_*, debug_*, par_*, profile_*, test_*, trace_*, verify_*).

### Core Examples (No Extra Features Required)

#### 1. `inference.rs` - End-to-End Text Generation

Demonstrates the complete text generation pipeline:
- Model initialization with configuration
- Forward pass through transformer blocks
- Text generation with various sampling strategies (greedy, top-k, top-p)

**Run:**
```bash
cargo run --example inference
```

#### 2. `api_server.rs` - HTTP API Server

Shows how to deploy realizar as a REST API service:
- Create a demo model
- Start the HTTP server (default: http://127.0.0.1:3000)
- Handle tokenization, generation, and batch requests

**Run:**
```bash
cargo run --example api_server
# Server runs at http://127.0.0.1:3000
# Test: curl http://127.0.0.1:3000/health
```

#### 3. `tokenization.rs` - Tokenizer Comparison

Compares different tokenization strategies:
- Basic tokenizer (vocabulary-based)
- BPE (Byte Pair Encoding) tokenizer
- SentencePiece tokenizer
- Encoding and decoding workflows

**Run:**
```bash
cargo run --example tokenization
```

#### 4. `safetensors_loading.rs` - SafeTensors Model Loading

Demonstrates SafeTensors file format support:
- Load SafeTensors files (aprender compatibility)
- Extract tensor data using helper API
- Inspect model structure and metadata
- Interoperability with aprender-trained models

**Run:**
```bash
cargo run --example safetensors_loading
```

#### 5. `model_cache.rs` - Model Caching with LRU

Demonstrates ModelCache for efficient model reuse:
- Cache creation with capacity limits
- Model loading with cache hits/misses
- Metrics tracking (hit rate, evictions)
- LRU eviction behavior
- Config-based cache keys

**Run:**
```bash
cargo run --example model_cache
```

#### 6. `gguf_loading.rs` - GGUF Format Loading

Demonstrates GGUF file format support (llama.cpp/Ollama):
- Load GGUF files (binary format parsing)
- Parse header, metadata, and tensor information
- Inspect model structure (dimensions, quantization types)
- Extract and dequantize tensor data (F32, Q4_0)
- Compatible with llama.cpp and Ollama models

**Run:**
```bash
cargo run --example gguf_loading
```

#### 7. `apr_loading.rs` - APR Format Loading (Sovereign Stack)

Demonstrates Aprender's native .apr format - the PRIMARY inference format
for the sovereign AI stack:
- APR format specification (magic, header, flags)
- All supported model types (Linear, NN, MoE, etc.)
- Header parsing and format detection
- Inference with test models
- Batch prediction

**Run:**
```bash
cargo run --example apr_loading
```

#### 8. `apr_gpu_benchmark.rs` - **APR 2.71x Ollama GPU Showcase** ⭐

**MILESTONE ACHIEVED (2026-01-15)**: APR format achieves 2.71x Ollama on GPU!

Demonstrates the APR format GPU inference pipeline:
- GGUF → APR conversion with quantization preserved (Q4_K, Q6_K)
- APR loading via `OwnedQuantizedModel::from_apr()`
- GPU inference via `OwnedQuantizedModelCuda`
- Side-by-side GGUF vs APR benchmarking

**Results:**
| Format | M=8 | M=16 | M=32 |
|--------|-----|------|------|
| GGUF | - | 816.4 tok/s (2.81x) | - |
| APR | 723.8 tok/s (2.49x) ✅ | 799.9 tok/s (2.75x) ✅ | 763.9 tok/s (2.63x) ✅ |

**Target**: 582 tok/s (2X Ollama) — ALL EXCEEDED

**Run:**
```bash
MODEL_PATH=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  cargo run --example apr_gpu_benchmark --release --features cuda
```

#### 9. `observability_demo.rs` - Distributed Tracing & Metrics

Demonstrates the observability stack:
- OpenTelemetry-style tracing with span creation
- W3C TraceContext propagation (traceparent header)
- Prometheus metrics export
- A/B testing with variant tracking
- Custom span attributes and events

**Run:**
```bash
cargo run --example observability_demo
```

### GPU & Performance Parity Examples

#### 9. `gpu_matvec_benchmark.rs` - GPU vs SIMD Matvec

Benchmarks GPU vs SIMD for matrix-vector operations:
- trueno SIMD backend performance
- trueno WGPU GPU backend performance
- Demonstrates GPU is 2.7x SLOWER for MATVEC (IMP-600 finding)

**Run:**
```bash
cargo run --example gpu_matvec_benchmark --features gpu
```

#### 10. `gpu_gemm_benchmark.rs` - GPU GEMM Performance

Benchmarks GPU GEMM (matrix-matrix) operations:
- trueno scalar baseline
- trueno WGPU GPU backend
- Demonstrates GPU is 57x FASTER for large GEMM (1024³)

**Run:**
```bash
cargo run --example gpu_gemm_benchmark --features gpu
```

#### 11. `parity_035_m4_verification.rs` - M4 Target Verification

Verifies M4 milestone (90% llama.cpp parity):
- Ollama baseline measurement
- realizar GPU inference
- Gap analysis and reporting

**Run:**
```bash
cargo run --example parity_035_m4_verification --features gpu
```

#### 12. `parity_036_gpu_attention.rs` - GPU Attention Benchmark

Benchmarks GPU attention implementation:
- Standard attention vs FlashAttention
- Memory usage comparison
- Throughput measurement

**Run:**
```bash
cargo run --example parity_036_gpu_attention --features gpu
```

#### 13. `parity_038_async_streams.rs` - CUDA Async Streams

Demonstrates CUDA stream-based async execution:
- Overlapped compute and transfer
- 2x speedup verification
- Multi-stream orchestration

**Run:**
```bash
cargo run --example parity_038_async_streams --features cuda
```

#### 14. `parity_039_flash_attention.rs` - FlashAttention O(N) Memory

Verifies FlashAttention memory complexity:
- O(N) vs O(N²) memory comparison
- 32x memory reduction at seq_len=512
- Numerical correctness verification

**Run:**
```bash
cargo run --example parity_039_flash_attention --features cuda
```

#### 15. `parity_040_fp16_attention.rs` - FP16 Tensor Core Baseline

Benchmarks FP16 Tensor Core attention:
- FP32 baseline
- FP16 tiled GEMM
- Tensor Core investigation results

**Run:**
```bash
cargo run --example parity_040_fp16_attention --features cuda
```

### IMP (Improvement) Verification Examples

#### 16. `imp_700_realworld_verification.rs` - Real-World Ollama Benchmark

Direct HTTP benchmarking against live Ollama server:
- Throughput measurement (240+ tok/s)
- CV-based statistical stopping
- Gap quantification

**Run:**
```bash
cargo run --example imp_700_realworld_verification
```

#### 17. `imp_701_performance_gap.rs` - Performance Gap Analysis

Analyzes the performance gap between realizar and Ollama:
- Component-level breakdown
- Bottleneck identification
- Optimization recommendations

**Run:**
```bash
cargo run --example imp_701_performance_gap
```

#### 18. `imp_800_kv_cache_falsification.rs` - KV Cache Verification

Falsifies KV cache performance claims:
- Cache hit/miss analysis
- Memory layout verification
- Speedup measurement (64-512x)

**Run:**
```bash
cargo run --example imp_800_kv_cache_falsification
```

#### 19. `imp800_gpu_parity.rs` - GPU Parity Benchmark

Full GPU parity benchmark suite:
- realizar vs Ollama on GPU
- Statistical analysis
- Gap reporting

**Run:**
```bash
cargo run --example imp800_gpu_parity --features cuda
```

#### 20. `imp_801_flash_attention_falsification.rs` - FlashAttention Falsification

Popperian falsification of FlashAttention claims:
- Memory complexity verification
- Numerical accuracy checks
- Performance comparison

**Run:**
```bash
cargo run --example imp_801_flash_attention_falsification --features cuda
```

#### 21. `imp900_optimized_gpu.rs` - Optimized GPU Pipeline

Demonstrates fully optimized GPU inference pipeline:
- Weight caching
- Async streams
- FlashAttention integration

**Run:**
```bash
cargo run --example imp900_optimized_gpu --features cuda
```

### Pipeline & TUI Examples

#### 22. `pipeline_tui.rs` - Inference Pipeline TUI

Terminal UI visualization of inference pipeline:
- Real-time token generation
- Latency sparklines
- Memory usage tracking
- ANSI 256-color rendering

**Run:**
```bash
cargo run --example pipeline_tui
```

### trueno Integration Examples

#### 23. `trueno_ab_test.rs` - trueno A/B Testing

Demonstrates trueno backend A/B testing:
- SIMD vs GPU backend comparison
- Statistical significance testing
- Performance regression detection

**Run:**
```bash
cargo run --example trueno_ab_test --features gpu
```

#### 24. `trueno_dot_test.rs` - trueno Dot Product Test

Tests trueno dot product implementations:
- 4-accumulator AVX2 optimization
- Scalar baseline comparison
- Numerical accuracy verification

**Run:**
```bash
cargo run --example trueno_dot_test
```

### Examples Requiring Features

#### 25. `wine_lambda.rs` - AWS Lambda Wine Quality Predictor

Production-ready wine quality rating predictor for AWS Lambda deployment.
Inspired by [paiml/wine-ratings](https://github.com/paiml/wine-ratings).

Features:
- Predicts wine quality (0-10) from 11 physicochemical properties
- Sub-millisecond inference latency
- Cold start detection and metrics
- Prometheus metrics export
- Drift detection for production monitoring
- Ready for ARM64 Graviton deployment

**Run:**
```bash
cargo run --example wine_lambda
```

**Deploy to Lambda:**
```bash
# Build for ARM64
cargo build --release --target aarch64-unknown-linux-gnu --features lambda

# Package and deploy
cp target/aarch64-unknown-linux-gnu/release/wine_lambda bootstrap
zip wine_lambda.zip bootstrap
aws lambda create-function --function-name wine-predictor --runtime provided.al2 \
  --architecture arm64 --handler bootstrap --zip-file fileb://wine_lambda.zip
```

#### 26. `data_pipeline.rs` - Alimentar Data Pipeline Integration

End-to-end ML data pipeline demonstrating alimentar + realizar integration:
- Load built-in Iris dataset (embedded, no download)
- Data quality checks
- Transform pipeline (shuffle)
- Train/test split
- Inference with classification
- Drift detection
- DataLoader batching

**Run:**
```bash
cargo run --example data_pipeline --features alimentar-data
```

#### 27. `train_model.rs` - Train Real .apr Models

Train actual ML models with aprender and save as .apr format:
- Wine quality linear regression (R² ~0.93)
- Save to .apr binary format
- Load and verify predictions
- Model coefficients extraction

**Run:**
```bash
cargo run --example train_model --features "aprender-serve"
```

**Output:**
- `wine_regressor.apr` (141 bytes, gitignored)

#### 28. `build_mnist_model.rs` - Build MNIST Classifier

Build and save an MNIST digit classifier:
- Create neural network architecture
- Save as .apr format for deployment
- Compatible with serve_mnist example

**Run:**
```bash
cargo run --example build_mnist_model --features "aprender-serve"
```

#### 29. `serve_mnist.rs` - Serve MNIST Model via HTTP

Serve an MNIST classifier via HTTP API:
- Load pre-built MNIST model
- REST API for digit classification
- Batch prediction support

**Run:**
```bash
cargo run --example serve_mnist --features "aprender-serve"
```

#### 30. `mnist_apr_benchmark.rs` - MNIST APR vs PyTorch Benchmark

Benchmark .apr format inference against PyTorch baseline:
- Cold start latency comparison
- Inference throughput measurement
- Memory usage analysis
- Pareto frontier visualization data

**Run:**
```bash
cargo run --example mnist_apr_benchmark --features "aprender-serve"
```

#### 31. `performance_parity.rs` - Full Performance Parity Suite

Comprehensive performance parity benchmark suite (109KB):
- All PARITY-xxx test implementations
- Ollama/llama.cpp comparison
- Statistical analysis

**Run:**
```bash
cargo run --example performance_parity --features "cuda"
```

#### 32. `cuda_debug.rs` - CUDA Debug Utilities

Debug utilities for CUDA development:
- PTX inspection
- Kernel launch debugging
- Memory transfer verification

**Run:**
```bash
cargo run --example cuda_debug --features cuda
```

#### 33. `quick_generate.rs` - Quick Text Generation (Real Models)

Fast text generation with real GGUF models:
- Load and run inference on actual model files
- Minimal boilerplate for quick experimentation

**Run:**
```bash
MODEL_PATH=/path/to/model.gguf cargo run --example quick_generate --release
```

#### 34. `cuda_chat_completions.rs` - CUDA Chat Completions

End-to-end chat completion on GPU:
- Chat template formatting
- CUDA-accelerated inference
- OpenAI-compatible interface

**Run:**
```bash
MODEL_PATH=/path/to/model.gguf cargo run --example cuda_chat_completions --release --features cuda
```

#### 35. `gpu_showcase_benchmark.rs` - GPU Showcase Benchmark

Comprehensive GPU performance showcase:
- GEMV, GEMM, attention, flash attention benchmarks
- RTX 4090 performance profiling
- Comparison across batch sizes

**Run:**
```bash
cargo run --example gpu_showcase_benchmark --release --features cuda
```

#### 36. `apr_mmap_loading.rs` - Memory-Mapped APR Loading

Demonstrates memory-mapped .apr file loading:
- Zero-copy model access via mmap
- Reduced memory footprint
- Fast model switching

**Run:**
```bash
cargo run --example apr_mmap_loading
```

#### 37. `convert_apr_q4k.rs` - GGUF to APR Q4_K Conversion

Convert GGUF models to APR format with Q4_K quantization:
- Format conversion pipeline
- Quantization preservation
- Output verification

**Run:**
```bash
MODEL_PATH=/path/to/model.gguf cargo run --example convert_apr_q4k --release --features cuda
```

### Development & Debugging Scripts (382 files)

The `examples/` directory also contains 382 development scripts organized by prefix:

| Prefix | Count | Purpose |
|--------|-------|---------|
| `bench_*` | ~40 | Microbenchmarks for specific operations |
| `check_*` | ~50 | Tensor/weight/metadata inspection |
| `compare_*` | ~30 | CPU vs GPU, GGUF vs APR parity checks |
| `debug_*` | ~70 | Layer-by-layer debugging, divergence tracing |
| `par_001_*` | ~30 | Parity verification against HuggingFace |
| `profile_*` | ~25 | Performance profiling scripts |
| `test_*` | ~40 | Integration test harnesses |
| `trace_*` | ~30 | Forward pass tracing, attention visualization |
| `verify_*` | ~20 | Correctness verification scripts |

These are used during development and are runnable but not documented individually.

## Quick Reference

| Example | Features Required | Description |
|---------|------------------|-------------|
| `inference` | None | Text generation with sampling |
| `api_server` | None | HTTP REST API server |
| `tokenization` | None | BPE/SentencePiece comparison |
| `safetensors_loading` | None | Load .safetensors files |
| `model_cache` | None | LRU caching demo |
| `gguf_loading` | None | Load llama.cpp models |
| `apr_loading` | None | Load .apr models |
| `observability_demo` | None | Tracing & metrics |
| `gpu_matvec_benchmark` | `gpu` | GPU vs SIMD matvec |
| `gpu_gemm_benchmark` | `gpu` | GPU GEMM benchmark |
| `parity_035_m4_verification` | `gpu` | M4 milestone verification |
| `parity_036_gpu_attention` | `gpu` | GPU attention benchmark |
| `parity_038_async_streams` | `cuda` | CUDA async streams |
| `parity_039_flash_attention` | `cuda` | FlashAttention O(N) memory |
| `parity_040_fp16_attention` | `cuda` | FP16 Tensor Core |
| `imp_700_realworld_verification` | None | Ollama HTTP benchmark |
| `imp_701_performance_gap` | None | Performance gap analysis |
| `imp_800_kv_cache_falsification` | None | KV cache verification |
| `imp800_gpu_parity` | `cuda` | GPU parity benchmark |
| `imp_801_flash_attention_falsification` | `cuda` | FlashAttention falsification |
| `imp900_optimized_gpu` | `cuda` | Optimized GPU pipeline |
| `pipeline_tui` | None | Inference pipeline TUI |
| `trueno_ab_test` | `gpu` | trueno A/B testing |
| `trueno_dot_test` | None | trueno dot product test |
| `wine_lambda` | None | AWS Lambda predictor |
| `data_pipeline` | `alimentar-data` | End-to-end ML pipeline |
| `train_model` | `aprender-serve` | Train & save models |
| `build_mnist_model` | `aprender-serve` | Build MNIST classifier |
| `serve_mnist` | `aprender-serve` | Serve MNIST via HTTP |
| `mnist_apr_benchmark` | `aprender-serve` | Benchmark .apr vs PyTorch |
| `performance_parity` | `cuda` | Full parity suite |
| `cuda_debug` | `cuda` | CUDA debug utilities |
| `quick_generate` | None | Quick text generation |
| `cuda_chat_completions` | `cuda` | GPU chat completions |
| `gpu_showcase_benchmark` | `cuda` | GPU performance showcase |
| `apr_mmap_loading` | None | Memory-mapped APR loading |
| `convert_apr_q4k` | `cuda` | GGUF→APR Q4_K conversion |

## Quick Start

Build all examples:
```bash
cargo build --examples
```

Run a specific example:
```bash
cargo run --example <name>
```

Run with features:
```bash
cargo run --example <name> --features "<feature-list>"
```

## Performance Parity Examples (PARITY-xxx)

These examples verify performance parity with Ollama and llama.cpp:

| Example | PARITY ID | What It Verifies |
|---------|-----------|------------------|
| `parity_035_m4_verification` | PARITY-035 | M4 milestone (90% parity) |
| `parity_036_gpu_attention` | PARITY-036 | GPU attention performance |
| `parity_038_async_streams` | PARITY-038 | CUDA async execution (2x) |
| `parity_039_flash_attention` | PARITY-039 | FlashAttention O(N) memory |
| `parity_040_fp16_attention` | PARITY-040 | FP16 Tensor Core baseline |

## trueno Simulation Research Integration

The GPU examples demonstrate findings from the trueno simulation research:

- **GPU threshold 100K**: `gpu_matvec_benchmark` shows GPU slower for small ops
- **PCG determinism**: All benchmarks use reproducible RNG seeds
- **SIMD math properties**: `trueno_dot_test` verifies 4-accumulator pattern
- **PTX barriers**: `parity_038_async_streams` uses correct synchronization
- **1e-4 GPU tolerance**: All GPU examples verify numerical accuracy

## Notes

- Examples use demo/test models for demonstration
- Real model loading requires proper GGUF or SafeTensors files
- API server examples run indefinitely (Ctrl+C to stop)
- All examples follow EXTREME TDD principles with comprehensive testing
- GPU examples require NVIDIA RTX 4090 or compatible GPU
