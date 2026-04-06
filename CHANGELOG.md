# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **GH-88: APR GPU inference for GQA models** — `AprMetadata` now accepts `num_key_value_heads` alias from Q4K converter metadata. Previously, GQA models (Qwen2.5-Coder) panicked during GPU weight preload because `num_kv_heads` defaulted to `num_heads`, causing KV dimension mismatch.
- **GH-89: bench_forward hardcoded model path** — Example now accepts model path as `argv[1]` or `GGUF_MODEL` env var instead of hardcoded path (SATD removal).
- **GH-215: Q5K/Q6K CUDA kernel bounds checks** — Added predicated loads for non-256-aligned K dimensions in batched Q6K and single Q5K kernels.

### Performance
- **Qwen2.5-Coder 1.5B (APR Q4_K, RTX 4090)**: 240 tok/s GPU, 18 tok/s CPU
- **Qwen2.5-Coder 0.5B (APR, RTX 4090)**: Verified correct output ("2+2=4") on both GPU and CPU

## [0.8.0] - 2026-02-26

### Changed
- Bumped trueno dependency to 0.16.0
- Bumped aprender dependency to >=0.27
- Minor version bump for PAIML Sovereign AI Stack coordinated release

## [0.6.0] - 2026-01-15

### 🏆 MILESTONE: APR 2.71x Ollama GPU Achieved

Both GGUF and APR formats now exceed 2X Ollama on GPU!

| Format | M=16 | vs Ollama | Status |
|--------|------|-----------|--------|
| **GGUF** | 824.7 tok/s | 2.83x | ✅ EXCEEDED |
| **APR** | 799.9 tok/s | 2.75x | ✅ EXCEEDED |
| Target | 582 tok/s | 2.00x | - |

### Added
- **APR GPU Inference** - Full APR format support on GPU via `OwnedQuantizedModelCuda`
  - GGUF → APR conversion with quantization preserved (Q4_K, Q6_K)
  - `OwnedQuantizedModel::to_apr_bytes()` - Serialize with quantized weights
  - `OwnedQuantizedModel::from_apr()` - Load APR files preserving quantization
  - Batched inference at M=8, M=16, M=32 all exceeding 2X Ollama target
- **`apr_gpu_benchmark.rs`** - Featured example for APR GPU showcase
  - Side-by-side GGUF vs APR benchmarking
  - Produces 1.9GB APR file with full model fidelity

### Fixed
- **LM Head Tensor Lookup** - Fixed `from_apr()` matching wrong tensor
  - Bug: `.contains("output.weight")` matched `blk.0.attn_output.weight`
  - Fix: Prioritize exact match `t.name == "output.weight"`

### Performance
- **APR M=8**: 723.8 tok/s (2.49x Ollama)
- **APR M=16**: 799.9 tok/s (2.75x Ollama)
- **APR M=32**: 763.9 tok/s (2.63x Ollama)
- **GGUF M=16**: 824.7 tok/s (2.83x Ollama) - control benchmark

### Quality
- APR format now fully interoperable with GGUF on GPU path
- Quantized weights (Q4_K, Q6_K) preserved through format conversion
- Criterion benchmark updated with M=32 for scientific validation

## [0.3.2] - 2025-12-30

### Added
- **Q4_0×Q8_0 Integer SIMD Matmul** - 2x inference speedup for GGUF Q4_0 models
  - Quantize activations to Q8_0 format for integer multiply-accumulate
  - Use `_mm256_maddubs_epi16` for AVX2 SIMD acceleration
  - Sign trick algorithm matching llama.cpp's approach
  - 2-block loop unrolling with prefetch hints
- **APR SIMD Matmul** - 5-7x inference speedup for APR transformer models
  - Trueno Matrix/Vector SIMD acceleration
  - Scalar fallback for edge cases
  - APR now achieves near-GGUF parity (1.4-6x vs 6-10x before)

### Changed
- **Aprender Dependency** - Updated from 0.14 to 0.20.1
  - Latest TransformerLM and MoE support
  - Improved APR format handling

### Performance
- **GGUF Q4_0**: 8.4-11.9 tok/s (was 4.2-7.1 tok/s) - 2x improvement
- **APR tiny_64x1**: 66 µs (was 500 µs) - 7.5x improvement
- **APR medium_256x4**: 9.0 ms (was 48 ms) - 5.3x improvement
- Achieved Candle parity (9.2-9.9 tok/s) for GGUF inference
- 20-26% of llama.cpp performance (42-45 tok/s)

### Quality
- All 806 tests pass (with aprender-serve feature)
- All falsification tests pass
- Clippy: 0 warnings

## [0.2.0] - 2025-01-19

### Added
- **Batch Inference API** - Process multiple prompts in a single request
  - `POST /batch/tokenize` - Tokenize multiple texts
  - `POST /batch/generate` - Generate text for multiple prompts
  - Linear scaling performance characteristics
  - Comprehensive integration tests
- **Server-Sent Events (SSE) Streaming** - Real-time token-by-token generation
  - `POST /stream/generate` - Stream generated tokens as they're produced
  - Token events and completion events
  - JavaScript and Python client examples
  - Reduced perceived latency for long generations
- **Model Caching Infrastructure** - LRU cache for reduced cold start latency
  - Thread-safe concurrent access with `Arc<RwLock>`
  - Configurable cache capacity
  - Automatic LRU eviction when capacity reached
  - Cache metrics tracking (hits, misses, evictions, size)
  - Hit rate calculation for monitoring
- **SafeTensors Interoperability** - Load models from aprender
  - `safetensors_loading.rs` example
  - Seamless integration with aprender ecosystem
  - Property-based tests for SafeTensors parsing
- **Performance Benchmarks** - Comprehensive benchmark suite
  - Cache performance benchmarks (hit/miss latency, eviction, concurrency)
  - Batch inference benchmarks
  - Cache key creation benchmarks
  - Hit rate calculation benchmarks

### Fixed
- Race condition in concurrent cache access test
- Float comparison in cache metrics tests
- Type complexity clippy warnings with type aliases

### Performance
- Cache hit latency: ~40 ns
- Cache miss + load: ~14 µs
- Concurrent cache access (4 threads): ~94 µs
- Metrics access: ~4.6 ns
- Hit rate calculation: ~430 ps

### Quality Metrics
- Test coverage: 95.76% (up from 95.46%)
- Tests: 286 total (228 unit + 6 integration + 52 property)
- TDG Score: 96.4/100 (A+ grade, up from 93.9)
- Dead code: 0%
- Clippy: 0 warnings
- Benchmarks: 3 suites (tensor_ops, inference, cache)

### Documentation
- Complete API documentation for batch endpoints
- SSE streaming documentation with client examples
- Cache architecture and usage documentation
- Expanded mdBook documentation (168 files)

## [0.1.0] - 2024-11-18

### Added
- Pure Rust ML inference engine from scratch
- GGUF format parser (v3 support)
  - Full metadata parsing (all value types including Arrays)
  - Tensor information parsing
  - Quantization type support (Q4_0, Q8_0)
- Safetensors format parser
  - JSON header parsing
  - Tensor data loading
  - Zero-copy tensor access
- Transformer model implementation (LLaMA architecture)
  - Multi-head attention with RoPE
  - Feed-forward networks with SwiGLU
  - RMSNorm layer normalization
  - Configurable layers, heads, dimensions
- Tokenization support
  - Basic character-level tokenizer
  - BPE (Byte Pair Encoding) tokenizer
  - SentencePiece tokenizer with unigram model
- Text generation strategies
  - Greedy sampling
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Temperature scaling
  - Configurable generation parameters
- REST API with Axum
  - `/health` - Health check endpoint
  - `/tokenize` - Text tokenization endpoint
  - `/generate` - Text generation endpoint
  - Demo mode for testing
- CLI binary (`realizar`)
  - `serve` command with `--demo` flag
  - `info` command for version information
  - Configurable host and port
- Comprehensive test suite
  - 260 total tests (211 unit + 42 property + 7 integration)
  - 95.46% code coverage (region)
  - 100% mutation score on api.rs
  - Property-based tests with proptest
  - Integration tests for CLI
- Performance benchmarks
  - Tensor operations benchmark suite
  - Inference benchmark suite
  - Sub-millisecond generation (<1ms p50)
- Examples
  - `inference.rs` - Model inference demonstration
  - `tokenization.rs` - Tokenizer comparison
  - `api_server.rs` - HTTP server demo
- GPU acceleration via Trueno (optional feature)

### Performance
- Forward pass (1 token): ~17.5 µs
- 5-token generation: ~504 µs
- 10-token generation: ~1.54 ms
- 20-token generation: ~5.52 ms

### Quality Metrics
- Test coverage: 95.46% (region), 91.33% (function)
- TDG Score: 93.9/100 (A grade)
- Mutation score: 100% (api.rs)
- Clippy: 0 warnings
- Rustfmt: compliant

### Dependencies
- Trueno v0.2.2 - SIMD/GPU compute primitives
- Axum v0.7 - HTTP server framework
- Tokio v1 - Async runtime
- Clap v4 - CLI argument parsing
- Serde v1 - Serialization
- Thiserror v1 - Error handling

[Unreleased]: https://github.com/paiml/realizar/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/paiml/realizar/compare/v0.6.0...v0.8.0
[0.6.0]: https://github.com/paiml/realizar/compare/v0.3.2...v0.6.0
[0.3.2]: https://github.com/paiml/realizar/compare/v0.2.0...v0.3.2
[0.2.0]: https://github.com/paiml/realizar/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/paiml/realizar/releases/tag/v0.1.0
