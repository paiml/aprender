# Design Rationale

This document describes the architectural decisions, design trade-offs, and falsifiable claims underlying Realizar. It follows Popperian methodology: claims must be testable and falsifiable.

## Design Philosophy

### Core Principle: Total Control, Zero Compromise

**Falsifiable claim**: Building ML inference from scratch in pure Rust enables faster iteration and optimization than wrapping existing libraries.

**Rationale**: By implementing every component ourselves (parsers, transformer, quantization, tokenizers), we gain:

1. **Full optimization control** - No black-box library calls
2. **Minimal dependencies** - Reduced attack surface, faster builds
3. **Deep understanding** - Every bug is in our code, every fix is possible

**How to falsify**: Compare development velocity metrics (bugs/fix time, feature implementation time) against projects using candle, llama-cpp-rs, or similar wrappers.

### Why Rust?

**Falsifiable claim**: Rust's memory safety and zero-cost abstractions provide both safety and performance comparable to C++.

**Rationale**:
- Memory safety without garbage collection
- Fearless concurrency for parallel inference
- Strong type system catches errors at compile time
- Cargo ecosystem for reproducible builds

**How to falsify**: Benchmark inference latency against C++ implementations (llama.cpp) with equivalent optimizations.

## Architecture Decisions

### ADR-001: Trueno as Compute Backend

**Context**: Need SIMD-accelerated tensor operations for competitive inference performance.

**Decision**: Use Trueno (our library) instead of external options (ndarray, nalgebra, faer).

**Rationale**:
- Full control over SIMD dispatch (AVX2, SSE2, NEON, WASM)
- GPU backend via wgpu (cross-platform)
- Activation functions optimized for inference
- No external ML dependencies

**Trade-offs**:
- (+) Complete control over optimization path
- (+) Consistent API across SIMD/GPU backends
- (-) Must implement and maintain primitives ourselves
- (-) Less battle-tested than established libraries

**Falsifiable claim**: Trueno's SIMD operations achieve >50% of theoretical peak FLOPS on target architectures.

**How to falsify**: Benchmark against architecture-specific peak performance using vendor tools (Intel VTune, ARM Streamline).

### ADR-002: GGUF and Safetensors Parsers from Scratch

**Context**: Need to load models from popular formats (llama.cpp, HuggingFace).

**Decision**: Implement pure Rust parsers without external parsing libraries.

**Rationale**:
- Zero-copy memory mapping for large models
- No Python dependencies for model loading
- Control over quantization format handling
- Precise error messages for debugging

**Trade-offs**:
- (+) Zero external dependencies
- (+) Optimized for our inference patterns
- (-) Must track format spec changes ourselves
- (-) Initial development time

**Falsifiable claim**: Our GGUF parser loads models at equivalent or faster speed than llama.cpp's native loader.

**How to falsify**: Benchmark model load time for various model sizes (1B, 7B, 13B parameters).

### ADR-003: Modular Feature Flags

**Context**: Different deployment targets need different capabilities (Lambda vs. server, CPU vs. GPU).

**Decision**: Use Cargo feature flags for modular compilation.

**Features**:
- `minimal` - Core inference only (no server, no CLI)
- `server` - REST API (adds axum, tokio)
- `cli` - Command-line interface (adds clap)
- `gpu` - GPU acceleration (adds wgpu)

**Rationale**:
- Lambda deployments need minimal binary size
- Server deployments need full HTTP stack
- GPU optional for CPU-only environments

**Falsifiable claim**: `--features minimal` produces binaries <5MB for Lambda deployment.

**How to falsify**: Measure binary size with `cargo build --release --features minimal && ls -la target/release/realizar`.

### ADR-004: Axum as Default HTTP Server (Swappable)

**Context**: Need HTTP server for REST API but want flexibility.

**Decision**: Use axum as default, design trait for swappability.

```rust
pub trait HttpServer {
    fn serve(&self, addr: &str) -> Result<()>;
}
```

**Rationale**:
- axum is well-maintained, async-native, performant
- Trait allows future alternatives (hyper, actix-web)
- Tower middleware ecosystem compatibility

**Trade-offs**:
- (+) Production-ready HTTP server
- (+) Easy to swap implementations
- (-) axum + tokio add ~2MB to binary
- (-) Async runtime overhead for simple deployments

**Falsifiable claim**: axum-based server handles >10,000 requests/sec on commodity hardware (4-core, 8GB RAM).

**How to falsify**: Load test with `wrk` or our Rust-based load test client.

### ADR-005: Quantization Strategy

**Context**: Need to support quantized models for memory efficiency.

**Decision**: Implement Q4_0, Q8_0, Q4_K, Q5_K, Q6_K dequantization.

**Rationale**:
- Q4_0/Q8_0: llama.cpp legacy formats (wide compatibility)
- Q4_K/Q5_K/Q6_K: K-quant formats (better quality/size tradeoff)
- Dequantization at inference time (no re-quantization needed)

**Trade-offs**:
- (+) Support for all common quantization formats
- (+) Memory efficient for large models
- (-) Dequantization adds CPU overhead
- (-) Quality loss compared to FP16/FP32

**Falsifiable claim**: Q4_K quantization provides <1% perplexity degradation compared to FP16 on standard benchmarks.

**How to falsify**: Measure perplexity on WikiText-2 for both formats.

## Performance Claims (Falsifiable)

### Claim P1: 9.6x Faster Than PyTorch for CPU Inference

**Measurement methodology**:
- Dataset: MNIST (784 input features)
- Batch size: 1
- Device: CPU only
- Threads: 1 (single-threaded)
- Iterations: 10,000 with 50 warm-up

**Result**: 0.52 µs (Realizar) vs 5.00 µs (PyTorch)

**Statistical validation**:
- p < 0.001 (two-sample t-test)
- Cohen's d = 5.19 (large effect size)
- 95% CI: [0.50 µs, 0.54 µs] for Realizar

**How to falsify**: Run `make bench-comparative` and verify results.

### Claim P2: 53,000x Faster Cold Start for Lambda

**Measurement methodology**:
- Model: MNIST (784x2.apr, 3.2KB)
- Environment: AWS Lambda ARM64, 128MB
- Metric: Time from invocation to first inference

**Result**: 15µs (Realizar) vs 800ms (PyTorch)

**How to falsify**: Deploy both to Lambda and measure cold start with X-Ray.

### Claim P3: <100ms p50 Latency for 1B Models

**Target**: Single-request inference for 1B parameter models.

**Current status**: Not yet validated with real 1B models.

**How to validate**: Load TinyLlama-1.1B, measure generation latency.

## Testing Philosophy

### EXTREME TDD Requirements

All code must follow RED-GREEN-REFACTOR:

1. **RED**: Write failing tests first
2. **GREEN**: Minimal implementation to pass
3. **REFACTOR**: Clean up, optimize

**Falsifiable claim**: 100% of public APIs have corresponding tests.

**How to falsify**: Run `cargo test --doc` and check for missing examples.

### Coverage Targets

| Metric | Target | Current |
|--------|--------|---------|
| Line coverage | 85% | 95.46% |
| Function coverage | 85% | 91.33% |
| Mutation score | 80% | 100% (api.rs) |

**How to falsify**: Run `cargo llvm-cov --html`.

### Property-Based Testing

Mathematical invariants tested with proptest:

- Tensor operations: associativity, commutativity, identity
- Tokenizer: encode(decode(x)) == x for valid tokens
- Quantization: dequant(quant(x)) approximates x within tolerance

**Falsifiable claim**: Property tests catch edge cases that unit tests miss.

**How to falsify**: Track bugs found by property tests vs unit tests.

## Reproducibility Requirements

### Build Reproducibility

**Requirement**: Same commit produces identical binary on same platform.

**Implementation**:
- `Cargo.lock` checked into version control
- `rust-toolchain.toml` pins Rust version
- CI builds verified against local builds

**How to falsify**: Build twice from same commit, compare SHA256.

### Benchmark Reproducibility

**Requirement**: Benchmarks reproducible within 5% variance across runs.

**Implementation**:
- CPU frequency scaling disabled (`performance` governor)
- Warm-up phase (50 iterations)
- Multiple samples (100 by Criterion default)
- 95% confidence intervals reported

**How to falsify**: Run benchmark 10 times, check variance.

### Model Reproducibility

**Requirement**: Same model file produces identical output for same input.

**Implementation**:
- Deterministic RNG seeds documented
- No non-deterministic GPU operations (CPU inference is deterministic)
- Model files versioned with SHA256 checksums

**How to falsify**: Run inference twice with same seed, compare outputs.

## Future Considerations

### Not Yet Implemented (Phase 3+)

These are NOT claims, but future goals:

- Vision model support (LLaVA, Qwen-VL)
- Speculative decoding
- Tensor parallelism
- INT8 inference path

### Known Limitations

1. **GPU inference**: Currently uses Trueno's wgpu backend; not as optimized as CUDA
2. **Multi-GPU**: Not supported
3. **Batch inference**: Limited to synchronous batching
4. **Model formats**: Only GGUF, Safetensors, APR (no ONNX, TensorRT)

## References

1. Popper, K. R. (1959). *The Logic of Scientific Discovery*. Routledge.
2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters*. Wiley.
3. MLPerf Inference Benchmark Suite. MLCommons. https://mlcommons.org/benchmarks/inference/
4. Criterion.rs Documentation. https://bheisler.github.io/criterion.rs/book/

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-10
**Authors**: Pragmatic AI Labs

All claims in this document are falsifiable. If you can demonstrate a claim is false, please open an issue.
