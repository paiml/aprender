# Realizar - Pure Rust ML Inference Engine

[Introduction](./introduction.md)

# Core Architecture

- [Design Philosophy](./architecture/design-philosophy.md)
- [Project Structure](./architecture/project-structure.md)
- [Zero Dependencies Strategy](./architecture/zero-dependencies.md)
- [Trueno Integration](./architecture/trueno-integration.md)
- [Feature Flags and Modularity](./architecture/feature-flags.md)

# Model Formats

- [Overview](./formats/overview.md)
- [GGUF Format](./formats/gguf.md)
  - [Binary Structure](./formats/gguf-binary-structure.md)
  - [Metadata Parsing](./formats/gguf-metadata.md)
  - [Tensor Information](./formats/gguf-tensors.md)
  - [Value Types](./formats/gguf-value-types.md)
  - [Array Support](./formats/gguf-arrays.md)
- [Safetensors Format](./formats/safetensors.md)
  - [JSON Header](./formats/safetensors-header.md)
  - [Zero-Copy Loading](./formats/safetensors-zero-copy.md)
  - [Tensor Data Access](./formats/safetensors-access.md)

# Quantization

- [What is Quantization?](./quantization/what-is-quantization.md)
- [Q4_0 Format](./quantization/q4-0.md)
  - [4-bit Encoding](./quantization/q4-0-encoding.md)
  - [Block Structure](./quantization/q4-0-blocks.md)
  - [Dequantization](./quantization/q4-0-dequantize.md)
- [Q8_0 Format](./quantization/q8-0.md)
  - [8-bit Encoding](./quantization/q8-0-encoding.md)
  - [Precision vs Size Tradeoffs](./quantization/q8-0-tradeoffs.md)
- [Advanced Quantization (Phase 2)](./quantization/advanced.md)
  - [Q4_K, Q5_K, Q6_K](./quantization/k-quants.md)

# Transformer Architecture

- [Overview](./transformer/overview.md)
- [Tensor Abstraction](./transformer/tensor.md)
- [Layer Components](./transformer/layers.md)
  - [Multi-Head Attention](./transformer/attention.md)
  - [RoPE Position Embeddings](./transformer/rope.md)
  - [Feed-Forward Networks](./transformer/ffn.md)
  - [SwiGLU Activation](./transformer/swiglu.md)
  - [RMSNorm](./transformer/rmsnorm.md)
  - [LayerNorm](./transformer/layernorm.md)
  - [Softmax](./transformer/softmax.md)
  - [GELU](./transformer/gelu.md)
- [KV Cache Management](./transformer/kv-cache.md)
- [Model Configuration](./transformer/config.md)
- [Forward Pass](./transformer/forward-pass.md)

# Tokenization

- [Overview](./tokenization/overview.md)
- [Vocabulary Management](./tokenization/vocabulary.md)
- [Character-Level Tokenizer](./tokenization/character-level.md)
- [BPE (Byte Pair Encoding)](./tokenization/bpe.md)
  - [BPE Algorithm](./tokenization/bpe-algorithm.md)
  - [Merge Rules](./tokenization/bpe-merges.md)
  - [Encoding Process](./tokenization/bpe-encoding.md)
  - [Decoding Process](./tokenization/bpe-decoding.md)
- [SentencePiece](./tokenization/sentencepiece.md)
  - [Unigram Model](./tokenization/sentencepiece-unigram.md)
  - [Special Tokens](./tokenization/sentencepiece-special-tokens.md)
  - [Score-Based Selection](./tokenization/sentencepiece-scores.md)

# Text Generation

- [Inference Loop](./generation/inference-loop.md)
- [Sampling Strategies](./generation/sampling.md)
  - [Greedy Sampling](./generation/greedy.md)
  - [Top-k Sampling](./generation/top-k.md)
  - [Top-p (Nucleus) Sampling](./generation/top-p.md)
  - [Temperature Scaling](./generation/temperature.md)
- [Generation Parameters](./generation/parameters.md)
- [Streaming (Phase 2)](./generation/streaming.md)

# REST API with Axum

- [Server Architecture](./api/architecture.md)
- [HTTP Framework Selection](./api/framework-selection.md)
- [Swappable Server Trait](./api/swappable-server.md)
- [Endpoints](./api/endpoints.md)
  - [Health Check](./api/health.md)
  - [Tokenize](./api/tokenize.md)
  - [Generate](./api/generate.md)
- [Request/Response Types](./api/types.md)
- [Error Handling](./api/error-handling.md)
- [Demo Mode](./api/demo-mode.md)
- [Testing HTTP Endpoints](./api/testing.md)

# CLI Binary

- [Command Structure](./cli/command-structure.md)
- [Serve Command](./cli/serve.md)
- [Bench Command](./cli/bench.md)
- [Viz Command](./cli/viz.md)
- [Info Command](./cli/info.md)
- [CLI Testing with assert_cmd](./cli/testing.md)

# Aprender Model Serving

- [Overview](./aprender/overview.md)
- [The .apr Format](./aprender/apr-format.md)
- [HTTP API for Aprender](./aprender/http-api.md)
- [Metrics and Model Evaluation](./aprender/metrics.md)
- [Drift Detection and Retraining](./aprender/drift.md)
- [Performance Targets](./aprender/performance.md)

# AWS Lambda

- [Overview](./lambda/overview.md)
- [MNIST Benchmark: .apr vs PyTorch](./lambda/mnist-benchmark.md)
- [Lambda Handler](./lambda/handler.md)
- [Batch Inference](./lambda/batch.md)
- [Metrics & Observability](./lambda/metrics.md)
- [ARM64 Graviton Optimization](./lambda/arm64.md)
- [Cold Start Optimization](./lambda/cold-start.md)

# Mixture-of-Experts (MOE)

- [Overview](./moe/overview.md)
- [Capacity Factor Routing](./moe/capacity-routing.md)
- [Andon Triggers (Jidoka)](./moe/andon.md)
- [A/B Testing Statistics](./moe/ab-testing.md)
- [Memory Pinning (mlock)](./moe/memory-pinning.md)
- [Load Testing](./moe/load-testing.md)
- [Reproducible Benchmarks](./moe/benchmarks.md)

# GPU Acceleration

- [Trueno Backend](./gpu/trueno-backend.md)
- [CUDA PTX Generation](./gpu/cuda-ptx.md)
- [SIMD Optimization](./gpu/simd.md)
- [GPU Dispatch Strategy](./gpu/dispatch-strategy.md)
- [BrickProfiler Decode Profiling](./gpu/brick-profiler-decode.md)
- [Memory Management](./gpu/memory-management.md)
- [CUDA Context Safety](./gpu/cuda-context-safety.md)
- [Simulation Research Findings](./gpu/simulation-research.md)
- [trueno-cuda-edge (Edge Testing)](./gpu/trueno-cuda-edge.md)

# EXTREME TDD for ML Serving

- [Applying RED-GREEN-REFACTOR](./tdd/red-green-refactor.md)
- [Unit Testing ML Code](./tdd/unit-testing.md)
- [Property-Based Testing](./tdd/property-based.md)
  - [Tensor Properties](./tdd/tensor-properties.md)
  - [Tokenization Properties](./tdd/tokenization-properties.md)
  - [Quantization Properties](./tdd/quantization-properties.md)
- [Mutation Testing](./tdd/mutation-testing.md)
  - [API Mutation Score (100%)](./tdd/api-mutation.md)
- [Integration Testing](./tdd/integration-testing.md)
- [Benchmark Testing](./tdd/benchmark-testing.md)

# Development Phases

- [Phase 1: Core Inference (Complete)](./phases/phase1.md)
  - [GGUF Parser](./phases/phase1-gguf.md)
  - [Safetensors Parser](./phases/phase1-safetensors.md)
  - [Transformer Implementation](./phases/phase1-transformer.md)
  - [Quantization Q4_0/Q8_0](./phases/phase1-quantization.md)
  - [Tokenizers](./phases/phase1-tokenizers.md)
  - [HTTP Server](./phases/phase1-server.md)
  - [CLI Binary](./phases/phase1-cli.md)
  - [Quality Metrics](./phases/phase1-metrics.md)
- [Phase 2: Optimization](./phases/phase2.md)
  - [Advanced Quantization](./phases/phase2-quantization.md)
  - [Flash Attention](./phases/phase2-flash-attention.md)
  - [Batch Inference](./phases/phase2-batch.md)
  - [Streaming Responses](./phases/phase2-streaming.md)
  - [Model Caching](./phases/phase2-caching.md)
- [Phase 3: Advanced Models](./phases/phase3.md)
  - [Multi-Query Attention (MQA)](./phases/phase3-mqa.md)
  - [Grouped-Query Attention (GQA)](./phases/phase3-gqa.md)
  - [Vision Models](./phases/phase3-vision.md)
- [Phase 4: Production](./phases/phase4.md)
  - [Multi-Model Serving](./phases/phase4-multi-model.md)
  - [Request Batching](./phases/phase4-batching.md)
  - [Monitoring & Metrics](./phases/phase4-monitoring.md)

# Quality Gates

- [Pre-Commit Checks](./quality/pre-commit.md)
- [Code Formatting (rustfmt)](./quality/formatting.md)
- [Linting (clippy)](./quality/linting.md)
- [Code Coverage (>95%)](./quality/coverage.md)
- [Mutation Testing](./quality/mutation.md)
- [TDG Score](./quality/tdg.md)
- [Rust Project Score (132.9/134 A+)](./quality/rust-project-score.md)
- [Continuous Integration](./quality/ci.md)
- [Provable Contracts](./quality/contracts.md)

# Performance

- [Benchmarking Strategy](./performance/benchmarking.md)
- [Tensor Operations](./performance/tensor-ops.md)
- [Inference Benchmarks](./performance/inference.md)
- [Generation Speed](./performance/generation-speed.md)
  - [1-token: ~17.5µs](./performance/single-token.md)
  - [5-token: ~504µs](./performance/five-tokens.md)
  - [10-token: ~1.54ms](./performance/ten-tokens.md)
- [Memory Usage](./performance/memory.md)
- [Profiling with Renacer](./performance/profiling.md)

# Real-World Examples

- [Examples Reference](./examples/examples-reference.md)
- [Quick Generate (Real Models)](./examples/quick-generate.md)
- [Inference Demo](./examples/inference.md)
- [TinyLlama Benchmark](./examples/tinyllama-benchmark.md)
- [Tokenization Comparison](./examples/tokenization.md)
- [Chat Templates](./examples/chat-template.md)
- [API Server Demo](./examples/api-server.md)
- [Observability Demo](./examples/observability.md)
- [Wine Quality Lambda](./examples/wine-lambda.md)
- [Data Pipeline (Alimentar)](./examples/data-pipeline.md)
- [MNIST Examples](./examples/mnist.md)
- [Custom Model Loading (Phase 2)](./examples/custom-model.md)
- [GPU & Performance Parity](./examples/gpu-parity.md)
- [Pipeline TUI](./examples/pipeline-tui.md)

# Tools and Setup

- [Development Environment](./tools/development-environment.md)
- [Installing Rust](./tools/installing-rust.md)
- [Installing mdbook](./tools/installing-mdbook.md)
- [cargo test](./tools/cargo-test.md)
- [cargo clippy](./tools/cargo-clippy.md)
- [cargo fmt](./tools/cargo-fmt.md)
- [cargo llvm-cov](./tools/cargo-llvm-cov.md)
- [cargo mutants](./tools/cargo-mutants.md)
- [cargo bench](./tools/cargo-bench.md)
- [proptest](./tools/proptest.md)
- [pmat](./tools/pmat.md)
- [bashrs](./tools/bashrs.md)

# Best Practices

- [Error Handling with thiserror](./best-practices/error-handling.md)
- [API Design](./best-practices/api-design.md)
- [Type Safety](./best-practices/type-safety.md)
- [Memory Safety](./best-practices/memory-safety.md)
- [Documentation Standards](./best-practices/documentation-standards.md)
- [Zero Unsafe Code](./best-practices/zero-unsafe.md)
- [Minimal Dependencies](./best-practices/minimal-dependencies.md)

# Design Decisions

- [Why Pure Rust?](./decisions/why-pure-rust.md)
- [Why Build from Scratch?](./decisions/why-from-scratch.md)
- [Why Axum for HTTP?](./decisions/why-axum.md)
- [Why Trueno for Compute?](./decisions/why-trueno.md)
- [Why No llama.cpp/candle/etc?](./decisions/why-no-existing-libs.md)
- [Swappable HTTP Backend](./decisions/swappable-http.md)

# Production Deployment

- [Multi-Target Deployment](./deployment/multi-target.md)
- [Docker Deployment](./deployment/docker.md)
  - [Docker Configuration](./deployment/docker-config.md)
- [Kubernetes](./deployment/kubernetes.md)
- [WebAssembly (WASM)](./deployment/wasm.md)
- [Monitoring & Observability](./deployment/monitoring.md)
- [Load Testing](./deployment/load-testing.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [GGUF Specification](./appendix/gguf-spec.md)
- [Safetensors Specification](./appendix/safetensors-spec.md)
- [Transformer Papers](./appendix/transformer-papers.md)
- [Contributing to This Book](./appendix/contributing.md)
