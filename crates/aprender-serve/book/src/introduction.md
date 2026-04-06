# Introduction

Welcome to **Realizar** - a comprehensive guide to building a production-grade ML inference engine in pure Rust, from absolute scratch. This book documents every design decision, implementation detail, and testing strategy used to create a zero-dependency model serving system with EXTREME TDD methodology.

## What is Realizar?

**Realizar** (Spanish: "to accomplish, to achieve") is a pure Rust ML inference engine built **100% from scratch**:

- ✅ **GGUF parser** - Binary format reader (no llama.cpp)
- ✅ **Safetensors parser** - Zero-copy tensor loader (no HuggingFace hub)
- ✅ **Transformer architecture** - Attention, FFN, RoPE, all our code
- ✅ **Quantization** - Q4_0, Q8_0 algorithms from scratch
- ✅ **Tokenizers** - BPE and SentencePiece native implementations
- ✅ **REST API** - Production HTTP server with Axum
- ✅ **AWS Lambda** - Serverless deployment with <50ms cold start
- ✅ **Multi-Target** - Native, Lambda, Docker, WASM deployment
- ✅ **Aprender Serving** - Native .apr model format support
- ✅ **GPU acceleration** - Via Trueno (our SIMD/GPU library)
- ❌ **NO candle** - We build our own compute primitives
- ❌ **NO llama.cpp bindings** - Pure Rust, total control
- ❌ **NO external ML deps** - Only HTTP infrastructure (swappable)

## What You'll Learn

This book is your complete guide to building ML infrastructure from first principles:

- **Model Format Parsing**: How to read GGUF and Safetensors binary formats
- **Transformer Implementation**: Multi-head attention, RoPE, SwiGLU, RMSNorm from scratch
- **Quantization Algorithms**: 4-bit and 8-bit weight compression and dequantization
- **Tokenization**: BPE and SentencePiece algorithms without external libraries
- **Text Generation**: Greedy, top-k, top-p sampling strategies
- **REST API Design**: Production HTTP server with swappable backends
- **GPU Acceleration**: Leveraging Trueno for SIMD and GPU dispatch
- **EXTREME TDD**: Applying rigorous test-driven development to ML systems

## Why Build from Scratch?

| Using Existing Libraries | Building from Scratch (Realizar) |
|--------------------------|-----------------------------------|
| Quick start | **Total control** over every line of code |
| Black box internals | **Deep understanding** of how it works |
| Hidden dependencies | **Zero external ML dependencies** |
| Opaque errors | **Debuggable** - it's all your code |
| Library limitations | **Customizable** - change anything |
| Trust maintainers | **Own your stack** - no supply chain risk |

**Philosophy:** Own your stack from bottom to top. When you build it yourself, you understand it completely, can debug any issue, and optimize without barriers.

## The Philosophy

> **"Build everything ourselves except HTTP infrastructure. Own the ML stack completely."**

Realizar is built on these core principles:

1. **Pure Rust** - Memory safe, zero garbage collection, blazing fast
2. **From Scratch** - Every ML component written by us (GGUF, transformers, quantization)
3. **Swappable Infrastructure** - HTTP server is a trait (Axum default, can swap to hyper/actix-web)
4. **Trueno-Backed** - Use our own SIMD/GPU library for compute primitives
5. **EXTREME TDD** - 95.09% line coverage, 100% mutation score on API, zero warnings
6. **Zero Tolerance** - All tests pass, all examples run, always

## Real-World Results

Phase 1 is **complete** with exceptional quality metrics:

- **14,614 passing tests** (unit + property + integration + CUDA)
- **95.09% line coverage**, 95.91% function coverage, 95.13% region coverage
- **1,196 CUDA/GPU tests** passing (RTX 4090)
- **146.5/134 Rust Project Score** (Grade A+, exceeds maximum!)
- **100% mutation score** on api.rs (18/18 viable mutants caught)
- **6 benchmark suites** (tensor_ops, inference, cache, tokenizer, quantize, lambda)
- **414 examples** (37 documented + 377 development scripts)
- **Zero clippy warnings** enforced
- **Zero dead code** detected
- **Blazing performance**: 504µs for 5-token generation, 35-676ns Lambda warm inference

## Performance Benchmarks

Realizar achieves sub-millisecond inference:

- **Forward pass (1 token)**: ~17.5µs
- **5-token generation**: ~504µs (<1ms target ✅)
- **10-token generation**: ~1.54ms
- **20-token generation**: ~5.52ms

All benchmarks run on CPU with Trueno SIMD acceleration.

## How This Book is Organized

### Part 1: Core Architecture (Chapters 1-5)
Foundation and design philosophy, feature flags, Trueno integration.

### Part 2: Model Formats (Chapters 6-7)
GGUF and Safetensors binary format parsing from scratch.

### Part 3: Quantization (Chapter 8)
Q4_0 and Q8_0 compression algorithms, block structures, dequantization.

### Part 4: Transformer Architecture (Chapters 9-10)
Multi-head attention, RoPE, feed-forward networks, KV cache, all layer components.

### Part 5: Tokenization (Chapter 11)
BPE and SentencePiece algorithms without external dependencies.

### Part 6: Text Generation (Chapter 12)
Inference loop, sampling strategies (greedy, top-k, top-p), temperature scaling.

### Part 7: REST API & CLI (Chapters 13-14)
Axum-based HTTP server, swappable backend trait, CLI binary with clap.

### Part 8: GPU Acceleration (Chapter 15)
Trueno backend, SIMD optimization, GPU dispatch strategy.

### Part 9: EXTREME TDD (Chapter 16)
RED-GREEN-REFACTOR applied to ML, property-based testing, mutation testing.

### Part 10: Development Phases (Chapter 17)
Phase 1 (complete), Phase 2 (optimization), Phase 3 (advanced models), Phase 4 (production).

### Part 11: Quality, Performance & Examples (Chapters 18-20)
Quality gates, benchmarking, real working examples.

### Part 12: Tools & Best Practices (Chapters 21-22)
Development environment, cargo tools, best practices, design decisions.

### Part 13: Appendix
Glossary, specifications, papers, contributing guidelines.

## Who This Book is For

- **ML Engineers** wanting to understand model serving internals
- **Systems Programmers** building high-performance Rust applications
- **Library Authors** seeking to build from-scratch ML infrastructure
- **Students** learning transformer architectures and quantization
- **Teams** adopting EXTREME TDD for zero-defect development

## Anti-Hallucination Guarantee

Every code example in this book is:
- ✅ **Test-backed** - Validated by 14,614 passing tests in realizar
- ✅ **CI-verified** - Automatically tested in GitHub Actions
- ✅ **Production-proven** - From a real, deployed codebase
- ✅ **Reproducible** - Clone the repo, run tests, see results

**If an example cannot be validated by tests, it will not appear in this book.**

## Getting Started

Ready to build an ML inference engine from scratch? Start with:

1. [Design Philosophy](./architecture/design-philosophy.md) - Why we build this way
2. [Project Structure](./architecture/project-structure.md) - How code is organized
3. [GGUF Format](./formats/gguf.md) - Parsing model files
4. [Transformer Architecture](./transformer/overview.md) - Core inference engine

Or jump straight to:
- [Development Environment](./tools/development-environment.md) - Get set up
- [Phase 1: Core Inference](./phases/phase1.md) - What we've built
- [Inference Example](./examples/inference.md) - See it in action

## Contributing to This Book

This book is open source and welcomes contributions. See [Contributing to This Book](./appendix/contributing.md) for guidelines.

All book content follows the same EXTREME TDD principles:
- Every example must be test-backed
- All code must compile and run
- Zero tolerance for hallucinated examples
- Continuous improvement through Kaizen

## The Journey Ahead

Building an ML inference engine from scratch is challenging but deeply rewarding. You will:

- **Understand** transformer architectures at the byte level
- **Master** binary format parsing (GGUF, Safetensors)
- **Implement** quantization algorithms from papers
- **Build** production HTTP servers with Rust
- **Apply** EXTREME TDD to complex systems
- **Own** your entire ML serving stack

**Phase 1 is complete.** Let's explore how we built it, why we made each decision, and what's next in Phases 2-4.

---

**Let's build ML infrastructure from scratch. Let's master Realizar.**
