# Trueno - High-Performance Compute Library

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [First Program](./getting-started/first-program.md)
- [Core Concepts](./getting-started/core-concepts.md)

# Architecture

- [Overview](./architecture/overview.md)
- [ComputeBrick Architecture](./architecture/compute-brick.md)
- [Backend Selection](./architecture/backend-selection.md)
- [Multi-Backend Design](./architecture/multi-backend-design.md)
- [SIMD Backends](./architecture/simd-backends.md)
  - [SSE2 (x86_64 Baseline)](./architecture/sse2-backend.md)
  - [AVX/AVX2 (256-bit)](./architecture/avx-backend.md)
  - [AVX-512 (512-bit)](./architecture/avx512-backend.md)
  - [NEON (ARM64)](./architecture/neon-backend.md)
  - [WASM SIMD128](./architecture/wasm-backend.md)
- [GPU Backend](./architecture/gpu-backend.md)
- [PTX Code Generation](./architecture/ptx-generation.md)
- [PTX Register Allocation](./architecture/ptx-register-allocation.md)
- [PTX Optimization Passes](./architecture/ptx-optimization.md)
- [Runtime Detection](./architecture/runtime-detection.md)

# API Reference

- [Vector Operations](./api-reference/vector-operations.md)
- [Matrix Operations](./api-reference/matrix-operations.md)
- [Eigendecomposition](./api-reference/eigendecomposition.md)
- [Element-wise Operations](./api-reference/element-wise.md)
- [Reduction Operations](./api-reference/reductions.md)
- [Transformation Operations](./api-reference/transformations.md)
- [Error Handling](./api-reference/error-handling.md)
- [Backend API](./api-reference/backend-api.md)
- [GPU Monitoring](./api-reference/gpu-monitoring.md)
- [Hash Functions](./api-reference/hash-functions.md)
- [LZ4 Compression](./api-reference/lz4-compression.md)

# Performance

- [Benchmarks Overview](./performance/benchmarks.md)
- [SIMD Performance Analysis](./performance/simd-performance.md)
- [GPU Performance](./performance/gpu-performance.md)
- [Optimization Guide](./performance/optimization-guide.md)
- [Profiling](./performance/profiling.md)
- [BrickProfiler Contract Compliance](./performance/brick-profiler-contract.md)
- [Model-Level Inference Tracing](./performance/model-tracing.md)
- [ML Tuner: Learned Kernel Selection](./performance/ml-tuner.md)
- [Golden Trace Validation](./performance/golden-trace-validation.md)
- [Performance Targets](./performance/targets.md)
- [Comparing Backends](./performance/backend-comparison.md)

# Safety and Correctness

- [Safety Philosophy](./safety/philosophy.md)
- [Unsafe Code Guidelines](./safety/unsafe-code.md)
- [Safety Invariants](./safety/safety-invariants.md)
- [Miri Validation](./safety/miri-validation.md)
- [Testing for Correctness](./safety/testing-correctness.md)
- [Backend Equivalence](./safety/backend-equivalence.md)

# Examples

- [Vector Math](./examples/vector-math.md)
- [Matrix Operations](./examples/matrix-operations.md)
- [Neural Networks](./examples/neural-networks.md)
- [Image Processing](./examples/image-processing.md)
- [Signal Processing](./examples/signal-processing.md)
- [Scientific Computing](./examples/scientific-computing.md)
- [Execution Path Graph](./examples/execution-graph.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [Extreme TDD](./development/extreme-tdd.md)
- [Testing](./development/testing.md)
  - [Unit Tests](./development/unit-tests.md)
  - [Property-Based Tests](./development/property-based-tests.md)
  - [Backend Equivalence Tests](./development/backend-equivalence-tests.md)
  - [Mutation Testing](./development/mutation-testing.md)
- [Benchmarking](./development/benchmarking.md)
- [Quality Gates](./development/quality-gates.md)
- [SATD Remediation](./development/satd-remediation.md)
- [PTX Best Practices](./development/ptx-best-practices.md)
- [PTX Bug Detection](./development/ptx-bug-detection.md)
- [Code Review Checklist](./development/code-review.md)

# Advanced Topics

- [SIMD Intrinsics](./advanced/simd-intrinsics.md)
- [BLIS-Style Matrix Multiplication](./advanced/blis-gemm.md)
- [Phase 2 Micro-Kernel](./advanced/phase2-microkernel.md)
- [Phase 15 Fused Q4K Kernels](./advanced/phase15-fused-q4k.md)
- [Tiling Compute Blocks](./advanced/tiling-compute-blocks.md)
- [GPU Compute Shaders](./advanced/gpu-shaders.md)
- [Memory Alignment](./advanced/memory-alignment.md)
- [Vectorization Patterns](./advanced/vectorization-patterns.md)
- [Cross-Platform Portability](./advanced/portability.md)
- [FFmpeg Case Study](./advanced/ffmpeg-case-study.md)

# Ecosystem Integration

- [Ruchy Integration](./ecosystem/ruchy.md)
- [Depyler (Python → Rust)](./ecosystem/depyler.md)
- [Decy (C → Rust)](./ecosystem/decy.md)
- [AWS Lambda (ruchy-lambda)](./ecosystem/ruchy-lambda.md)
- [Docker Benchmarking](./ecosystem/ruchy-docker.md)
- [PMAT Quality Gates](./ecosystem/pmat.md)
- [cbtop (Compute Block Top)](./ecosystem/cbtop.md)
- [aprender (ML Integration)](./ecosystem/aprender.md)
- [trueno-cuda-edge (GPU Edge Testing)](./ecosystem/trueno-cuda-edge.md)

# Specifications

- [Design Philosophy](./specifications/design-philosophy.md)
- [Initial Three-Target Spec](./specifications/three-target-spec.md)
- [Ecosystem Positioning (NumPy-like)](./specifications/pytorch-numpy-spec.md)
- [Ruchy Language Support](./specifications/ruchy-support.md)
- [Quality Gates (TRUENO-SPEC-013)](./specifications/solidify-quality-spec.md)
- [Academic Foundations](./specifications/academic-foundations.md)
- [CUDA Library Parity](./specifications/cuda-parity.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Migration Guide](./appendix/migration-guide.md)
- [Performance Comparison Tables](./appendix/performance-tables.md)
