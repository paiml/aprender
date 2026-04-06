# Trueno-DB Guide

[Introduction](./introduction.md)

# Architecture

- [System Overview](./architecture/system-overview.md)
- [Design Principles](./architecture/design-principles.md)
- [Cost-Based Backend Selection](./architecture/cost-based-backend.md)
- [Out-of-Core Execution](./architecture/out-of-core-execution.md)
- [Heterogeneous Computing](./architecture/heterogeneous-computing.md)

# Core Components

## Storage Layer

- [Arrow Storage Backend](./components/storage/arrow-backend.md)
- [Parquet Integration](./components/storage/parquet-integration.md)
- [Morsel-Driven Parallelism](./components/storage/morsel-driven.md)
- [GPU Transfer Queue](./components/storage/gpu-transfer-queue.md)

## Backend Dispatcher

- [Backend Selection Algorithm](./components/dispatcher/selection-algorithm.md)
- [Physics-Based Cost Model](./components/dispatcher/cost-model.md)
- [5x Transfer Rule](./components/dispatcher/5x-rule.md)
- [Performance Characteristics](./components/dispatcher/performance.md)

## Query Engine

- [SQL Query Interface](./components/query/sql-interface.md)
- [JIT WGSL Compiler](./components/query/jit-compiler.md)
- [Kernel Fusion](./components/query/kernel-fusion.md)
- [Operator Variants](./components/query/operator-variants.md)

## GPU Kernels

- [Parallel Reduction](./components/gpu/parallel-reduction.md)
- [Aggregations](./components/gpu/aggregations.md)
- [Hash Join](./components/gpu/hash-join.md)
- [Memory Management](./components/gpu/memory-management.md)

## SIMD Fallback

- [Trueno Integration](./components/simd/trueno-integration.md)
- [SIMD Primitives](./components/simd/simd-primitives.md)
- [CPU Optimization](./components/simd/cpu-optimization.md)

## Experiment Tracking

- [Experiment Schema](./components/experiment/schema.md)

## Key-Value Store (Phase 6)

- [KV Store Overview](./components/kv/overview.md)
- [Hash Functions](./components/kv/hash-functions.md)
- [Compression (GH-5)](./components/kv/compression.md)
- [pforge Integration](./components/kv/pforge-integration.md)

# EXTREME TDD Methodology

- [RED-GREEN-REFACTOR Cycle](./tdd/red-green-refactor.md)
- [Test-First Philosophy](./tdd/test-first.md)
- [Property-Based Testing](./tdd/property-based-testing.md)
- [Integration Testing](./tdd/integration-testing.md)
- [Backend Equivalence Tests](./tdd/backend-equivalence.md)

# Toyota Way Principles

- [Poka-Yoke (Mistake Proofing)](./toyota/poka-yoke.md)
- [Genchi Genbutsu (Go and See)](./toyota/genchi-genbutsu.md)
- [Muda (Waste Elimination)](./toyota/muda.md)
- [Jidoka (Built-in Quality)](./toyota/jidoka.md)
- [Heijunka (Load Leveling)](./toyota/heijunka.md)
- [Kaizen (Continuous Improvement)](./toyota/kaizen.md)

# Quality Gates

- [TDG Score (≥85)](./quality/tdg-score.md)
- [Code Coverage (>90%)](./quality/code-coverage.md)
- [Mutation Testing (≥80%)](./quality/mutation-testing.md)
- [Clippy Linting](./quality/clippy.md)
- [Continuous Integration](./quality/ci.md)

# Academic Foundation

- [Research Papers](./academic/research-papers.md)
- [Morsel-Driven Parallelism (Leis 2014)](./academic/leis-2014.md)
- [GPU Paging (Funke 2018)](./academic/funke-2018.md)
- [PCIe Bottlenecks (Gregg 2011)](./academic/gregg-2011.md)
- [Heterogeneous Query Processing (Breß 2014)](./academic/bress-2014.md)
- [JIT Compilation (Neumann 2011)](./academic/neumann-2011.md)
- [Kernel Fusion (Wu 2012)](./academic/wu-2012.md)

# Development Guide

- [Getting Started](./dev/getting-started.md)
- [Building from Source](./dev/building.md)
- [Running Tests](./dev/running-tests.md)
- [Examples](./dev/examples.md)
- [Contributing](./dev/contributing.md)
- [Roadmap](./dev/roadmap.md)

# Milestones

- [Phase 1 MVP: Complete](./milestones/phase1-mvp.md)

# Case Studies

- [CORE-001: Arrow Storage Backend](./case-studies/core-001.md)
- [CORE-002: Cost-Based Dispatcher](./case-studies/core-002.md)
- [Property-Based Testing Morsels](./case-studies/proptest-morsels.md)
- [Integration Testing Pipeline](./case-studies/integration-pipeline.md)

# Performance

- [Benchmarking Methodology](./performance/benchmarking.md)
- [Competitive Benchmarks](./performance/competitive-benchmarks.md)
- [GPU vs SIMD vs Scalar](./performance/backend-comparison.md)
- [GPU Syscall Tracing](./performance/gpu-syscall-tracing.md)
- [Scalability Analysis](./performance/scalability.md)
- [Optimization Techniques](./performance/optimization.md)

# Troubleshooting

- [Common Issues](./troubleshooting/common-issues.md)
- [GPU Setup](./troubleshooting/gpu-setup.md)
- [WebGPU Browser Setup](./troubleshooting/webgpu-browser.md)
- [Debugging Tips](./troubleshooting/debugging.md)
- [Performance Debugging](./troubleshooting/performance-debugging.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [API Documentation](./appendix/api-docs.md)
- [License](./appendix/license.md)
