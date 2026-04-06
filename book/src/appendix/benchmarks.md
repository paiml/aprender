# Appendix F: Performance Benchmarks

This appendix presents benchmark data for transpilation speed, runtime performance comparisons between Python and Rust, and memory usage across the Sovereign AI Stack.

## Transpilation Speed

Time to transpile source code to Rust, measured on a 24-core AMD EPYC system:

| Source | Files | Lines | Transpile Time | Lines/sec |
|--------|-------|-------|----------------|-----------|
| Python (pure functions) | 50 | 5,000 | 1.2s | 4,167 |
| Python (ML with numpy) | 120 | 25,000 | 8.4s | 2,976 |
| C (systems code) | 30 | 12,000 | 3.1s | 3,871 |
| Shell scripts | 15 | 2,000 | 0.6s | 3,333 |
| Mixed (Python + C + Shell) | 200 | 40,000 | 12.8s | 3,125 |

Transpilation is I/O-bound for small projects and CPU-bound for large ones. Files within a language group are transpiled in parallel.

## Runtime Performance: Python vs Rust

Benchmarks comparing original Python code against transpiled and optimized Rust code:

### Compute-Intensive Workloads

| Workload | Python | Rust (scalar) | Rust (SIMD) | Rust (GPU) |
|----------|--------|---------------|-------------|------------|
| Matrix multiply 1024x1024 | 2,400 ms | 85 ms (28x) | 12 ms (200x) | 2.1 ms (1,143x) |
| FFT 1M points | 180 ms | 14 ms (13x) | 3.2 ms (56x) | 0.8 ms (225x) |
| K-means (10K pts, 10 clusters) | 850 ms | 32 ms (27x) | 8.5 ms (100x) | 1.9 ms (447x) |
| Random Forest inference (1K) | 45 ms | 1.8 ms (25x) | 0.9 ms (50x) | N/A |

### I/O-Intensive Workloads

| Workload | Python | Rust | Speedup | Notes |
|----------|--------|------|---------|-------|
| CSV parse 100MB | 4.2s | 0.38s | 11x | Rust uses zero-copy parsing |
| JSON serialize 1M records | 3.8s | 0.22s | 17x | serde vs json module |
| File scan 10K files | 1.9s | 0.15s | 13x | Parallel with rayon |
| HTTP server (req/sec) | 2,800 | 95,000 | 34x | axum vs flask |

### ML Inference

| Model | Python (PyTorch) | Rust (realizar) | Speedup | Notes |
|-------|------------------|-----------------|---------|-------|
| BERT-base (batch=1) | 12 ms | 4.2 ms | 2.9x | CPU |
| Qwen 1.5B (tok/s, CPU) | 8.5 | 18 | 2.1x | AVX2 |
| Qwen 1.5B (tok/s, GPU) | — | 240 | — | RTX 4090 CUDA, APR Q4K (GH-88) |
| Whisper-tiny (1s audio) | 180 ms | 45 ms | 4.0x | CPU |

## Memory Usage Comparisons

| Workload | Python Peak RSS | Rust Peak RSS | Reduction |
|----------|----------------|---------------|-----------|
| Idle process | 28 MB | 1.2 MB | 23x |
| Load 100MB dataset | 380 MB | 105 MB | 3.6x |
| BERT inference | 1.2 GB | 420 MB | 2.9x |
| Qwen 1.5B Q4K | 4.8 GB | 1.1 GB | 4.4x |
| 10K concurrent connections | 2.1 GB | 85 MB | 25x |

## Benchmark Methodology

All benchmarks follow these principles:

- **Warm-up**: 5 iterations discarded before measurement
- **Iterations**: Minimum 100 iterations or 10 seconds
- **Statistics**: Median reported with 95% confidence interval
- **Environment**: Isolated system, no other workloads
- **Reproduction**: Benchmark code included in `benches/` directory

```bash
# Run the full benchmark suite
cargo bench

# Run a specific benchmark
cargo bench -- matrix_multiply

# Compare against baseline
cargo bench -- --baseline python_baseline
```

## Hardware Reference

Benchmark hardware unless otherwise noted:

| Component | Specification |
|-----------|--------------|
| CPU | AMD EPYC 7443P (24 cores, 48 threads) |
| RAM | 256 GB DDR4-3200 ECC |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Storage | NVMe SSD (7 GB/s read) |
| OS | Linux 6.8.0, Ubuntu 24.04 |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
