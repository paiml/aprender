# GPU & Performance Parity Examples

This chapter covers the examples that verify performance parity with Ollama and llama.cpp.

## Overview

The PARITY examples systematically verify performance claims through Popperian falsification - each example is designed to **disprove** a performance claim, and surviving tests represent verified guarantees.

## GPU Benchmark Examples

### `gpu_matvec_benchmark.rs` - GPU vs SIMD Matvec

Demonstrates that GPU is **2.7x SLOWER** for matrix-vector (MATVEC) operations due to transfer overhead.

```bash
cargo run --example gpu_matvec_benchmark --features gpu
```

**Key Finding (IMP-600):**
- MATVEC: GPU 10.8ms vs SIMD 4.0ms
- GPU transfer overhead dominates for small batch sizes
- SIMD is optimal for token generation (batch_size=1)

### `gpu_gemm_benchmark.rs` - GPU GEMM Performance

Demonstrates that GPU is **57x FASTER** for large GEMM (matrix-matrix) operations.

```bash
cargo run --example gpu_gemm_benchmark --features gpu
```

**Key Finding:**
- 1024x1024x1024 GEMM: GPU 41.8ms vs Scalar 2384ms
- GPU excels at batch/prompt processing
- Crossover point: batch_size > 32

## PARITY Verification Examples

### `parity_035_m4_verification.rs` - M4 Milestone

Verifies M4 target (90% of llama.cpp throughput):

```bash
cargo run --example parity_035_m4_verification --features gpu
```

### `parity_036_gpu_attention.rs` - GPU Attention

Benchmarks GPU attention implementation:

```bash
cargo run --example parity_036_gpu_attention --features gpu
```

### `parity_038_async_streams.rs` - CUDA Async Streams

Verifies 2x speedup from CUDA stream overlap:

```bash
cargo run --example parity_038_async_streams --features cuda
```

**Key Finding:**
- Overlapped compute + transfer: 101.99µs/token
- Sequential: 203.44µs/token
- **2x speedup confirmed**

### `parity_039_flash_attention.rs` - FlashAttention O(N) Memory

Verifies FlashAttention memory complexity:

```bash
cargo run --example parity_039_flash_attention --features cuda
```

**Key Finding:**
- O(N) vs O(N²) memory verified
- 32x reduction at seq_len=512

### `parity_040_fp16_attention.rs` - FP16 Tensor Core

Investigates FP16 Tensor Core performance:

```bash
cargo run --example parity_040_fp16_attention --features cuda
```

**Key Finding:**
- Current FP16 path uses tiled GEMM, not true Tensor Cores
- FP32: 74.4 GFLOPS vs FP16 Tiled: 65.0 GFLOPS
- True speedup requires WMMA PTX intrinsics

## IMP Verification Examples

### `imp_700_realworld_verification.rs` - Ollama HTTP Benchmark

Direct HTTP benchmarking against live Ollama server:

```bash
cargo run --example imp_700_realworld_verification
```

**Results:**
- Ollama: 240+ tok/s (CV=0.0388)
- realizar: 0.22 tok/s (pre-optimization)
- Gap: ~1090x

### `imp_800_kv_cache_falsification.rs` - KV Cache Verification

Verifies KV cache speedup claims:

```bash
cargo run --example imp_800_kv_cache_falsification
```

**Key Finding:**
- Speedup: 64-512x depending on sequence length
- Cache hit rate: >99%

## trueno Simulation Integration

These examples validate findings from trueno simulation research:

| Finding | Example That Demonstrates |
|---------|--------------------------|
| GPU threshold 100K | `gpu_matvec_benchmark` |
| PCG determinism | All benchmarks use fixed seeds |
| SIMD math properties | `trueno_dot_test` |
| PTX barriers correct | `parity_038_async_streams` |
| 1e-4 GPU tolerance | All GPU examples |

## Running All PARITY Examples

```bash
# Build all PARITY examples
cargo build --examples --features cuda

# Run individual examples
for ex in parity_035 parity_036 parity_038 parity_039 parity_040; do
    cargo run --example ${ex}* --features cuda 2>&1 | head -50
done
```

## Quick Reference

| Example | PARITY ID | What It Verifies |
|---------|-----------|------------------|
| `parity_035_m4_verification` | PARITY-035 | M4 milestone (90% parity) |
| `parity_036_gpu_attention` | PARITY-036 | GPU attention performance |
| `parity_038_async_streams` | PARITY-038 | CUDA async execution (2x) |
| `parity_039_flash_attention` | PARITY-039 | FlashAttention O(N) memory |
| `parity_040_fp16_attention` | PARITY-040 | FP16 Tensor Core baseline |
