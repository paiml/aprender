# TinyLlama 1.1B Benchmark

This guide demonstrates benchmarking TinyLlama-1.1B Q4_0 inference using Realizar's APR transformer format.

## Quick Start

```bash
# Run the benchmark (requires a GGUF model)
GGUF_MODEL=/path/to/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  cargo run --example convert_and_bench_apr --release
```

## What This Benchmark Measures

The `convert_and_bench_apr` example compares four inference modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **GGUF Q4_0** | Direct GGUF inference | Baseline comparison |
| **APR Q4_0** | Parallel matmul with allocations | High throughput |
| **APR + KV Cache** | Context-aware generation | Multi-turn conversations |
| **APR + Scratch** | Zero-allocation inference | Lowest latency |

## Sample Output

```
╔══════════════════════════════════════════════════════════════════╗
║           GGUF vs APR Inference Benchmark                        ║
╚══════════════════════════════════════════════════════════════════╝

   CPU Features:
     AVX2 FMA AVX-VNNI

1. Loading GGUF model: /path/to/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
   File size: 637.0 MB
   GGUF loaded in 0.12s
   Config: hidden_dim=2048, vocab_size=32000, layers=22

2. Creating APR Q4_0...
   Conversion completed in 0.00s

3. Benchmarking GGUF (single token)...
   Warming up...
   Throughput: 5.2 tok/s (192.3ms/tok)

4. Benchmarking APR Q4_0 (single token, parallel)...
   Warming up...
   Throughput: 7.5 tok/s (133.3ms/tok)

5. Benchmarking APR Q4_0 + KV Cache (context-aware)...
   Generated 20 tokens in 2800ms
   Throughput: 7.1 tok/s (140.0ms/tok avg)

6. Benchmarking APR Q4_0 + Scratch Buffers (zero-alloc)...
   Throughput: 7.4 tok/s (135.1ms/tok)
   vs allocating: 0.99x

╔══════════════════════════════════════════════════════════════════════╗
║                           Summary                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ Format           │ Throughput  │ Latency   │ vs GGUF │ Context      ║
╠══════════════════════════════════════════════════════════════════════╣
║ GGUF Q4_0        │     5.2 tok/s │  192.3ms │ 1.00x   │ None         ║
║ APR Q4_0         │     7.5 tok/s │  133.3ms │ 1.44x   │ None         ║
║ APR + KV Cache   │     7.1 tok/s │  140.0ms │ 1.37x   │ 20 tokens   ║
║ APR + Scratch    │     7.4 tok/s │  135.1ms │ 1.42x   │ Zero-alloc   ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Performance Analysis

### Current Results (v0.3.1)

| Metric | Realizar APR | Candle | llama.cpp |
|--------|--------------|--------|-----------|
| TinyLlama-1.1B Q4_0 | **7-11 tok/s** | 9.2-9.9 tok/s | ~42 tok/s |
| Startup time | 118-176ms | 80-180ms | ~100ms |
| Candle parity | 76-111% | - | - |

### Optimizations Implemented

The benchmark leverages these SIMD optimizations:

1. **Fused Q4_0 SIMD matmul** - 7x speedup via `vpshufb` nibble extraction
2. **AVX2+FMA attention** - 2x speedup for dot products and AXPY
3. **Parallel output rows** - ~4x on 22-core CPU via Rayon
4. **f16-to-f32 LUT** - ~1.1x for scale conversions
5. **Zero-copy model loading** - 6.7x faster startup via mmap
6. **Arena scratch buffers** - Eliminates per-token allocations

### CPU Feature Detection

The benchmark automatically detects available SIMD features:

```rust
#[cfg(target_arch = "x86_64")]
fn print_cpu_features() {
    use std::arch::is_x86_feature_detected;

    println!("   CPU Features:");
    print!("     ");
    if is_x86_feature_detected!("avx2") { print!("AVX2 "); }
    if is_x86_feature_detected!("fma") { print!("FMA "); }
    if is_x86_feature_detected!("avx512f") { print!("AVX-512 "); }
    if is_x86_feature_detected!("avx512vnni") { print!("AVX512-VNNI "); }
    if is_x86_feature_detected!("avxvnni") { print!("AVX-VNNI "); }
    println!();
}
```

## Understanding the Results

### APR vs GGUF

APR (Aprender Portable Runtime) format provides:

- **Parallel matmul**: Distributes output rows across CPU cores
- **Q4×Q8 integer path**: Avoids f32 accumulation overhead
- **Pre-computed quantization**: Activations quantized once per layer

### KV Cache Benefits

The KV cache stores past key/value projections for efficient generation:

```rust
let mut cache = apr_q4.create_kv_cache();

// First token processes full context
let _ = apr_q4.forward_with_cache(&[token], &mut cache);

// Subsequent tokens reuse cached K/V
for token in tokens {
    let _ = apr_q4.forward_with_cache(&[token], &mut cache);
}
```

### Scratch Buffer Zero-Allocation

Scratch buffers eliminate heap allocations during inference:

```rust
let mut scratch = apr_q4.create_scratch();

// All intermediate results written to pre-allocated buffers
let logits = apr_q4.forward_single_with_scratch(token, &mut scratch);
```

## Roofline Analysis

Memory bandwidth limits theoretical throughput:

```
Model size: 637 MB (Q4_0)
DDR5 bandwidth: ~50 GB/s theoretical, ~30 GB/s practical
Minimum time per token: 637 MB / 30 GB/s = 21 ms
Maximum theoretical throughput: 1000 / 21 = ~47 tok/s

Current performance: 7-11 tok/s → 15-23% of roofline
```

## Running Custom Models

To benchmark your own GGUF model:

```bash
# Set the model path
export GGUF_MODEL=/path/to/your-model.Q4_0.gguf

# Run benchmark
cargo run --example convert_and_bench_apr --release
```

Supported quantizations:
- Q4_0 (4-bit, block size 32)
- Q8_0 (8-bit, block size 32)

## See Also

- [SIMD Optimization Spec](https://github.com/paiml/realizar/blob/main/docs/specifications/simd-optimization-spec.md) - Detailed optimization documentation
- [Quantization Overview](../quantization/what-is-quantization.md) - Q4_0/Q8_0 format details
- [GPU Acceleration](../gpu/simd.md) - SIMD backend information
- [Performance Benchmarks](../performance/inference.md) - General inference performance
