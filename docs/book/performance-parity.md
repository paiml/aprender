# Chapter: Performance Parity with Ollama & llama.cpp

## Introduction

This chapter covers the implementation of performance parity between Realizar and production-grade LLM inference engines like Ollama and llama.cpp. We follow the Toyota Production System principles and EXTREME TDD methodology to achieve measurable, reproducible performance improvements.

## Prerequisites

Before reading this chapter, you should be familiar with:
- Rust basics and the Realizar crate structure
- Quantization formats (Q4_0, Q8_0, Q4_K)
- Transformer architecture fundamentals
- Basic benchmarking concepts

## Learning Objectives

By the end of this chapter, you will understand:
1. How to implement SIMD-accelerated dequantization
2. Memory-mapped weight streaming for large models
3. Fused attention kernels for improved throughput
4. KV cache optimization strategies
5. Batch processing and prefill optimization

## 1. The Performance Challenge

### Current State vs Target (v0.3.4)

| Runtime | Backend | Format | Throughput | vs llama.cpp | Status |
|---------|---------|--------|------------|--------------|--------|
| llama.cpp | CPU (OpenMP) | GGUF | 42-45 tok/s | 100% | Production |
| **Realizar** | CPU (AVX2) | **APR Q4_0** | **17.0-17.3 tok/s** | **38-41%** | **v0.3.4** |
| **Realizar** | CPU (AVX2) | GGUF | **12.5-13.0 tok/s** | **28-31%** | v0.3.4 |
| Candle | CPU | GGUF | 9.2-9.9 tok/s | 22% | Production |
| Realizar | WGPU | APR | TBD | TBD | In Development |

**Note:** All benchmarks on TinyLlama-1.1B Q4_0 model, Intel Core Ultra 7 155H (22 cores).

### APR Format Achievement (v0.3.4)

The `QuantizedAprTransformerQ4` format with **rayon parallelization** achieves **1.36x faster than GGUF**:

| Format | Size (MB) | Forward (ms) | Throughput | vs GGUF |
|--------|-----------|--------------|------------|---------|
| GGUF Q4_0 | 637.7 | 79.9 | 12.5 tok/s | 1.00x |
| **APR Q4_0** | 861.3 | 58.9 | **17.0 tok/s** | **1.36x** |
| APR F32 | 4196.4 | 12019.2 | 0.1 tok/s | 0.01x |

**Key optimizations in v0.3.4:**
- Parallel attention heads via rayon (32 heads parallelized)
- Parallel FFN up/gate computation via rayon::join
- 2.1x speedup from parallel optimizations (8.0 → 17.0 tok/s)

### Milestones

| Milestone | Target | Metric | Status |
|-----------|--------|--------|--------|
| M1: CPU Parity | Candle parity | ~10 tok/s CPU | **✅ Achieved (v0.3.2)** |
| M2: APR Q4_0 Parity | GGUF parity | Match GGUF tok/s | **✅ Achieved (v0.3.3)** |
| M3: Rayon Parallelization | 2x speedup | 16+ tok/s | **✅ Achieved (v0.3.4)** |
| M4: KV Cache | Context-aware gen | Efficient attention | **✅ Achieved (v0.3.5)** |
| M5: llama.cpp 50% | Half of llama.cpp | 21 tok/s | **In Progress (32-45%)** |
| M6: llama.cpp 100% | Full llama.cpp parity | 42+ tok/s | Planned |
| M7: Exceed llama.cpp | Beat llama.cpp | 50+ tok/s | **Target** |

### Roadmap to Exceed llama.cpp

Progress from ~8 tok/s to 50+ tok/s:

| Optimization | Expected Speedup | Cumulative | Status |
|--------------|------------------|------------|--------|
| **Baseline APR Q4_0** | 1.0x | 8 tok/s | ✅ Done (v0.3.3) |
| **Parallel attention heads (rayon)** | 1.5x | 12 tok/s | **✅ Done (v0.3.4)** |
| **Parallel FFN up/gate (rayon::join)** | 1.4x | **17 tok/s** | **✅ Done (v0.3.4)** |
| **KV Cache (context-aware)** | - | **13-19 tok/s** | **✅ Done (v0.3.5)** |
| SIMD RoPE (trueno vectors) | 1.1-1.2x | 15-23 tok/s | Next |
| Flash Attention (tiled) | 1.2-1.5x | 18-35 tok/s | Planned |
| Memory pooling (avoid allocs) | 1.1-1.2x | 20-42 tok/s | Planned |

**Progress:** 32-45% of llama.cpp achieved (13-19 tok/s vs 42 tok/s)
**KV Cache benefit:** Enables efficient autoregressive generation with context
**Next target:** SIMD RoPE and Flash Attention for additional speedup
**Conservative target:** 42+ tok/s (llama.cpp parity)
**Stretch target:** 50+ tok/s (exceed llama.cpp)

## 2. Toyota Production System Framework

We apply TPS principles throughout our optimization work:

### Genchi Genbutsu (Go and See)
Measure actual performance, not theoretical. Always benchmark before and after changes.

### Jidoka (Stop on Defects)
If a change regresses performance, stop immediately and investigate.

### Kaizen (Continuous Improvement)
Small, incremental improvements compound over time.

### Poka-yoke (Error-proofing)
Build benchmark infrastructure that catches regressions automatically.

## 3. SIMD-Accelerated Dequantization (IMP-001)

### The Problem

Q4_K quantization stores 256 values in a 144-byte super-block. Dequantization involves:
1. Reading scale factors (d, dmin)
2. Unpacking 4-bit values
3. Applying scales and minimums
4. Converting to f32

### Scalar Implementation

```rust
pub fn dequantize_q4_k(data: &[u8]) -> Result<Vec<f32>> {
    const SUPER_BLOCK_BYTES: usize = 144;

    // Process each super-block sequentially
    let num_super_blocks = data.len() / SUPER_BLOCK_BYTES;
    let mut result = Vec::with_capacity(num_super_blocks * 256);

    for sb_idx in 0..num_super_blocks {
        let sb_data = &data[sb_idx * SUPER_BLOCK_BYTES..];
        // Read d (f16), dmin (f16), scales, quantized values
        // Dequantize: value = d * q + dmin
        // ...
    }

    Ok(result)
}
```

### SIMD Implementation

```rust
pub fn dequantize_q4_k_simd(data: &[u8]) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dequantize_q4_k_avx2_parallel(data) };
        }
    }
    // Fallback to parallel scalar
    dequantize_q4_k_parallel(data)
}
```

### Performance Results

| Implementation | Throughput | Speedup |
|----------------|------------|---------|
| Scalar | ~2.5 GB/s | 1.0x |
| SIMD (AVX2) | ~10+ GB/s | 4x+ |
| SIMD + Rayon | ~40+ GB/s | 16x+ |

### Q4_0×Q8_0 Integer SIMD Matmul (v0.3.2)

The key optimization in v0.3.2 is integer SIMD matmul using `_mm256_maddubs_epi16`:

```rust
// Quantize activations to Q8_0 for integer multiply-accumulate
// Sign trick: ax = |x|, sy = sign(y, x), then maddubs(ax, sy) = x * y
unsafe {
    let ax = _mm256_and_si256(x_bytes, _mm256_set1_epi8(0x7F));
    let sy = _mm256_sign_epi8(y_bytes, x_bytes);
    let products = _mm256_maddubs_epi16(ax, sy);
    // ...
}
```

| Metric | Before (f32 FMA) | After (Q8_0 integer) |
|--------|------------------|----------------------|
| TinyLlama-1.1B Q4_0 | 4.2-7.1 tok/s | **8.4-11.9 tok/s** |
| vs Candle | 55-72% | **91-120%** |
| vs llama.cpp | 10-16% | **20-26%** |

### Running the Benchmark

```bash
# Download TinyLlama Q4_0 model
wget -O /path/to/tinyllama.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

# Run benchmark (with warmup and 10 iterations)
GGUF_MODEL=/path/to/tinyllama.gguf cargo run --example bench_forward --release
```

**Sample Output (verified 2025-12-30):**
```
Loading model: /mnt/ssd/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
Model loaded in 0.84s
Config: hidden_dim=2048, vocab_size=32000, num_layers=22

Warming up (3 iterations)...

Benchmarking single-token forward (10 iterations)...
Total time for 10 iterations: 1.072s
Average forward pass: 107.2ms
Throughput: 9.3 tok/s
```

### Test

```rust
#[test]
fn test_imp_001_q4k_simd_dequantize() {
    use crate::quantize::{dequantize_q4_k, dequantize_q4_k_simd};

    let data = vec![0u8; 144 * 4]; // 4 super-blocks

    // Verify correctness
    let scalar = dequantize_q4_k(&data).unwrap();
    let simd = dequantize_q4_k_simd(&data).unwrap();

    assert_eq!(scalar.len(), simd.len());
    for (s, p) in scalar.iter().zip(simd.iter()) {
        assert!((s - p).abs() < 1e-4);
    }
}
```

## 4. Memory-Mapped Weight Streaming (IMP-002)

### The Problem

A 7B parameter model requires ~4GB of weights (with Q4 quantization). Loading all weights into RAM wastes memory and causes slow startup.

### Solution: memmap2

```rust
use memmap2::Mmap;

pub fn load_weights_mmap(path: &Path) -> Result<Mmap> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}
```

### Benefits

1. **Lazy Loading**: Only pages accessed are loaded
2. **Shared Memory**: Multiple processes can share the same mapping
3. **OS Cache**: The OS manages caching automatically
4. **Fast Startup**: Model "loads" in milliseconds

### Test

```rust
#[test]
fn test_imp_002_mmap_weight_streaming() {
    let temp_file = std::env::temp_dir().join("test_mmap.bin");
    let weight_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    std::fs::write(&temp_file, &bytes).unwrap();

    let file = File::open(&temp_file).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    // Access memory without loading everything
    assert_eq!(mmap.len(), bytes.len());
}
```

## 5. Fused Attention Kernel (IMP-003)

### The Problem

Standard attention requires:
1. Compute Q*K^T (O(n^2) memory)
2. Apply softmax
3. Multiply by V

This creates large intermediate tensors.

### Solution: Fused QKV Attention

```rust
pub struct FusedQKVAttention {
    head_dim: usize,
    qkv_proj: Linear,  // Single projection for Q, K, V
    out_proj: Linear,
}

impl FusedQKVAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Single fused operation
        let qkv = self.qkv_proj.forward(x)?;

        // Split and compute attention in blocks
        let (q, k, v) = split_qkv(&qkv, self.head_dim);
        let attn = scaled_dot_product_attention(&q, &k, &v)?;

        self.out_proj.forward(&attn)
    }
}
```

### Performance Impact

| Approach | Memory | Latency |
|----------|--------|---------|
| Separate Q,K,V | O(3n) | 1.0x |
| Fused QKV | O(n) | ~2x faster |

## 6. KV Cache Optimization (IMP-004)

### The Problem

During autoregressive generation, we recompute K and V for all previous tokens, wasting compute.

### Solution: KV Cache

```rust
pub struct KVCache {
    k_cache: Vec<f32>,  // [layers, max_seq, hidden_dim]
    v_cache: Vec<f32>,
    seq_len: usize,
}

impl KVCache {
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        // Append new K, V to cache
        let offset = layer * self.max_seq_len * self.hidden_dim
                   + self.seq_len * self.hidden_dim;
        self.k_cache[offset..offset + k.len()].copy_from_slice(k);
        self.v_cache[offset..offset + v.len()].copy_from_slice(v);
        self.seq_len += 1;
    }

    pub fn get_k(&self, layer: usize) -> &[f32] {
        // Return all cached K values for this layer
        let start = layer * self.max_seq_len * self.hidden_dim;
        let end = start + self.seq_len * self.hidden_dim;
        &self.k_cache[start..end]
    }
}
```

### Performance Impact

Without KV cache: O(n^2) compute per token
With KV cache: O(n) compute per token (3x+ speedup for long contexts)

## 7. Batch Prefill (IMP-005)

### The Problem

Processing prompt tokens one at a time is inefficient.

### Solution: Batch Processing

```rust
pub fn prefill(&self, tokens: &[usize]) -> Result<Tensor> {
    // Process all prompt tokens in parallel
    let embeddings = self.embed_tokens(tokens)?;  // [seq_len, hidden_dim]

    // Single forward pass through transformer
    let hidden = self.transformer.forward(&embeddings)?;

    // Return logits for last position only (for generation)
    Ok(hidden.slice_last())
}
```

### Performance Impact

| Approach | Throughput |
|----------|------------|
| Sequential | ~100 tok/s |
| Batch prefill | ~1000+ tok/s |

## 8. Running the Benchmarks

### Example Command

```bash
cargo run --example performance_parity --release
```

### Sample Output

```
╔══════════════════════════════════════════════════════════════════╗
║        Performance Parity Benchmark Suite (PERF-PARITY-001)      ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────┬────────────┬────────────┬────────────┬────────┐
│ Benchmark               │ Metric     │ Value      │ Target     │ Status │
├─────────────────────────┼────────────┼────────────┼────────────┼────────┤
│ IMP-001: SIMD Q4_K      │ Throughput │ 12.34 GB/s │ 10.00 GB/s │ ✅ PASS │
│ IMP-002: Mmap Streaming │ Throughput │ 8.50 GB/s  │ 5.00 GB/s  │ ✅ PASS │
│ IMP-003: Fused Attention│ Latency    │ 5.20 ms    │ 10.00 ms   │ ✅ PASS │
│ IMP-004: KV Cache       │ Ops/sec    │ 150K ops/s │ 100K ops/s │ ✅ PASS │
│ IMP-005: Batch Prefill  │ Speedup    │ 6.2x       │ 5.0x       │ ✅ PASS │
└─────────────────────────┴────────────┴────────────┴────────────┴────────┘
```

## 9. Quality Assurance

### The 50-Point QA Checklist

Our implementation is validated by 50 QA tests covering:

1. **Correctness (QA-001 to QA-010)**: Output matches reference implementations
2. **Performance (QA-011 to QA-020)**: Throughput and latency targets met
3. **Reliability (QA-021 to QA-030)**: Graceful error handling
4. **Benchmarking (QA-031 to QA-040)**: Statistical validity
5. **Integration (QA-041 to QA-050)**: CI/CD integration

### Running Tests

```bash
# Run all IMP tests
cargo test --lib test_imp_

# Run all QA tests
cargo test --lib test_qa_

# Run full suite
cargo test --lib
```

## 10. Next Steps

After mastering these fundamentals, explore:

1. **GPU Acceleration (Phase 2)**: WGPU compute shaders
2. **Advanced Quantization (Phase 3)**: Q5_K, Q6_K, I-quant
3. **Attention Optimization (Phase 4)**: Flash Attention, GQA
4. **System Integration (Phase 5)**: Continuous batching, speculative decoding

## Summary

Key takeaways:
- SIMD dequantization provides 4x+ speedup
- Memory-mapped I/O enables efficient large model loading
- Fused kernels reduce memory traffic
- KV cache eliminates redundant computation
- Batch prefill maximizes throughput

The Toyota Production System principles guide our optimization work:
- **Genchi Genbutsu**: Measure everything
- **Jidoka**: Stop on quality issues
- **Kaizen**: Continuous improvement
- **Poka-yoke**: Automated regression detection

## References

1. Liker, J.K. "The Toyota Way" (2004)
2. Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems" (SC'15)
3. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (NeurIPS 2022)
4. Kwon et al., "Efficient Memory Management with PagedAttention" (SOSP '23)
5. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication" (NeurIPS 2022)
