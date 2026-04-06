# Phase 15: Fused Q4K Quantized GEMV

**Version:** 1.0.0
**Status:** SPECIFICATION
**Date:** 2025-01-15
**Target:** 2x Ollama CPU throughput for APR format
**Scope:** Fused dequantization + dot product for Q4_K quantized inference

---

## Executive Summary

This specification defines the implementation of **fused Q4K dequant+dot kernels** for trueno, enabling APR format to achieve performance parity with GGUF/llama.cpp. The current APR CPU path achieves ~15 tok/s vs llama.cpp's ~42 tok/s on TinyLlama-1.1B Q4_0.

**Root Cause:** APR's current path separates dequantization from GEMV, causing:
1. Extra memory traffic (dequant to temp buffer, then dot)
2. Cache pollution from intermediate f32 expansion
3. Missed SIMD fusion opportunities

**Solution:** Fused Q4K kernels that dequantize directly into SIMD registers during dot product computation.

---

## 1. Problem Statement

### 1.1 Current Architecture Gap

| Implementation | Q4K Approach | Throughput (1.1B) |
|----------------|--------------|-------------------|
| llama.cpp | Fused Q4×Q8 SIMD (llamafile) | ~42 tok/s |
| candle | Inline AVX2 dequant+dot | ~10 tok/s |
| APR (current) | Separate dequant → f32 dot | ~15 tok/s |
| **APR (target)** | **Fused Q4×Q8 SIMD** | **≥42 tok/s** |

### 1.2 Memory Bandwidth Analysis

For a 4096×4096 Q4_K matmul (single token decode):

| Approach | Memory Read | Memory Write | Efficiency |
|----------|-------------|--------------|------------|
| Separate dequant | 8 MB (Q4) + 64 MB (f32 temp) | 64 MB | 17% |
| Fused Q4×Q8 | 8 MB (Q4) + 4 MB (Q8 input) | 0 | 100% |

**Key Insight:** Fused approach eliminates 128 MB of unnecessary memory traffic per matmul.

### 1.3 Scope

**IN SCOPE:**
- Fused Q4_K × Q8_K SIMD dot product (AVX2, AVX-512, NEON)
- Cache-blocked quantized GEMV
- Integration with realizar APR transformer
- Benchmarks against llama.cpp baseline

**OUT OF SCOPE:**
- GPU quantized kernels (see trueno-gpu Phase 16)
- Training quantization
- New quantization formats

---

## 2. Root Cause Analysis (5 Whys)

### Why #1: Why is APR 2.8x slower than llama.cpp on CPU?

**Answer:** Each token generation performs ~200 matmuls, and APR's matmul is 2.8x slower.

### Why #2: Why is APR matmul 2.8x slower?

**Answer:** APR dequantizes Q4_K weights to f32 before GEMV, while llama.cpp keeps data quantized.

### Why #3: Why does dequantization hurt performance?

**Answer:** Dequantizing 4096×4096 Q4_K (8 MB) produces 64 MB of f32, exceeding L3 cache.

### Why #4: Why not keep data quantized during GEMV?

**Answer:** trueno's current SIMD kernels only support f32 dot products, not Q4×Q8.

### Why #5: Why weren't fused Q4×Q8 kernels implemented?

**Answer:** Phase 2 focused on f32 matmul parity with NumPy. **This is the root cause.**

---

## 3. Solution Architecture

### 3.1 Q4_K Block Format

Q4_K quantization uses 256-element blocks with super-blocks:

```
Block layout (256 elements = 32 super-blocks × 8 elements):
  - scales: [f16; 12] (12 bytes) - scale factors per super-block
  - d: f16 (2 bytes) - block-wide scale
  - dmin: f16 (2 bytes) - block-wide minimum
  - qs: [u8; 128] (128 bytes) - packed 4-bit quantized values

Total: 144 bytes per 256 elements = 4.5 bits/element
```

### 3.2 Fused Q4K×Q8K Kernel Design

The fused kernel computes dot(Q4_K_weights, Q8_K_input) without intermediate dequantization:

```rust
/// Fused Q4K × Q8K dot product
///
/// Computes: sum(dequant(q4) * dequant(q8)) directly in SIMD registers
#[target_feature(enable = "avx2")]
unsafe fn fused_q4k_q8k_dot_avx2(
    q4_block: &BlockQ4K,  // 256 quantized weights
    q8_block: &BlockQ8K,  // 256 quantized inputs
) -> f32 {
    // Step 1: Load scales (stays in registers)
    let d = f16_to_f32(q4_block.d);
    let dmin = f16_to_f32(q4_block.dmin);

    // Step 2: Process 32 elements at a time (4 super-blocks)
    let mut acc = _mm256_setzero_ps();

    for sb in 0..8 {  // 8 iterations × 32 elements = 256
        let offset = sb * 32;

        // Load Q4 nibbles (16 bytes = 32 values)
        let q4_packed = _mm_loadu_si128(&q4_block.qs[offset/2]);

        // Unpack nibbles to bytes: [n0|n1, n2|n3, ...] → [n0, n1, n2, n3, ...]
        let q4_lo = _mm256_and_si256(
            _mm256_cvtepu8_epi16(q4_packed),
            _mm256_set1_epi16(0x0F)
        );
        let q4_hi = _mm256_and_si256(
            _mm256_srli_epi16(_mm256_cvtepu8_epi16(q4_packed), 4),
            _mm256_set1_epi16(0x0F)
        );

        // Load Q8 values (32 bytes = 32 int8)
        let q8_vec = _mm256_loadu_si256(&q8_block.qs[offset]);

        // Integer multiply-add: q4 * q8
        let prod_lo = _mm256_maddubs_epi16(q4_lo, q8_vec_lo);
        let prod_hi = _mm256_maddubs_epi16(q4_hi, q8_vec_hi);

        // Accumulate with scale
        let scale = get_scale(q4_block.scales, sb);
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_add_epi32(prod_lo, prod_hi)),
            _mm256_set1_ps(d * scale),
            acc
        );
    }

    // Horizontal sum
    horizontal_sum_avx2(acc)
}
```

### 3.3 Cache-Blocked GEMV

For large matrices, apply L2 cache blocking:

```rust
/// Cache-blocked Q4K GEMV
///
/// y[M] = A[M×K] × x[K] where A is Q4K quantized
pub fn q4k_gemv_blocked(
    output: &mut [f32],        // M outputs
    weights: &[BlockQ4K],      // M×K/256 blocks
    input: &BlockQ8K,          // K/256 blocks (quantized input)
    m: usize,
    k: usize,
) {
    const BLOCK_M: usize = 64;  // Rows per L2 block
    const BLOCK_K: usize = 4096; // Columns per L2 block (fits in L2)

    // Process in L2-friendly blocks
    for m_start in (0..m).step_by(BLOCK_M) {
        let m_end = (m_start + BLOCK_M).min(m);

        // Parallel over output rows
        (m_start..m_end).into_par_iter()
            .with_min_len(16)  // Avoid Rayon overhead
            .for_each(|row| {
                let mut sum = 0.0f32;

                // Process K dimension in blocks
                for k_block in 0..(k / 256) {
                    sum += fused_q4k_q8k_dot_avx2(
                        &weights[row * (k/256) + k_block],
                        &input[k_block],
                    );
                }

                output[row] = sum;
            });
    }
}
```

---

## 4. Implementation Plan

### 4.1 File Structure

```
trueno/src/
├── quantize/
│   ├── mod.rs           # Module exports
│   ├── formats.rs       # Q4_K, Q5_K, Q6_K, Q8_K structs
│   ├── fused_avx2.rs    # AVX2 fused kernels
│   ├── fused_avx512.rs  # AVX-512 fused kernels
│   ├── fused_neon.rs    # ARM NEON fused kernels
│   └── blocked_gemv.rs  # Cache-blocked GEMV
```

### 4.2 API Design

```rust
// trueno/src/lib.rs
pub mod quantize;

// Public API
pub use quantize::{
    BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K,
    q4k_gemv, q4k_gemm,
    quantize_f32_to_q4k, quantize_f32_to_q8k,
};
```

### 4.3 Integration with realizar

```rust
// realizar/src/apr_transformer.rs
use trueno::quantize::{q4k_gemv, BlockQ4K, BlockQ8K};

fn forward_ffn(&self, hidden: &mut [f32]) {
    // Quantize input to Q8
    let input_q8 = quantize_f32_to_q8k(hidden);

    // Fused Q4K × Q8K GEMV (no intermediate f32)
    q4k_gemv(
        &mut self.up_out,
        &self.up_weights_q4k,
        &input_q8,
        self.intermediate_dim,
        self.hidden_dim,
    );

    // ... rest of FFN
}
```

---

## 5. Falsifiable Hypotheses

### H1: Fused Kernel Throughput

**Claim:** Fused Q4K×Q8K dot product achieves ≥2x throughput vs separate dequant+dot.

**Falsification:** Benchmark 10M dot products. If fused < 1.5x separate, hypothesis falsified.

**Prediction:** fused_throughput / separate_throughput ≥ 2.0

### H2: Memory Bandwidth Reduction

**Claim:** Fused approach reduces memory traffic by ≥80% for Q4K matmul.

**Falsification:** Profile with `perf stat`. If LLC-load-misses reduced <50%, hypothesis falsified.

**Prediction:** memory_traffic_fused / memory_traffic_separate ≤ 0.2

### H3: End-to-End Inference Speedup

**Claim:** APR with fused kernels achieves ≥2x Ollama throughput on TinyLlama-1.1B.

**Falsification:** Benchmark 100 tokens. If throughput < 1.5x Ollama, hypothesis falsified.

**Prediction:** apr_fused_throughput ≥ 2.0 × ollama_throughput

### H4: Numerical Accuracy

**Claim:** Fused kernel produces results within 1e-3 relative error of f32 reference.

**Falsification:** Compare 1000 random dot products. If max_rel_error > 1e-3, falsified.

**Prediction:** max_relative_error < 1e-3

---

## 6. Benchmark Targets

### 6.1 Micro-Benchmarks

| Kernel | Current (ns) | Target (ns) | Speedup |
|--------|--------------|-------------|---------|
| Q4K dequant (256 elem) | 180 | N/A | - |
| f32 dot (256 elem) | 45 | N/A | - |
| Separate (dequant+dot) | 225 | N/A | baseline |
| **Fused Q4K×Q8K** | N/A | **<100** | **>2.2x** |

### 6.2 End-to-End Targets

| Model | Format | Current | Target | vs Ollama |
|-------|--------|---------|--------|-----------|
| TinyLlama-1.1B | APR Q4_K | 15 tok/s | **≥42 tok/s** | **≥2x** |
| Qwen2.5-0.5B | APR Q4_K | 21 tok/s | **≥60 tok/s** | **≥2x** |
| Phi-2 2.7B | APR Q4_K | 7 tok/s | **≥20 tok/s** | **≥2x** |

---

## 7. llamafile Reference Analysis

### 7.1 Key Techniques from llamafile sgemm

1. **Matrix Repacking**: Transpose and repack B matrix for sequential access
2. **4×1 Micro-kernel**: Process 4 output rows simultaneously
3. **L2 Cache Blocking**: 64×64 blocks fit in L2 (256KB)
4. **Fused Dequant**: Q4 nibble extraction inline with FMA

### 7.2 Adaptation for trueno

| llamafile | trueno Adaptation |
|-----------|-------------------|
| C++ with inline asm | Rust with `std::arch` intrinsics |
| Fixed block sizes | Configurable via `BlockConfig` |
| OpenMP parallelism | Rayon with `with_min_len()` |
| Platform-specific files | Feature-gated backends |

---

## 8. Testing Strategy

### 8.1 Unit Tests

```rust
#[test]
fn test_fused_q4k_q8k_dot_correctness() {
    let q4 = random_q4k_block();
    let q8 = random_q8k_block();

    let fused_result = fused_q4k_q8k_dot_avx2(&q4, &q8);
    let reference = reference_q4k_q8k_dot(&q4, &q8);

    assert!((fused_result - reference).abs() < 1e-3 * reference.abs());
}

#[test]
fn test_fused_kernel_speedup() {
    let (q4_blocks, q8_blocks) = setup_benchmark_data();

    let separate_time = bench(|| separate_dequant_dot(&q4_blocks, &q8_blocks));
    let fused_time = bench(|| fused_q4k_q8k_dot(&q4_blocks, &q8_blocks));

    assert!(separate_time / fused_time >= 1.5, "Fused must be ≥1.5x faster");
}
```

### 8.2 Integration Tests

```rust
#[test]
fn test_apr_inference_with_fused_kernels() {
    let model = load_apr_model("tinyllama-1.1b-q4k.apr");
    let input = "Hello, world!";

    let (output, throughput) = benchmark_inference(&model, input, 50);

    assert!(throughput >= 30.0, "Must achieve ≥30 tok/s");
    assert!(output.contains_coherent_text());
}
```

---

## 9. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-15 | Initial specification |

---

## 10. References

[1] Goto, K., & Van Geijn, R. A. (2008). "Anatomy of High-Performance Matrix Multiplication." ACM TOMS.
[2] Intel Corporation. (2024). "Intel 64 and IA-32 Architectures Optimization Reference Manual."
[3] Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS.
[4] llamafile sgemm: https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/sgemm.cpp
[5] Trueno Phase 2 Micro-Kernel: [phase2-microkernel.md](./phase2-microkernel.md)

---

*Specification for Trueno Phase 15 (2025-01-15)*
*Zero excuses. Zero defects. APR IS THE FORMAT.*
