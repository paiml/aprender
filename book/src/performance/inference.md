# Inference Performance

> **Status**: Implemented - All 4 Phases Complete
>
> See: `src/quantize.rs`, `src/layers.rs`, `src/gpu.rs`

## Overview

Realizar implements a 4-phase performance optimization strategy based on the llama-cpp-style-performance-spec.md specification. Each phase builds on the previous, with acceptance tests verifying targets.

## Performance Summary

| Phase | Target | Achieved | Status |
|-------|--------|----------|--------|
| Phase 1: Quantized Compute | ULP ≤4, <5s | ULP ≤4, 0.01s | ✅ 500x margin |
| Phase 2: Memory Hierarchy | <1000ms, <30s | 0.6ms, 3463 tok/s | ✅ 1666x margin |
| Phase 3: Algorithmic | ≥25 tok/s | 553.7 tok/s | ✅ 22x target |
| Phase 4: GPU Acceleration | ≥25 tok/s (wgpu) | 35.0 tok/s | ✅ 1.4x target |

## Phase 1: Quantized Compute Foundation

Eliminates the 8x memory bandwidth gap via fused quantized operations.

### Key Optimizations

- **Fused Q4_K dequant+dot**: No intermediate f32 buffer
- **AVX2 SIMD kernels**: 4-8x compute speedup
- **L2-aware tiling**: Cache-optimized matrix operations

### Acceptance Test

```rust
#[test]
fn test_phase1_acceptance_fused_q4k_inference() {
    // Correctness: ULP ≤ 4 per Goldberg [9]
    assert_ulp_eq(fused, reference, 4, "Phase 1: fused Q4_K dot product");

    // Performance: < 5 seconds for 100 passes × 4 layers
    assert!(elapsed < Duration::from_secs(5));
}
```

**Result**: ULP ≤4, 0.01s < 5s ✅

## Phase 2: Memory Hierarchy Optimization

Maximizes cache utilization and eliminates TLB misses.

### Key Optimizations

- **L2-aware tiled matmul**: 64-element tiles fit L2 cache
- **Cache-oblivious blocking**: Works across cache sizes
- **Prefetching hints**: 1.3x improvement

### Acceptance Test

```rust
#[test]
fn test_phase2_acceptance_memory_hierarchy() {
    // Single forward pass < 1000ms
    assert!(forward_elapsed < Duration::from_millis(1000));

    // Long-context (2048 tokens) benchmark < 30s
    assert!(long_context_elapsed < Duration::from_secs(30));
}
```

**Result**: forward=0.6ms, long-context 3463 tok/s ✅

## Phase 3: Algorithmic Optimization

Implements Flash Attention and operator fusion.

### Key Optimizations

- **Flash Attention v2**: O(N) memory instead of O(N²)
- **Fused LayerNorm+Linear**: Reduces kernel launches
- **Parallel layer execution**: Rayon work-stealing

### Acceptance Test

```rust
#[test]
fn test_phase3_acceptance_tokens_per_second() {
    assert!(
        tok_per_sec >= 25.0,
        "Phase 3 acceptance FAILED: {:.1} tok/s < 25.0 tok/s target",
        tok_per_sec
    );
}
```

**Result**: 553.7 tok/s ✅ (22x target)

## Phase 4: GPU Acceleration

Portable GPU inference via wgpu backend.

### Key Components

- **GpuCompute**: Trueno wgpu backend wrapper
- **HybridScheduler**: Auto CPU/GPU selection (threshold: 1000 elements)
- **GpuBufferPool**: Buffer reuse for efficiency
- **AsyncGpuResult**: Non-blocking GPU operations

### Acceptance Test

```rust
#[test]
fn test_phase4_acceptance_gpu_throughput() {
    let mut compute = GpuCompute::auto().unwrap();

    // GPU (wgpu) target: ≥25 tok/s
    assert!(tok_per_sec >= 25.0);
}
```

**Result**: 35.0 tok/s on GPU (wgpu) ✅

## Running Acceptance Tests

```bash
# Run all acceptance tests
cargo test --lib --release "acceptance" -- --nocapture

# Individual phases
cargo test --lib --release phase1_acceptance -- --nocapture
cargo test --lib --release phase2_acceptance -- --nocapture
cargo test --lib --release phase3_acceptance -- --nocapture
cargo test --lib --release phase4_acceptance -- --nocapture
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Realizar Inference Stack v2                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 5: Public API (100% Safe)                                            │
│    - GGUFTransformer::forward(), predict_next()                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Model Orchestration (Safe)                                        │
│    - KV Cache Manager, Sampler, Token Generation Loop                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Operator Fusion (Safe wrappers)                                   │
│    - Fused Dequant+MatMul, Fused LayerNorm+Linear                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Trueno Safe API (Safe wrappers around unsafe cores)               │
│    - Matrix::matmul(), Vector::dot(), activations                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Unsafe Compute Kernels (Encapsulated, fuzz-tested)                │
│    - AVX2/AVX-512 SIMD intrinsics                                          │
│    - Quantized dot products (Q4_K, Q6_K)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 0: Backend Dispatch (compile-time feature flags)                     │
│    - AVX2 | AVX-512 | NEON | WASM SIMD | wgpu (GPU)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## References

- [llama-cpp-style-performance-spec.md](../../../docs/specifications/llama-cpp-style-performance-spec.md) - Full specification
- [Goldberg (1991)](https://doi.org/10.1145/103162.103163) - Floating-point arithmetic
- [Dao et al. (2022)](https://arxiv.org/abs/2205.14135) - FlashAttention

## See Also

- [GPU Dispatch Strategy](../gpu/dispatch-strategy.md)
- [GPU Memory Management](../gpu/memory-management.md)
- [Quantization](../quantization/what-is-quantization.md)
