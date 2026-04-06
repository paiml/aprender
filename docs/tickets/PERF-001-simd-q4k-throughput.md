# PERF-001: SIMD Q4_K Dequantization Throughput Below Target

**Status:** Open
**Priority:** High
**Component:** quantize.rs
**Spec Reference:** IMP-001 in performance-parity-ollama-llamacpp-gpu-inference-llms.md

## Summary

SIMD Q4_K dequantization throughput is measured at ~0.45 GB/s, significantly below the target of 10 GB/s.

## Current Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Throughput | 0.45 GB/s | 10.0 GB/s | 22x |
| Speedup vs Scalar | ~1x | 4x | 4x |

## Root Cause Analysis

1. **Rayon Parallel Overhead**: The `dequantize_q4_k_simd` function uses rayon for parallelization, which has significant overhead for the benchmark's iteration pattern

2. **Super-block Size**: Each super-block is only 144 bytes, causing excessive parallel task scheduling overhead

3. **Memory Allocation**: New `Vec<f32>` is allocated for each dequantization call

## Proposed Solutions

### Option A: Chunk-based Parallelism (Recommended)
Process multiple super-blocks per parallel task to reduce scheduling overhead:

```rust
const CHUNK_SIZE: usize = 64; // Process 64 super-blocks per task
data.par_chunks(SUPER_BLOCK_BYTES * CHUNK_SIZE)
    .flat_map(|chunk| dequantize_chunk_avx2(chunk))
    .collect()
```

### Option B: Pre-allocated Output Buffer
Avoid allocation overhead by accepting output slice:

```rust
pub fn dequantize_q4_k_into(data: &[u8], output: &mut [f32]) -> Result<()>
```

### Option C: Single-threaded SIMD for Small Data
Skip rayon for data under threshold:

```rust
if num_super_blocks < 256 {
    return dequantize_q4_k_avx2_sequential(data);
}
```

## Acceptance Criteria

- [ ] Throughput > 5 GB/s for 1MB+ data
- [ ] Throughput > 10 GB/s for 10MB+ data
- [ ] No regression in correctness tests
- [ ] `cargo test --lib test_imp_001` passes

## Test Command

```bash
cargo run --example performance_parity --release
cargo bench --bench quantize
```

## Related

- IMP-001 in specification
- `src/quantize.rs:2126` - `dequantize_q4_k_simd`
- `benches/quantize.rs` - Criterion benchmarks
