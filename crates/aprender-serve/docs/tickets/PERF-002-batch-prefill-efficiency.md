# PERF-002: Batch Prefill Efficiency Below Target

**Status:** Open
**Priority:** Medium
**Component:** layers.rs
**Spec Reference:** IMP-005 in performance-parity-ollama-llamacpp-gpu-inference-llms.md

## Summary

Batch prefill throughput ratio is ~0.97x, indicating no efficiency gain from batching. The target is 5x throughput improvement when processing 8 tokens vs 1 token.

## Current Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Throughput Ratio (batch=8) | 0.97x | 5.0x | 5x |
| Batch Efficiency | ~12.5% | 62.5% | 5x |

## Root Cause Analysis

1. **Sequential Processing**: FusedQKVAttention processes batches sequentially internally, not leveraging batch parallelism

2. **Matrix Multiplication Layout**: Current matmul implementation doesn't optimize for batch dimension

3. **Memory Access Pattern**: Row-major layout causes cache inefficiency for batch operations

## Proposed Solutions

### Option A: Batched Matrix Multiplication (Recommended)
Implement true batched matmul that processes all batch elements in parallel:

```rust
pub fn batched_matmul(
    a: &Tensor,  // [batch, m, k]
    b: &Tensor,  // [k, n] or [batch, k, n]
) -> Tensor    // [batch, m, n]
```

### Option B: SIMD Batch Processing
Use SIMD to process multiple batch elements simultaneously:

```rust
// Process 4 batch elements at once with AVX2
#[target_feature(enable = "avx2")]
unsafe fn attention_batch4(q: &[f32; 4], k: &[f32; 4], v: &[f32; 4]) -> [f32; 4]
```

### Option C: Rayon Parallel Batches
Process batch elements in parallel with rayon:

```rust
(0..batch_size)
    .into_par_iter()
    .map(|b| self.forward_single(&input.slice(b)))
    .collect()
```

## Acceptance Criteria

- [ ] Throughput ratio > 3x for batch=8
- [ ] Throughput ratio > 5x for batch=16
- [ ] Linear scaling up to batch=32
- [ ] `cargo test --lib test_imp_005` passes

## Test Command

```bash
cargo run --example performance_parity --release
cargo bench --bench performance_parity
```

## Related

- IMP-005 in specification
- `src/layers.rs` - FusedQKVAttention
- QA-018 batch scaling test
