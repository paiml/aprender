# PERF-009: M8 Extended Context (4096+ Positions)

**Status:** RESOLVED
**Priority:** Medium
**Spec Reference:** docs/specifications/performance-parity-ollama-llamacpp-gpu-inference-llms.md
**Parent:** PERF-008 (M7 Complete)

## Objective

Extend context length support to 4096+ positions while maintaining memory efficiency and throughput.

| Milestone | Target | Current | Status |
|-----------|--------|---------|--------|
| M8: Extended Context | 4096 positions | **4096 positions** | ✅ COMPLETE |

## M8 Resolution Summary

**Key Achievement**: Extended context support works out of the box!

The StreamingKVCache implementation already supports arbitrary context lengths. The M8 benchmark validates:

1. **4096 positions filled** successfully
2. **Fill rate** > 500 positions/sec (achieved)
3. **Retrieve rate** > 5000 ops/sec (achieved)
4. **Memory** < 4.5 GB for KV cache (achieved ~4.3 GB)
5. **GPU-011 Benchmark**: PASS

## Current State

- M1-M7: All complete ✅
- Current Context: 2048 positions (GPU-009 benchmark)
- Memory Efficiency: 6.15 GB for 7B simulation
- Sustained Throughput: 86.61 tok/s
- All 18/18 benchmarks passing

## M8 Requirements

### Target Performance

| Metric | M7 (Current) | M8 Target |
|--------|--------------|-----------|
| Context Length | 2048 | **4096** |
| KV Cache Memory | 2.15 GB | < 4.5 GB |
| Token Gen | 86.61 tok/s | > 40 tok/s |
| Total VRAM (7B) | 6.15 GB | < 10 GB |

### Success Criteria

1. **Context length 4096+ supported** without OOM
2. **KV Cache scales efficiently** with context length
3. **GPU-011 benchmark passes** with 4096 positions
4. **Throughput maintained** at > 40 tok/s

## Implementation Plan

### Phase 1: Extended StreamingKVCache

The StreamingKVCache already supports arbitrary `max_positions`. We need to:
1. Add benchmark with 4096 positions
2. Verify memory bounds are maintained
3. Test fill and retrieval performance

### Phase 2: Memory Efficiency Optimization

For 4096 context with 7B model config:
- 32 layers × 4096 positions × 32 heads × 128 dim × 2 (K+V) × 4 bytes
- = 4.29 GB KV cache (vs 2.15 GB for 2048)

Optimization strategies:
1. **Quantized KV Cache** - Store KV in FP16 or INT8
2. **Sliding Window** - Only keep recent N positions
3. **Chunked Processing** - Process long contexts in chunks

### Phase 3: GPU-011 Benchmark

Add extended context benchmark:
```rust
/// GPU-011: Extended Context (M8 target: 4096 positions)
fn bench_extended_context() -> BenchResult {
    // Create cache with 4096 max positions
    // Fill to capacity
    // Verify memory bounds and performance
}
```

## Test Plan (EXTREME TDD)

### Unit Tests

1. `test_streaming_kv_cache_4096_positions` - Cache handles 4096 positions
2. `test_extended_context_memory_bound` - Memory stays bounded
3. `test_extended_context_fill_performance` - Fill rate maintained

### Benchmark Tests

1. GPU-011: Extended Context benchmark (4096 positions)

## Memory Budget (4096 Context)

| Component | Size | Notes |
|-----------|------|-------|
| Model Weights | 4.0 GB | 7B Q4_K |
| KV Cache | 4.3 GB | 4096 ctx, 32 layers |
| Activations | 1.0 GB | Peak during forward |
| Buffer Pool | 0.5 GB | Reusable GPU buffers |
| **Total** | **9.8 GB** | Fits in 10GB VRAM |

## Commands

```bash
# Run with extended context benchmark
cargo run --example performance_parity --features gpu --release

# Run specific extended context tests
cargo test --lib test_extended_context --features gpu
```

## Related

- PERF-007: M6 Memory Efficiency (✅ Complete)
- PERF-008: M7 Production Parity (✅ Complete)
- StreamingKVCache implementation in src/gpu.rs
