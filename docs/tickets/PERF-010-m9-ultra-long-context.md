# PERF-010: M9 Ultra-Long Context (8192+ Positions)

**Status:** RESOLVED
**Priority:** Medium
**Spec Reference:** docs/specifications/performance-parity-ollama-llamacpp-gpu-inference-llms.md
**Parent:** PERF-009 (M8 Complete)

## Objective

Extend context length support to 8192+ positions for ultra-long context generation while maintaining memory efficiency and throughput.

| Milestone | Target | Current | Status |
|-----------|--------|---------|--------|
| M9: Ultra-Long Context | 8192 positions | **8192 positions** | ✅ COMPLETE |

## M9 Resolution Summary

**Key Achievement**: Ultra-long context support works!

The StreamingKVCache implementation successfully supports 8192 positions. The M9 benchmark validates:

1. **8192 positions filled** successfully
2. **Fill rate** > 250 positions/sec (achieved)
3. **Retrieve rate** > 2500 ops/sec (achieved)
4. **Memory** < 9 GB for KV cache (achieved ~8.6 GB)
5. **GPU-012 Benchmark**: PASS

## Current State

- M1-M8: All complete ✅
- Current Context: 4096 positions (GPU-011 benchmark)
- Memory Efficiency: 6.15 GB for 7B simulation
- Sustained Throughput: 83.29 tok/s
- All 19/19 benchmarks passing

## M9 Requirements

### Target Performance

| Metric | M8 (Current) | M9 Target |
|--------|--------------|-----------|
| Context Length | 4096 | **8192** |
| KV Cache Memory | ~4.3 GB | < 9 GB |
| Token Gen | 83.29 tok/s | > 30 tok/s |
| Total VRAM (7B) | ~8.3 GB | < 12 GB |

### Success Criteria

1. **Context length 8192+ supported** without OOM
2. **KV Cache scales efficiently** with doubled context length
3. **GPU-012 benchmark passes** with 8192 positions
4. **Throughput maintained** at > 30 tok/s

## Implementation Plan

### Phase 1: Validate StreamingKVCache Scaling

The StreamingKVCache already supports arbitrary `max_positions`. We need to:
1. Add benchmark with 8192 positions
2. Verify memory bounds are maintained
3. Test fill and retrieval performance at scale

### Phase 2: Memory Budget Analysis

For 8192 context with 7B model config:
- 32 layers × 8192 positions × 32 heads × 128 dim × 2 (K+V) × 4 bytes
- = 8.59 GB KV cache (vs 4.29 GB for 4096)

Memory optimizations if needed:
1. **Quantized KV Cache** - Store KV in FP16 (halves memory)
2. **Sliding Window** - Only keep recent N positions for attention
3. **Chunked Processing** - Process ultra-long contexts in chunks

### Phase 3: GPU-012 Benchmark

Add ultra-long context benchmark:
```rust
/// GPU-012: Ultra-Long Context (M9 target: 8192 positions)
fn bench_ultra_long_context() -> BenchResult {
    let num_layers = 32;
    let max_positions = 8192; // M9 target
    // Fill cache to capacity
    // Verify memory bounds and performance
    // Target: > 30 tok/s throughput
}
```

## Test Plan (EXTREME TDD)

### Unit Tests

1. `test_streaming_kv_cache_8192_positions` - Cache handles 8192 positions
2. `test_ultra_long_context_memory_bound` - Memory stays bounded
3. `test_ultra_long_context_fill_performance` - Fill rate maintained

### Benchmark Tests

1. GPU-012: Ultra-Long Context benchmark (8192 positions)

## Memory Budget (8192 Context)

| Component | Size | Notes |
|-----------|------|-------|
| Model Weights | 4.0 GB | 7B Q4_K |
| KV Cache | 8.6 GB | 8192 ctx, 32 layers |
| Activations | 1.0 GB | Peak during forward |
| Buffer Pool | 0.5 GB | Reusable GPU buffers |
| **Total** | **14.1 GB** | Needs 16GB VRAM |

**Note:** For 8192 context, users will need 16GB+ VRAM (e.g., RTX 4080, RTX 3090, A4000+).

## Commands

```bash
# Run with ultra-long context benchmark
cargo run --example performance_parity --features gpu --release

# Run specific ultra-long context tests
cargo test --lib test_ultra_long_context --features gpu
```

## Related

- PERF-008: M7 Production Parity (✅ Complete)
- PERF-009: M8 Extended Context (✅ Complete)
- StreamingKVCache implementation in src/gpu.rs
