# Backend Comparison

This chapter compares Trueno's three execution backends (Scalar, SIMD, GPU) across different operation types and workload sizes, providing guidance on when to use each.

## Backend Overview

| Backend | Availability | Typical Speedup | Best Use Case |
|---------|--------------|-----------------|---------------|
| **Scalar** | All platforms | 1x (baseline) | Small workloads, reference implementation |
| **SIMD** | x86_64 (SSE2+), ARM (NEON), WASM | 2-4x | Most operations, <1M elements |
| **GPU** | Vulkan/Metal/DX12 systems | 10-80x | Large matrix ops (>500×500) |

## Decision Matrix

Use this table to choose the optimal backend for your workload:

| Operation Type | Size Range | Recommended Backend | Expected Speedup |
|----------------|------------|---------------------|------------------|
| Vector Add/Mul | Any | **SIMD** | 1.1-1.3x |
| Dot Product | <1M | **SIMD** | 3-4x |
| Dot Product | >1M | **SIMD** | 3-4x |
| Matrix Mul | <500×500 | **SIMD** | 2-4x |
| Matrix Mul | 500×500-1000×1000 | **GPU** | 16-81x |
| Matrix Mul | >1000×1000 | **GPU** | 80x+ |
| Activations (ReLU, Sigmoid) | Any | **SIMD** | 1.2-7x |
| Reductions (Sum, Max) | Any | **SIMD** | 3-4x |

## Scalar Backend

### Characteristics

- **Pros**:
  - Zero overhead
  - Simple, maintainable code
  - Predictable performance
  - Works everywhere

- **Cons**:
  - No parallelism
  - Slowest for compute-heavy operations

### When to Use Scalar

- Reference implementation for correctness testing
- Platforms without SIMD support (rare)
- Debugging (simpler code paths)
- Very small workloads (<100 elements) where SIMD overhead dominates

### Performance

| Operation | Size | Time | Throughput |
|-----------|------|------|------------|
| Vector Add | 10K | 819 ns | 12.21 Gelem/s |
| Dot Product | 10K | 6.30 µs | 1.59 Gelem/s |
| Matrix Mul | 1000×1000 | 638.7 ms | 1.57 Gelem/s |

## SIMD Backend

### Characteristics

- **Pros**:
  - Zero transfer overhead
  - 2-4x speedup for most operations
  - Low latency (<10µs for typical ops)
  - Works on all modern CPUs

- **Cons**:
  - Limited parallelism (4-8 elements)
  - Complex implementation
  - Platform-specific code

### SIMD Instruction Sets

| ISA | Register Width | Elements (f32) | Availability |
|-----|----------------|----------------|--------------|
| **SSE2** | 128-bit | 4 | All x86_64 CPUs |
| **AVX** | 256-bit | 8 | Intel Sandy Bridge+ (2011+) |
| **AVX2** | 256-bit + FMA | 8 | Intel Haswell+ (2013+) |
| **AVX-512** | 512-bit | 16 | Intel Skylake-X+ (2017+), AMD Zen 4+ (2022+) |
| **NEON** | 128-bit | 4 | All ARM64 CPUs |
| **SIMD128** | 128-bit | 4 | Modern browsers (WASM) |

### SIMD Performance (SSE2)

From golden traces (`golden_traces/performance_demo_summary.txt`):

| Operation | Size | Scalar | SIMD (SSE2) | Speedup | Runtime | Syscalls |
|-----------|------|--------|-------------|---------|---------|----------|
| **Dot Product** | 10K | 6.26µs | 1.55µs | **4.0x** ✅ | 1.507ms | 138 |
| **Sum Reduction** | 10K | 7.12µs | 1.69µs | **4.2x** ✅ | 1.507ms | 138 |
| **Max Finding** | 10K | 4.19µs | 1.06µs | **4.0x** ✅ | 1.507ms | 138 |
| **Element-wise Add** | 10K | 1.44µs | 1.10µs | 1.3x | 1.507ms | 138 |
| **Element-wise Mul** | 10K | 1.10µs | 1.10µs | 1.0x | 1.507ms | 138 |

### Why SIMD Excels

**Zero overhead architecture**:
- No data transfer (operates directly on CPU cache)
- No synchronization (single-threaded execution)
- Immediate execution (no queuing or dispatch)

**Optimal for**:
- ✅ Reduction operations (dot, sum, max): Parallel accumulation
- ✅ Compute-intensive ops (tanh, sigmoid): Amortizes instruction overhead
- ⚠️ Memory-bound ops (add, mul): Limited by RAM bandwidth, not compute

## GPU Backend

### Characteristics

- **Pros**:
  - Massive parallelism (thousands of cores)
  - 80x+ speedup for large matrix operations
  - Excellent for O(n³) algorithms

- **Cons**:
  - **3.5ms fixed overhead** per operation
  - Requires PCIe transfer (CPU↔GPU)
  - Only beneficial for large workloads
  - Not always available

### GPU Transfer Overhead

**Critical limitation**: Every GPU operation incurs ~3.5ms fixed cost:

| Component | Time | Description |
|-----------|------|-------------|
| Buffer creation | 0.5 ms | Allocate GPU-side memory |
| CPU→GPU transfer | 1.5 ms | PCIe bandwidth limitation |
| Kernel dispatch | 0.3 ms | GPU scheduling |
| GPU→CPU readback | 1.2 ms | PCIe bandwidth limitation |
| **Total** | **3.5 ms** | **Minimum per operation** |

### GPU Performance (RTX 4090)

**Vector operations (❌ GPU fails)**:

| Operation | Size | GPU Time | SIMD Time | Verdict |
|-----------|------|----------|-----------|---------|
| Vector Add | 10K | 3.44 ms | 1.10 µs | **SIMD 3127x faster** |
| Dot Product | 10K | 3.32 ms | 1.55 µs | **SIMD 2144x faster** |
| ReLU | 1M | 6.03 ms | 67.1 µs | **SIMD 90x faster** |
| Sigmoid | 1M | 5.81 ms | 3.18 ms | **SIMD 1.8x faster** |

**Matrix operations (✅ GPU wins)**:

| Size | GPU Time | Scalar Time | Speedup |
|------|----------|-------------|---------|
| 100×100 | 4.14 ms | 530.8 µs | 0.13x ❌ |
| 500×500 | 4.59 ms | 77.4 ms | **16.9x** ✅ |
| 1000×1000 | 7.84 ms | 638.7 ms | **81.5x** ✅ |

### Why GPU Fails for Vector Operations

**Transfer overhead dominates**:
- 10K vector add: 1.1µs compute vs 3500µs transfer → **3182x overhead**
- 1M vector add: 96.5µs compute vs 3500µs transfer → **36x overhead**

**Even compute-heavy ops suffer**:
- 1M sigmoid: 3.18ms compute vs 3.5ms transfer → Barely competitive

### Why GPU Wins for Matrix Operations

**O(n³) complexity overwhelms transfer cost**:
- 500×500 matmul: 125M ops → 77ms scalar → GPU wins at 4.6ms (13x amortization)
- 1000×1000 matmul: 1B ops → 639ms scalar → GPU wins at 7.8ms (81x amortization)

**GPU becomes competitive when**: `compute_time_scalar > 10 × transfer_overhead`

For matrix multiplication:
- 500×500: 77ms compute >> 3.5ms transfer → GPU wins
- 100×100: 531µs compute << 3.5ms transfer → GPU loses

## Backend Comparison by Operation Type

### Element-Wise Operations (add, mul, scale)

| Backend | Typical Time (10K) | Speedup vs Scalar | Verdict |
|---------|-------------------|-------------------|---------|
| Scalar | 800 ns | 1.0x | Baseline |
| SIMD | 600 ns | 1.3x | ✅ Use SIMD |
| GPU | 3400 µs | 0.0002x | ❌ Never use GPU |

**Recommendation**: Always use SIMD. Memory-bound, but SIMD has zero overhead.

### Reduction Operations (dot, sum, max)

| Backend | Typical Time (10K) | Speedup vs Scalar | Verdict |
|---------|-------------------|-------------------|---------|
| Scalar | 6.3 µs | 1.0x | Baseline |
| SIMD | 1.5 µs | 4.0x | ✅ Use SIMD |
| GPU | 3320 µs | 0.002x | ❌ Never use GPU |

**Recommendation**: Always use SIMD. Excellent parallel accumulation, zero overhead.

### Activation Functions (relu, sigmoid, tanh)

| Backend | Typical Time (1M) | Speedup vs Scalar | Verdict |
|---------|------------------|-------------------|---------|
| Scalar (ReLU) | 67.1 µs | 1.0x | Baseline |
| SIMD (ReLU) | ~20 µs | ~3x | ✅ Use SIMD |
| GPU (ReLU) | 6030 µs | 0.011x | ❌ Never use GPU |
| | | | |
| Scalar (Sigmoid) | 3.18 ms | 1.0x | Baseline |
| SIMD (Sigmoid) | ~1 ms | ~3x | ✅ Use SIMD |
| GPU (Sigmoid) | 5.81 ms | 0.55x | ❌ Never use GPU |

**Recommendation**: Always use SIMD, even for compute-heavy activations.

### Matrix Multiplication

| Backend | Time (1000×1000) | Speedup vs Scalar | Verdict |
|---------|-----------------|-------------------|---------|
| Scalar | 638.7 ms | 1.0x | Baseline |
| SIMD | ~160 ms | ~4x | ✅ Use for <500×500 |
| GPU | 7.84 ms | 81.5x | ✅ Use for >500×500 |

**Recommendation**: Use GPU for matrices >500×500, otherwise SIMD.

## Threshold Guidelines

### Current Trueno Thresholds

```rust
// Vector operations (src/vector.rs:1316)
const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED

// Matrix operations (src/matrix.rs:268)
const GPU_THRESHOLD: usize = 500; // 500×500 minimum
```

### Size-Based Recommendations

| Workload Size | Vector Ops | Matrix Ops | Rationale |
|---------------|------------|------------|-----------|
| <100 | Scalar/SIMD | Scalar/SIMD | SIMD overhead marginal |
| 100-1K | SIMD | SIMD | Sweet spot for SIMD |
| 1K-100K | SIMD | SIMD | SIMD still optimal |
| 100K-500×500 | SIMD | SIMD | GPU overhead too high |
| 500×500-1000×1000 | SIMD | **GPU** | O(n³) amortizes overhead |
| >1000×1000 | SIMD | **GPU** | Massive compute dominates |

### Operation Complexity Classes

Trueno categorizes operations by complexity:

```rust
pub enum OpComplexity {
    Low,    // Simple ops: add, mul (GPU disabled)
    Medium, // Moderate: dot, reduce (GPU at 100K+)
    High,   // Complex: matmul, conv2d (GPU at 500×500+)
}
```

## Performance Validation

### Golden Trace Baselines

Performance budgets in `renacer.toml` ensure SIMD doesn't regress:

```toml
[performance.budgets]
backend_detection = { max_time_ms = 2.0, max_syscalls = 200 }
matrix_operations = { max_time_ms = 2.0, max_syscalls = 200 }
activation_functions = { max_time_ms = 2.0, max_syscalls = 200 }
```

**All SIMD operations must complete in <2ms with <200 syscalls.**

### Validation Tests

`tests/golden_trace_validation.rs` ensures:
- SIMD performance matches golden traces (±10%)
- No unexpected syscall patterns
- Runtime stays under budget

## Future: Hybrid Scheduling (v2.0)

Current API forces a single backend per operation. Future hybrid scheduling will:

1. **Profile operation characteristics** at runtime
2. **Dynamically select backend** based on actual compute time
3. **Batch GPU operations** to amortize transfer overhead
4. **Overlap CPU and GPU work** for pipeline parallelism

**Example future API**:

```rust
let scheduler = HybridScheduler::new()
    .prefer_simd_threshold_ms(5.0)  // Use SIMD if op <5ms
    .gpu_batch_window_ms(10.0);     // Batch GPU ops within 10ms

scheduler.execute_pipeline(|pipe| {
    let a = pipe.add(&x, &y);       // SIMD (fast)
    let b = pipe.dot(&a, &z);       // SIMD (fast)
    let c = pipe.matmul(&b, &w);    // GPU (queued)
    let d = pipe.matmul(&c, &v);    // GPU (batched!)
    d
});
```

## Recommendations Summary

### For Vector Operations

1. **Always use SIMD** - Zero overhead, 2-4x speedup
2. **Never use GPU** - 2000x+ slower due to transfer overhead
3. **Use scalar** only for <100 elements or debugging

### For Matrix Operations

1. **Use SIMD** for matrices <500×500
2. **Use GPU** for matrices ≥500×500 (16-81x speedup)
3. **Consider batching** multiple GPU operations in future

### General Guidelines

- **Latency-critical**: Always SIMD (microsecond-scale)
- **Throughput-critical**: GPU for large batches, SIMD otherwise
- **Portable**: SIMD works everywhere (x86, ARM, WASM)
- **Maximum performance**: Profile and choose dynamically

## References

- **[GPU Performance](./gpu-performance.md)** - Detailed GPU benchmarks (RTX 4090)
- **[SIMD Performance](./simd-performance.md)** - SIMD optimization techniques
- **[Benchmarks Overview](./benchmarks.md)** - Complete benchmark methodology
- **Full report**: `docs/gpu-benchmark-report-2025-11-23.md`
- **Golden traces**: `golden_traces/ANALYSIS.md`
- **Configuration**: `renacer.toml`
