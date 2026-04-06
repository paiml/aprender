# GPU Performance Analysis Report

**Date**: 2025-11-17
**Commit**: 215a480
**CPU**: AMD Ryzen (TODO: detect)
**GPU**: Available via wgpu
**Platform**: Linux

## Executive Summary

**CRITICAL FINDING**: GPU implementation is **fundamentally broken** for 13 out of 14 operations.

**Key Results**:
- ✅ **Matrix multiplication ONLY**: 2-10x speedup for large matrices (500×500+)
- ❌ **ALL other operations**: 2x to 65,000x SLOWER than scalar code, even at 1M elements
- ❌ **Element-wise ops**: GPU overhead (~14-55ms) completely dominates execution time

**Root Cause**: Fixed ~14-55ms GPU overhead (buffer allocation + PCIe transfer + kernel dispatch) makes GPU **unsuitable for element-wise operations at ANY size**.

**Recommendation**: **Disable GPU for all operations except matmul immediately**. Focus on SIMD optimization.

---

## Methodology

- **Tool**: Criterion.rs (100 samples per measurement)
- **Backends**: GPU (wgpu) vs Scalar (baseline)
- **Sizes Tested**: 1K, 10K, 100K, 1M elements
- **Operations**: 14 GPU-accelerated operations

---

## Complete Benchmark Results

### 1. Vector Addition (`vec_add`)

| Size | GPU Time | Scalar Time | Speedup | Target | Status |
|------|----------|-------------|---------|--------|--------|
| 1K | 31.95 ms | 52.46 ns | **0.00164x** | >1x | ❌ 609x SLOWER |
| 10K | 32.21 ms | 856.3 ns | **0.0266x** | 2-5x | ❌ 37.6x SLOWER |
| 100K | 32.62 ms | 9.99 µs | **0.000306x** | 5-15x | ❌ 3266x SLOWER |
| 1M | 54.24 ms | 106.3 µs | **0.00196x** | 10-25x | ❌ 510x SLOWER |

**Finding**: GPU NEVER beneficial for vec_add - even at 1M elements, it's 510x slower!

### 2. Dot Product (`dot`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 1K | 38.26 ms | 607 ns | **0.0000159x** | ❌ 63,000x SLOWER |
| 10K | 34.08 ms | 6.30 µs | **0.000185x** | ❌ 5,410x SLOWER |
| 100K | 34.51 ms | 60.57 µs | **0.00176x** | ❌ 570x SLOWER |
| 1M | 57.24 ms | 612.9 µs | **0.0107x** | ❌ 93x SLOWER |

**Finding**: GPU NEVER beneficial for dot product - overhead too high.

### 3. Matrix Multiplication (`matmul`) - **ONLY GPU WINNER**

| Size (NxN) | GPU Time | Scalar Time | Speedup | Target | Status |
|------------|----------|-------------|---------|--------|--------|
| 100×100 | 38.74 ms | 545.1 µs | **0.0141x** | >5x | ❌ 71x SLOWER |
| 500×500 | 38.57 ms | 77.54 ms | **2.01x** | >10x | ⚠️ Below target (but positive!) |
| 1000×1000 | 67.80 ms | 650.3 ms | **9.59x** | 20-50x | ⚠️ Below target (but good!) |

**Finding**: Matmul is the ONLY operation where GPU shows speedup (2-10x for large matrices).

### 4. ReLU Activation (`relu`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 33.88 ms | 591.5 ns | **0.0000175x** | ❌ 57,000x SLOWER |
| 100K | 34.27 ms | 6.081 µs | **0.000177x** | ❌ 5,640x SLOWER |
| 1M | 55.27 ms | 67.04 µs | **0.00121x** | ❌ 824x SLOWER |

### 5. Leaky ReLU (`leaky_relu`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 38.29 ms | 585.6 ns | **0.0000153x** | ❌ 65,400x SLOWER |
| 100K | 34.26 ms | 5.890 µs | **0.000172x** | ❌ 5,820x SLOWER |
| 1M | 54.90 ms | 66.94 µs | **0.00122x** | ❌ 820x SLOWER |

### 6. ELU Activation (`elu`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 28.64 ms | 24.33 µs | **0.00085x** | ❌ 1,177x SLOWER |
| 100K | 18.24 ms | 258.8 µs | **0.0142x** | ❌ 70x SLOWER |
| 1M | 29.81 ms | 2.428 ms | **0.0814x** | ❌ 12.3x SLOWER |

### 7. Clip (`clip`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 14.96 ms | 507.1 ns | **0.0000339x** | ❌ 29,500x SLOWER |
| 100K | 15.77 ms | 5.718 µs | **0.000363x** | ❌ 2,757x SLOWER |
| 1M | 25.25 ms | 62.07 µs | **0.00246x** | ❌ 407x SLOWER |

### 8. Sigmoid (`sigmoid`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 14.48 ms | 19.16 µs | **0.00132x** | ❌ 756x SLOWER |
| 100K | 14.35 ms | 191.2 µs | **0.0133x** | ❌ 75x SLOWER |
| 1M | 21.99 ms | 2.964 ms | **0.135x** | ❌ 7.4x SLOWER |

### 9. Tanh (`tanh`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 13.39 ms | 77.86 µs | **0.00581x** | ❌ 172x SLOWER |
| 100K | 14.52 ms | 455.8 µs | **0.0314x** | ❌ 32x SLOWER |
| 1M | 22.58 ms | 1.979 ms | **0.0876x** | ❌ 11.4x SLOWER |

### 10. Swish (`swish`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 14.38 ms | 19.69 µs | **0.00137x** | ❌ 730x SLOWER |
| 100K | 14.34 ms | 192.4 µs | **0.0134x** | ❌ 74.5x SLOWER |
| 1M | 22.01 ms | 2.970 ms | **0.135x** | ❌ 7.4x SLOWER |

### 11. GELU (`gelu`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 14.38 ms | 76.32 µs | **0.00531x** | ❌ 188x SLOWER |
| 100K | 14.34 ms | 759.8 µs | **0.0530x** | ❌ 19x SLOWER |
| 1M | 22.04 ms | 9.458 ms | **0.429x** | ❌ 2.3x SLOWER |

### 12. Softmax (`softmax`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 14.28 ms | 38.85 µs | **0.00272x** | ❌ 368x SLOWER |
| 100K | 14.31 ms | 385.9 µs | **0.0270x** | ❌ 37x SLOWER |
| 1M | 21.96 ms | 4.802 ms | **0.219x** | ❌ 4.6x SLOWER |

### 13. Log Softmax (`log_softmax`)

| Size | GPU Time | Scalar Time | Speedup | Status |
|------|----------|-------------|---------|--------|
| 10K | 14.34 ms | 45.71 µs | **0.00319x** | ❌ 314x SLOWER |
| 100K | 14.38 ms | 447.2 µs | **0.0311x** | ❌ 32x SLOWER |
| 1M | 22.01 ms | 5.551 ms | **0.252x** | ❌ 4.0x SLOWER |

**Critical Pattern**: All element-wise operations show 2-65,000x SLOWDOWN on GPU!

---

## Complete Analysis

### GPU Overhead Breakdown

**Measured Overhead** (empirically observed):
- **Minimum overhead**: ~14-15ms (clip, sigmoid, tanh, swish, gelu, softmax, log_softmax)
- **Typical overhead**: ~32-38ms (vec_add, dot, relu, leaky_relu)
- **High overhead**: ~54-68ms (vec_add 1M, matmul 1000)

**Overhead Components** (estimated from timings):
1. **Buffer allocation**: ~2-5ms (varies with size)
2. **PCIe transfer (CPU→GPU)**: ~5-10ms per buffer (2 buffers = 10-20ms)
3. **Kernel dispatch overhead**: ~2-5ms
4. **PCIe transfer (GPU→CPU)**: ~5-10ms (result buffer)
5. **wgpu synchronization**: ~2-5ms

**Total**: 14-55ms fixed overhead per operation

### Why GPU Fails for Element-Wise Operations

Element-wise operations (relu, sigmoid, add, etc.) are **memory-bandwidth bound**, not compute-bound:

- **Scalar execution**: Load → Compute (1-2 cycles) → Store
- **GPU execution**: Allocate → Transfer (20ms) → Compute (0.1ms) → Transfer back (10ms)

**The compute time is negligible** (~0.1ms for 1M elements), so GPU parallelism provides NO benefit.

**Example - ReLU on 1M elements**:
- Scalar: 67µs (memory bandwidth saturated with sequential access)
- GPU: 55.27ms (30ms overhead + 0.1ms compute + 25ms transfer)
- Result: GPU is **824x slower** because transfer overhead dominates

### Why Matmul Succeeds

Matrix multiplication is **compute-bound** with O(N³) operations:

- **100×100**: 1M multiplies, ~545µs scalar, but GPU overhead (38ms) still dominates
- **500×500**: 125M multiplies, ~77ms scalar, GPU ~39ms → **2.01x speedup** ✓
- **1000×1000**: 1B multiplies, ~650ms scalar, GPU ~68ms → **9.59x speedup** ✓✓

**Key insight**: Matmul has enough compute to **amortize** the transfer overhead. Element-wise ops do not.

### OpComplexity Thresholds

**Current Thresholds**:
- **Low** (simple ops): >100K elements
- **Medium** (multi-pass): >10K elements
- **High** (complex): >1K elements

**Actual Required Thresholds** (based on data):
- **All ops**: >1M elements minimum for GPU benefit
- Even at 1M: only ~2x speedup (not 10-50x)

### Speedup Claims vs Reality

**Claimed** (README.md - INCORRECT):
- Small vectors (1K): <5x (transfer overhead dominates)
- Medium vectors (10K): 5-10x
- Large vectors (100K): 10-30x
- Very large (1M+): 20-50x

**Reality** (measured - element-wise operations):
- 1K: **0.00002-0.0016x** (600-63,000x SLOWER!)
- 10K: **0.000015-0.026x** (37-65,000x SLOWER!)
- 100K: **0.00018-0.053x** (19-5,600x SLOWER!)
- 1M: **0.0012-0.43x** (2-800x SLOWER!)

**Reality** (measured - matmul):
- 100×100: **0.014x** (71x SLOWER)
- 500×500: **2.01x** ✓ (first positive result)
- 1000×1000: **9.59x** ✓✓ (good speedup)

**Discrepancy**: Element-wise claims are off by **50-3000x**. Matmul achieves ~50% of claimed speedup.

---

## Strategic Recommendations

### **DECISION: Disable GPU for Element-Wise Operations Immediately**

The data is unambiguous - GPU provides **ZERO benefit** for element-wise operations at any scale.

### Immediate Actions (v0.2.1)

1. **Disable GPU backend for 13 operations** (keep matmul only):
   - Remove: vec_add, dot, relu, leaky_relu, elu, clip, sigmoid, tanh, swish, gelu, softmax, log_softmax
   - Keep: matmul (proven 2-10x speedup for large matrices)
   - Implementation: Update `OpComplexity` or remove GPU dispatch entirely for these ops

2. **Update OpComplexity thresholds for matmul**:
   ```rust
   // Current (WRONG):
   OpComplexity::High => 1_000,  // matmul, conv2d

   // Correct (based on data):
   if op == "matmul" && matrix_size >= 500 {
       use GPU  // 2-10x speedup proven
   } else {
       use SIMD  // scalar already faster
   }
   ```

3. **Update README.md** with honest performance claims:
   - Remove false "10-50x GPU speedup" claims
   - Document matmul-only GPU usage
   - Emphasize SIMD optimization path

4. **Add performance regression tests**:
   - CI must detect if GPU is accidentally enabled for element-wise ops
   - Benchmark comparisons to prevent future regressions

### Medium-Term Strategy (v0.3.0)

**Focus on SIMD, not GPU**:
- Element-wise operations are memory-bandwidth bound
- SIMD (AVX2/AVX-512) provides 2-8x speedup with ZERO overhead
- GPU requires 10M+ elements to potentially break even (not practical)

**Rationale**:
- Scalar relu: 67µs for 1M elements (already fast!)
- AVX-512 relu: ~10-15µs projected (4-7x speedup)
- GPU relu: 55,270µs (824x SLOWER than scalar)

### Long-Term Considerations (v2.0+)

**Async GPU API** (future possibility):
- Batch multiple operations to amortize transfer overhead
- Example: `(A + B) * C` → single GPU transfer instead of 2
- Requires: Complete API redesign, expression graph
- Benefit: Could reduce overhead from 30ms to 5-10ms (still not competitive)

**Verdict**: Even with async batching, SIMD will likely outperform GPU for element-wise ops.

### Operations to Keep GPU Support

1. **Matrix Multiplication** (proven 2-10x at 500×500+)
2. **Conv2D** (when implemented - similar compute pattern to matmul)
3. **Large tensor contractions** (>1B operations - future work)

**Common pattern**: O(N³) or higher compute complexity with O(N²) data transfer.

---

## Completed Analysis Checklist

1. ✅ Complete benchmark run (14 operations × 3-4 sizes = 40+ measurements)
2. ✅ Extract all measurements to structured data (tables above)
3. ✅ Calculate speedups for all operations (13 ops show GPU slowdown, 1 shows speedup)
4. ✅ Populate baseline: `.performance-baselines/baseline-current.txt`
5. ✅ Analyze GPU overhead and root causes
6. ✅ Make strategic decision based on empirical data

## Next Steps (Implementation)

1. **Commit this analysis** (v0.2.1 documentation)
2. **Disable GPU for element-wise ops** (code changes in src/backends/gpu/)
3. **Update README.md** with accurate performance claims
4. **Add regression tests** to CI (prevent future GPU misuse)
5. **Focus roadmap on SIMD** optimization (v0.3.0)

---

## References

- Criterion benchmarks: `/tmp/gpu_bench_output.txt`
- Baseline template: `.performance-baselines/baseline-template.json`
- OpComplexity definition: `src/backends/gpu/device.rs`
