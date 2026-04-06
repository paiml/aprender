# nf4-tensor-core-gemm-v1

**Version:** 1.0.0

NF4 tensor core GEMM — WMMA 16×16×16 with inline NF4 dequantization. Dequantizes NF4 blocks to FP16 in shared memory, uses tensor cores for matmul. Expected 10-40x compute improvement over naive tiled NF4 GEMM.


## References

- TensorCoreQ4KGemmKernel: proven WMMA+quantized GEMM pattern in trueno
- NVIDIA WMMA: 16×16×16 FP16 → FP32 accumulate

## Equations

### naive_nf4_gemm

$$
Current: 1 thread per output element, scalar FMA
Compute: M×N×K scalar FMA operations at ~2 TFLOPS (Ada SIMD)
For Qwen 1.5B (M=2048, K=1536, N=1536):
  FLOPs = 2 × 2048 × 1536 × 1536 = 9.66 GFLOP
  Time at 2 TFLOPS = 4.8 ms per GEMM

$$

**Domain:** $M, K, N in u32$

### tensor_core_nf4_gemm

$$
Proposed: WMMA 16×16×16 tiles, FP16 compute \to FP32 accumulate
Compute: same FLOPs but at ~83 TFLOPS (Ada tensor cores)
For Qwen 1.5B: 9.66 GFLOP at 83 TFLOPS = 0.12 ms per GEMM
Speedup: ~40x compute (if not memory-bound)

$$

**Domain:** $same$

**Invariants:**

- $NF4 dequant to FP16 in shared memory (16 values per block)$
- $WMMA load from shared memory (row-major A, col-major B)$
- $FP32 accumulator written to global memory$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | equivalence | Tensor core NF4 GEMM matches naive NF4 GEMM | $\|tc_gemm(A, B_nf4) - naive_gemm(A, B_nf4)\| < \varepsilon element-wise$ |
| 2 | bound | Throughput improvement via tensor cores | $throughput(tc_gemm) >= 5 * throughput(naive_gemm)$ |
| 3 | invariant | NF4 dequant to FP16 in shared memory before WMMA load | `dequant_location == shared_memory AND wmma_input_type == fp16` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| F-NF4-TC-001 | Tensor core NF4 GEMM matches naive within 1e-3 | Max absolute difference < 1e-3 at Qwen dimensions | FP16 accumulation precision loss — check if FP32 accumulate is used in WMMA |
| F-NF4-TC-002 | Throughput improvement via tensor cores | Throughput >= 5x naive NF4 GEMM | Memory-bound at these dimensions — profile with nsys to check compute vs memory |
| F-NF4-TC-003 | NF4 dequant to FP16 in shared memory before WMMA load | Dequantized FP16 values in shared memory match standalone NF4 dequant | Shared memory bank conflicts corrupting dequant values — check padding and stride |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-NF4TC-001 | Tensor core GEMM equivalence | 4 | stub_float |
| KANI-NF4TC-002 | NF4 dequant to FP16 precision | 8 | exhaustive |

## QA Gate

**nf4-tensor-core-gemm-v1 Contract** (F-NF4TC-001)

**Checks:** validation, falsification

