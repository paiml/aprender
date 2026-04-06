# Phase 4: Implement -- Scalar Reference then SIMD

## Order of Implementation

```
1. Scalar reference (ground truth)
   ↓
2. Tests pass for scalar
   ↓
3. SIMD variant (AVX2, AVX-512, NEON)
   ↓
4. SIMD tests pass AND SIMD matches scalar within ULP tolerance
   ↓
5. Dispatch table updated in contract YAML
```

## The Scalar Reference is Sacrosanct

The scalar implementation is the **mathematical reference**. It must be:
- As close to the paper's equation as possible
- Readable (no bit tricks, no manual unrolling)
- Correct to within f32 arithmetic limits

The SIMD implementation is an **optimization** of the scalar reference. It is
allowed to diverge only within the contract's ULP tolerance. If SIMD and scalar
disagree beyond tolerance, the SIMD implementation is wrong -- not the scalar.

This mirrors the existing pattern in `quantized-dot-product-v1.yaml`:
```yaml
kernel_correctness:
  description: "Every SIMD kernel must produce output within ULP_TOLERANCE of scalar reference"
  check: "contract_tests::FALSIFY-QDOT-001 — proptest with random weights and activations"
  severity: "ERROR"
```

## SIMD Dispatch Table

Following the established pattern, every contract has an exhaustive dispatch
table. No `_ =>` catch-all allowed (lesson from `tensor-layout-v1.yaml`
PMAT-232).

```yaml
simd_dispatch:
  softmax:
    scalar: "softmax_scalar"
    avx2: "softmax_avx2"
    avx512: "softmax_avx512"
    neon: "softmax_neon"
```
