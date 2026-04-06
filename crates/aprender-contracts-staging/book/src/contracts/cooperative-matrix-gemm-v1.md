# cooperative-matrix-gemm-v1

**Version:** 1.0.0

Cooperative matrix GEMM — hardware tensor core acceleration via VK_KHR_cooperative_matrix (wgpu 29.0+). Replaces software tiled GEMM (375 GFLOPS) with hardware WMMA (expected 1000+ GFLOPS on GB10).


## References

- VK_KHR_cooperative_matrix Vulkan extension
- wgpu v29.0.0 cooperative matrix support (2026-03-19)
- NVIDIA Blackwell GB10: BF16+FP8 cooperative matrix, revision 2

## Equations

### cooperative_gemm

$$
C[m,n] = \alpha * \sum_k A[m,k] * B[k,n] + \beta * C[m,n]
$$

**Domain:** $A \in \mathbb{R}^{M×K}, B \in \mathbb{R}^{K×N}, \alpha,\beta \in \mathbb{R}$

**Codomain:** $C \in \mathbb{R}^{M×N}$

**Invariants:**

- $Result matches software tiled GEMM within \varepsilon < 1e-3 (f32)$
- $F16 input, F32 accumulation (GB10 config 3: M=16 K=16 N=16)$

### f16_error_bound

```
|C_f32_accum - C_exact| ≤ K * ε_f16 * max|A| * max|B|
```

**Domain:** $K reduction dimension, \varepsilon_f16 = 2^{-10}$

**Codomain:** $error bound \in \mathbb{R}\geq0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | postcondition | Parity with tiled reference | $\|coop - tiled\| < 1e-3$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-COOP-001 | parity_with_tiled | max \|coop - tiled\| < 1e-3 | Cooperative matrix GEMM diverges from tiled reference |
| FALSIFY-COOP-002 | throughput_improvement | coop GFLOPS > 2 * tiled GFLOPS | Cooperative matrix provides no performance benefit |
| FALSIFY-COOP-003 | fallback | Falls back to tiled GEMM, no crash | Missing fallback causes crash on unsupported hardware |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-COOP-001 | parity_with_tiled | 8 | bounded_int |

## QA Gate

**Cooperative Matrix GEMM Contract** (F-COOP-GEMM-001)

Quality gate for cooperative matrix GEMM hardware tensor core acceleration

**Checks:** validation, falsification

