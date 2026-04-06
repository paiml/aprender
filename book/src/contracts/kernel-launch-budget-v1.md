# kernel-launch-budget-v1

**Version:** 1.0.0

GPU kernel launch count budget for transformer inference

## References

- Qwen2.5-Coder Showcase Spec §13.10 — kernel launch decomposition
- Qwen3 Performance Parity Spec — bsum instruction budget

## Equations

### bsum_budget

$$
waste = L * P * ceil(D/256) * C
$$

**Domain:** `L=layers, P=passes_per_layer, D=hidden_dim, C=cycles_per_bsum`

**Codomain:** $waste \in \mathbb{Z}^{+} (wasted cycles)$

**Invariants:**

- $Waste proportional to layer count$
- $Waste proportional to hidden dim (via ceil)$

### per_layer_decomposition

$$
12 = 2(norm) + 5(matmul) + 1(rope) + 1(attn) + 1(swiglu) + 2(residual)
$$

**Domain:** $fixed transformer architecture$

**Invariants:**

- $Decomposition sums to 12$
- $Each component count >= 1$

### per_token_launches

$$
kernel_launches(L) = 12 * L + 2
$$

**Domain:** $L \in \mathbb{Z}^{+} (number of transformer layers)$

**Codomain:** $kernel_launches \in \mathbb{Z}^{+}$

**Invariants:**

- $Linear in L$
- $Minimum: 14 launches for L=1$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Per-token formula | $kernel_launches(L) = 12 * L + 2 for all L >= 1$ |
| 2 | invariant | Decomposition sum | $2 + 5 + 1 + 1 + 1 + 2 = 12$ |
| 3 | monotonicity | Launch count monotonic | $L1 < L2 => kernel_launches(L1) < kernel_launches(L2)$ |
| 4 | equivalence | SIMD kernel equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-KL-001 | Per-token formula | kernel_launches agrees with formula for random L | Formula constant wrong |
| FALSIFY-KL-002 | Decomposition | Component sum = 12 | Missing or extra kernel in decomposition |
| FALSIFY-KL-003 | Monotonicity | More layers => more launches | Non-monotonic launch count |
| FALSIFY-KL-004 | SIMD kernel equivalence | SIMD kernel budget matches scalar |  compare scalar vs SIMD budget calc:SIMD budget computation differs |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-KL-001 | KL-INV-001 | 8 | bounded_int |
| KANI-KERNEL-002 | Per-token formula | 8 | exhaustive |
| KANI-KERNEL-003 | Decomposition sum | 8 | stub_float |
| KANI-KERNEL-004 | Launch count monotonic | 8 | exhaustive |
| KANI-KERNEL-005 | SIMD kernel equivalence | 8 | exhaustive |

## QA Gate

**Kernel Launch Budget Contract** (F-KL-001)

GPU kernel launch count quality gate

**Checks:** per_token_formula, decomposition_sum, monotonicity

**Pass criteria:** All 4 falsification tests pass

