# silu-kernel-v1

**Version:** 1.0.0

SiLU kernel — sigmoid linear unit activation function

## References

- Ramachandran et al. (2017) Searching for Activation Functions
- Elfwing et al. (2018) Sigmoid-Weighted Linear Units

## Dependency Graph

```mermaid
graph LR
    swiglu_kernel_v1["swiglu-kernel-v1"] --> silu_kernel_v1["silu-kernel-v1"]
```

## Equations

### sigmoid

$$
sigmoid(x) = 1 / (1 + \exp(-x))
$$

**Domain:** $x in R$

**Codomain:** $sigmoid(x) in (0, 1)$

**Invariants:**

- $sigmoid(0) = 0.5$
- $sigmoid(-x) = 1 - sigmoid(x) (symmetry)$

### silu

$$
SiLU(x) = x * sigmoid(x) = x / (1 + \exp(-x))
$$

**Domain:** $x in R$

**Codomain:** $SiLU(x) in (-0.279, +inf)$

**Invariants:**

- $SiLU(0) = 0 (zero preservation)$
- $SiLU(x) > -0.279 for all x (global minimum at x ~ -1.278)$
- $SiLU(x) ~ x for large positive x (asymptotic linearity)$
- $SiLU is monotonic for x > 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Zero preservation | $SiLU(0) = 0$ |
| 2 | bound | Global lower bound | $SiLU(x) > -0.279 for all x$ |
| 3 | monotonicity | Monotonic for positive inputs | $x > y > 0 implies SiLU(x) > SiLU(y)$ |
| 4 | equivalence | SIMD matches scalar within ULP |  |
| 5 | bound | Asymptotic linearity | $\|SiLU(x) - x\| < 0.01 for x > 10$ |

## Kernel Phases

1. **compute_sigmoid**: Compute sigmoid(x) = 1 / (1 + exp(-x)) — *output in (0, 1)*
2. **multiply**: Compute x * sigmoid(x) — *result > -0.279*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| silu | avx2 | `silu_avx2` |
| silu | ptx | `silu_ptx` |
| silu | scalar | `silu_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-SI-001 | Zero preservation | SiLU(0) = 0 exactly | Implementation does not handle zero correctly |
| FALSIFY-SI-002 | Global lower bound | SiLU(x) > -0.279 for x in [-1000, 1000] | Numerical instability near global minimum |
| FALSIFY-SI-003 | Positive monotonicity | SiLU(x) > SiLU(y) when x > y > 0 | Sigmoid saturation causing non-monotonicity |
| FALSIFY-SI-004 | SIMD equivalence | \|silu_avx2(x) - silu_scalar(x)\| < 8 ULP | SIMD exp approximation differs from scalar |
| FALSIFY-SI-005 | Asymptotic linearity | \|SiLU(x) - x\| < 0.01 for x > 10 | sigmoid(x) not approaching 1 for large x |
| FALSIFY-SI-006 | Boundary - large negative input | \|SiLU(x)\| < 0.01 for x < -10 | exp overflow for large negative input |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-SI-001 | SI-INV-001 | 1 | stub_float |
| KANI-SI-002 | SI-BND-001 | 8 | stub_float |
| KANI-SI-003 | SI-MON-001 | 8 | stub_float |
| KANI-SILU_K-004 | Zero preservation | 8 | exhaustive |
| KANI-SILU_K-005 | Global lower bound | 8 | stub_float |
| KANI-SILU_K-006 | Monotonic for positive inputs | 8 | exhaustive |
| KANI-SILU_K-007 | SIMD matches scalar within ULP | 8 | exhaustive |
| KANI-SILU_K-008 | Asymptotic linearity | 8 | exhaustive |

## QA Gate

**SiLU Contract** (F-SI-001)

Sigmoid linear unit activation quality gate

**Checks:** zero_preservation, lower_bound, positive_monotonicity, simd_equivalence

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

