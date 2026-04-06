# activation-kernel-v1

**Version:** 1.0.0

Activation functions — GELU, SiLU/Swish, ReLU kernels

## References

- Hendrycks & Gimpel (2016) Gaussian Error Linear Units (GELUs)
- Ramachandran et al. (2017) Searching for Activation Functions (SiLU)
- Nair & Hinton (2010) Rectified Linear Units Improve Restricted Boltzmann Machines

## Equations

### gelu

$$
GELU(x) = x · \Phi(x) \approx 0.5x(1 + tanh(√(2/\pi)(x + 0.044715x³)))
$$

**Domain:** $x \in \mathbb{R}$

**Codomain:** $\mathbb{R}$

**Invariants:**

- $GELU(x) \to x as x \to +∞$
- $GELU(x) \to 0 as x \to -∞$
- $GELU(0) = 0$

### relu

$$
ReLU(x) = max(0, x)
$$

**Domain:** $x \in \mathbb{R}$

**Codomain:** $[0, ∞)$

**Invariants:**

- $ReLU(x) \geq 0 (non-negativity)$
- $ReLU(x) = x for x > 0$
- $ReLU(x) = 0 for x \leq 0$

### silu

$$
SiLU(x) = x · \sigma(x) = x / (1 + \exp(-x))
$$

**Domain:** $x \in \mathbb{R}$

**Codomain:** $\mathbb{R}$

**Invariants:**

- $SiLU(x) \to x as x \to +∞$
- $SiLU(x) \to 0 as x \to -∞$
- $SiLU(0) = 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | GELU at zero | $GELU(0) = 0$ |
| 2 | bound | GELU approximation error | $\|GELU_approx(x) - GELU_exact(x)\| < \varepsilon for \|x\| < 10$ |
| 3 | invariant | SiLU at zero | $SiLU(0) = 0$ |
| 4 | monotonicity | ReLU monotonic | $x \geq y ⟹ ReLU(x) \geq ReLU(y)$ |
| 5 | invariant | ReLU non-negative | $ReLU(x) \geq 0 for all x$ |
| 6 | equivalence | SIMD matches scalar |  |

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| activation | avx2 | `activation_avx2` |
| activation | ptx | `activation_ptx` |
| activation | scalar | `activation_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-ACT-001 | GELU zero | GELU(0) = 0 | Constant term in approximation |
| FALSIFY-ACT-002 | GELU approximation | \|GELU_fast(x) - GELU_ref(x)\| < 1e-4 for \|x\| < 10 | tanh approximation coefficients wrong |
| FALSIFY-ACT-003 | SiLU zero | SiLU(0) = 0 | sigmoid(0) not exactly 0.5 |
| FALSIFY-ACT-004 | ReLU non-negative | ReLU(x) ≥ 0 for all x including -0.0 | Signed zero handling |
| FALSIFY-ACT-005 | SIMD equivalence | \|act_avx2(x) - act_scalar(x)\| < 4 ULP for each activation | SIMD fast-math approximation diverges |
| FALSIFY-ACT-006 | ReLU monotonic | x1 <= x2 => ReLU(x1) <= ReLU(x2) | ReLU implementation not monotonic |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-ACT-001 | ACT-INV-001 | 32 | exhaustive |
| KANI-ACT-002 | ACT-MON-001 | 32 | exhaustive |
| KANI-ACTIVA-003 | GELU at zero | 8 | exhaustive |
| KANI-ACTIVA-004 | GELU approximation error | 8 | exhaustive |
| KANI-ACTIVA-005 | SiLU at zero | 8 | exhaustive |
| KANI-ACTIVA-006 | ReLU monotonic | 8 | exhaustive |
| KANI-ACTIVA-007 | ReLU non-negative | 8 | exhaustive |
| KANI-ACTIVA-008 | SIMD matches scalar | 8 | exhaustive |

## QA Gate

**Activation Contract** (F-ACT-001)

**Checks:** gelu_zero, relu_nonnegative, simd_equivalence

**Pass criteria:** All 6 falsification tests pass + Kani verifies ReLU

