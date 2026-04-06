# bias-add-v1

**Version:** 1.0.0

Bias addition kernel — broadcast bias vector over batch dimension

## References

- Standard neural network practice — affine transformation bias term

## Equations

### bias_add

$$
y[b, i] = x[b, i] + bias[i] for all b in [0, B), i in [0, D)
$$

**Domain:** $x in R^{B x D}, bias in R^D$

**Codomain:** $y in R^{B x D}$

**Invariants:**

- $Output shape equals input shape: shape(y) = shape(x) = (B, D)$
- $Zero-bias identity: y = x when bias = 0$
- `Additivity: bias_add(bias_add(x, b1), b2) = bias_add(x, b1 + b2)`
- $Broadcast: same bias vector applied to every batch element$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Shape preservation | $shape(bias_add(x, bias)) = shape(x) = (B, D)$ |
| 2 | invariant | Zero-bias identity | $bias_add(x, 0) = x for all x$ |
| 3 | invariant | Additivity | `bias_add(bias_add(x, b1), b2) = bias_add(x, b1 + b2)` |
| 4 | equivalence | SIMD matches scalar | `\|bias_add_avx2(x, b) - bias_add_scalar(x, b)\| = 0 (exact for addition)` |

## Kernel Phases

1. **broadcast_bias**: Replicate bias vector D across batch dimension B — *bias_expanded[b, i] = bias[i] for all b*
2. **elementwise_add**: Add expanded bias to input element-wise — *y[b, i] = x[b, i] + bias[i]*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| bias_add | avx2 | `bias_add_avx2` |
| bias_add | ptx | `bias_add_ptx` |
| bias_add | scalar | `bias_add_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-BA-001 | Shape preservation | shape(bias_add(x, bias)) = shape(x) for B in 1..64, D in 1..256 | Output buffer allocated with wrong dimensions |
| FALSIFY-BA-002 | Zero-bias identity | bias_add(x, zeros(D)) = x bitwise for all x | Bias addition modifies input even when bias is zero |
| FALSIFY-BA-003 | Additivity | bias_add(bias_add(x, b1), b2) = bias_add(x, b1 + b2) within 0 ULP | Floating-point addition order differs between paths |
| FALSIFY-BA-004 | SIMD equivalence | bias_add_avx2(x, b) = bias_add_scalar(x, b) exactly | SIMD lane ordering or alignment differs from scalar |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-BA-001 | BA-INV-001 | 8 | stub_float |
| KANI-BA-002 | BA-INV-002 | 8 | stub_float |
| KANI-BIAS_A-003 | Shape preservation | 8 | exhaustive |
| KANI-BIAS_A-004 | Zero-bias identity | 8 | exhaustive |
| KANI-BIAS_A-005 | Additivity | 8 | exhaustive |
| KANI-BIAS_A-006 | SIMD matches scalar | 8 | exhaustive |

## QA Gate

**Bias Add Contract** (F-BA-001)

Broadcast bias addition quality gate

**Checks:** shape_preservation, zero_identity, additivity, simd_equivalence

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

