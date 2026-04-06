# linear-projection-v1

**Version:** 1.0.0

Linear projection — matrix multiply with optional bias (dense layer forward pass)

## References

- Bishop (2006) Pattern Recognition and Machine Learning

## Equations

### linear_forward

$$
y = x @ W^T + b
$$

**Domain:** $x in R^{batch x d_in}, W in R^{d_out x d_in}, b in R^{d_out}$

**Codomain:** $y in R^{batch x d_out}$

**Invariants:**

- $y.shape = (batch, d_out) for x.shape = (batch, d_in)$
- $y[i] = sum_j(x[i][j] * W[k][j]) + b[k] for each output element$
- $f(alpha * x) + b = alpha * (x @ W^T) + b (scaling with bias)$

### linear_no_bias

$$
y = x @ W^T
$$

**Domain:** $x in R^{batch x d_in}, W in R^{d_out x d_in}$

**Codomain:** $y in R^{batch x d_out}$

**Invariants:**

- $f(alpha * x) = alpha * f(x) (homogeneity / linearity)$
- $f(0) = 0 (zero preservation without bias)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Output shape correctness | $y.shape = (batch, d_out) for x.shape = (batch, d_in), W.shape = (d_out, d_in)$ |
| 2 | linearity | Homogeneity without bias | `linear_no_bias(alpha * x, W) = alpha * linear_no_bias(x, W)` |
| 3 | invariant | Bias additivity | `linear_forward(x, W, b) = linear_no_bias(x, W) + b (broadcast)` |
| 4 | invariant | Zero input produces bias | $linear_forward(0, W, b) = b (broadcast to batch)$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **matmul**: Compute x @ W^T via tiled matrix multiplication — *intermediate.shape = (batch, d_out)*
2. **bias_add**: Add bias vector b to each row of intermediate — *output[i] = intermediate[i] + b for all rows i*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| linear_forward | avx2 | `linear_forward_avx2` |
| linear_forward | ptx | `linear_forward_ptx` |
| linear_forward | scalar | `linear_forward_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-LP-001 | Output shape correctness | y.shape = (batch, d_out) for all valid dimension combinations | Matmul dimension mismatch or transposition error |
| FALSIFY-LP-002 | Homogeneity without bias | linear_no_bias(alpha * x, W) = alpha * linear_no_bias(x, W) within 4 ULP | Floating-point accumulation order violates linearity |
| FALSIFY-LP-003 | Bias additivity | linear_forward(x, W, b) - linear_no_bias(x, W) = b (broadcast) | Bias fused into matmul incorrectly |
| FALSIFY-LP-004 | Zero input produces bias | linear_forward(0, W, b) = b for every row | Matmul of zero not producing zero intermediate |
| FALSIFY-LP-005 | SIMD equivalence | \|linear_forward_avx2(x, W, b) - linear_forward_scalar(x, W, b)\| < 4 ULP | SIMD FMA instruction accumulation differs from scalar |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-LP-001 | LP-SHP-001 | 4 | stub_float |
| KANI-LP-002 | LP-LIN-001 | 4 | stub_float |
| KANI-LINEAR-003 | Output shape correctness | 8 | exhaustive |
| KANI-LINEAR-004 | Homogeneity without bias | 8 | exhaustive |
| KANI-LINEAR-005 | Bias additivity | 8 | exhaustive |
| KANI-LINEAR-006 | Zero input produces bias | 8 | exhaustive |
| KANI-LINEAR-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**Linear Projection Contract** (F-LP-001)

Dense layer forward pass (matmul + bias) quality gate

**Checks:** output_shape, homogeneity, bias_additivity, zero_input_bias, simd_equivalence

**Pass criteria:** All 5 falsification tests pass + Kani harnesses verify

