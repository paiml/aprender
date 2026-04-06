# alibi-kernel-v1

**Version:** 1.0.0

ALiBi kernel — Attention with Linear Biases positional encoding

## References

- Press et al. (2022) Train Short, Test Long

## Equations

### alibi_bias

$$
scores[i,j] += -m_h * |i - j|
$$

**Domain:** $i, j in {0, ..., seq_len - 1}, h in {0, ..., H - 1}$

**Codomain:** $bias in (-inf, 0]$

**Invariants:**

- $bias <= 0 for all positions (scores only decrease)$
- $bias = 0 when i = j (self-position has zero penalty)$
- $bias decreases linearly with distance |i - j|$
- $future positions (j > i) receive -inf bias in causal mode$

### alibi_slopes

$$
m_h = 2^(-8h/H)
$$

**Domain:** $h in {0, ..., H - 1}, H >= 1$

**Codomain:** $m_h in (0, 1]$

**Invariants:**

- $m_h > 0 for all heads (slopes are strictly positive)$
- $m_0 > m_1 > ... > m_{H-1} (slopes decrease with head index)$
- $m_0 = 2^(-8/H) (first head slope)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Negative bias | $-m_h * \|i - j\| <= 0 for all i, j, h$ |
| 2 | bound | Slope positivity | $m_h = 2^(-8h/H) > 0 for all h in {0, ..., H-1}$ |
| 3 | invariant | Causal consistency | $j > i implies scores[i,j] = -inf in causal mode$ |
| 4 | monotonicity | Head-monotonic slopes | $h1 < h2 implies m_{h1} > m_{h2}$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **compute_slopes**: Compute m_h = 2^(-8h/H) for each attention head — *all slopes in (0, 1]*
2. **compute_bias**: Compute -m_h * |i - j| for each position pair — *all biases <= 0*
3. **apply_causal_mask**: Set future positions to -inf in causal mode — *j > i implies score = -inf*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| alibi_bias | avx2 | `alibi_bias_avx2` |
| alibi_bias | ptx | `alibi_bias_ptx` |
| alibi_bias | scalar | `alibi_bias_scalar` |
| alibi_slopes | avx2 | `alibi_slopes_avx2` |
| alibi_slopes | ptx | `alibi_slopes_ptx` |
| alibi_slopes | scalar | `alibi_slopes_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-AL-001 | Negative bias | -m_h * \|i - j\| <= 0 for all i, j, h | Slope sign error causing positive bias values |
| FALSIFY-AL-002 | Slope positivity | m_h > 0 for all h in {0, ..., H-1} | Exponentiation underflow producing zero or negative slopes |
| FALSIFY-AL-003 | Causal consistency | scores[i,j] = -inf when j > i in causal mode | Causal mask not applied or applied to wrong positions |
| FALSIFY-AL-004 | Head-monotonic slopes | m_{h} > m_{h+1} for consecutive heads | Slope formula error breaking monotonic ordering |
| FALSIFY-AL-005 | SIMD equivalence | \|alibi_bias_avx2(x) - alibi_bias_scalar(x)\| < 8 ULP | SIMD exp2 approximation differs from scalar pow |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-AL-001 | AL-BND-001 | 8 | stub_float |
| KANI-AL-002 | AL-BND-002 | 8 | stub_float |
| KANI-ALIBI_-003 | Negative bias | 8 | exhaustive |
| KANI-ALIBI_-004 | Slope positivity | 8 | exhaustive |
| KANI-ALIBI_-005 | Causal consistency | 8 | exhaustive |
| KANI-ALIBI_-006 | Head-monotonic slopes | 8 | exhaustive |
| KANI-ALIBI_-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**ALiBi Contract** (F-AL-001)

Attention with Linear Biases positional encoding quality gate

**Checks:** negative_bias, slope_positivity, causal_consistency, head_monotonic_slopes, simd_equivalence

**Pass criteria:** All 5 falsification tests pass + Kani harnesses verify

