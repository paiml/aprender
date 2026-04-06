# q4k-q6k-superblock-v1

**Version:** 1.0.0

Q4K and Q6K quantization superblock layout and dequantization formula

## References

- GGML Q4_K_M/Q6_K format specification
- Qwen2.5-Coder Showcase Spec Appendix F, §11.5
- Qwen3 Performance Parity Spec — Dot Product Algebra

## Equations

### bsum

$$
bsum_j = sum(q_i for i in block_j)
$$

**Domain:** $Block of quantized values$

**Invariants:**

- $bsum depends only on input x, not on weight W$

### dequantization

$$
x_i = d * s_j * q_i - dmin * m_j
$$

**Domain:** `d,dmin ∈ f16, s_j,m_j ∈ uint6, q_i ∈ uint4/uint6`

**Invariants:**

- $Output is finite for valid superblock$
- $has_dmin=false => offset term = 0$

### q4k_superblock

$$
sizeof(Q4K_superblock) = 2(d) + 2(dmin) + 12(scales) + 128(quants) = 144 bytes
$$

**Domain:** $256 elements per superblock$

**Invariants:**

- $144 bytes encodes exactly 256 elements$
- $Effective bits per weight: 144*8/256 = 4.5$

### q6k_superblock

$$
sizeof(Q6K_superblock) = 128(ql) + 64(qh) + 16(scales) + 2(d) = 210 bytes
$$

**Domain:** $256 elements per superblock$

**Invariants:**

- $210 bytes encodes exactly 256 elements$
- $Effective bits per weight: 210*8/256 = 6.5625$

### total_bytes

$$
total_bytes(rows, cols) = rows * ceil(cols / 256) * block_size
$$

**Domain:** $rows > 0, cols > 0, block_size \in {144, 210}$

**Invariants:**

- $total_bytes proportional to rows$
- $total_bytes monotonically increases with cols$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Q4K superblock size | $2 + 2 + 12 + 128 = 144$ |
| 2 | invariant | Q6K superblock size | $128 + 64 + 16 + 2 = 210$ |
| 3 | monotonicity | Total bytes monotonic | $cols1 < cols2 => total_bytes(r, cols1) <= total_bytes(r, cols2)$ |
| 4 | invariant | Dequant produces finite | $x_i is finite for valid superblock inputs$ |
| 5 | invariant | Offset vanishing | $has_dmin=false => offset_term = 0$ |
| 6 | invariant | bsum weight independence | $bsum depends on x only$ |
| 7 | equivalence | SIMD dequant equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-QS-001 | Superblock sizes | Q4K=144, Q6K=210 | Field size constant wrong |
| FALSIFY-QS-002 | Total bytes monotonic | More cols => more bytes | ceil rounding bug |
| FALSIFY-QS-003 | Dequant finite | All dequantized values finite | NaN from scale overflow |
| FALSIFY-QS-004 | Offset vanishing | Scale-only format has zero offset term | Offset term present in scale-only format |
| FALSIFY-QS-005 | bsum weight independence | bsums depend only on activations, not weights | bsum incorrectly depends on weight values |
| FALSIFY-QS-006 | Byte layout consistency | Superblock byte count matches format spec | Padding or alignment error in superblock |
| FALSIFY-QS-007 | SIMD dequant equivalence | SIMD dequantization matches scalar | SIMD path produces different dequant values |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-QS-001 | QS-INV-001 | 1 | exhaustive |
| KANI-Q4K_Q6-002 | Q4K superblock size | 8 | exhaustive |
| KANI-Q4K_Q6-003 | Q6K superblock size | 8 | exhaustive |
| KANI-Q4K_Q6-004 | Total bytes monotonic | 8 | exhaustive |
| KANI-Q4K_Q6-005 | Dequant produces finite | 8 | exhaustive |
| KANI-Q4K_Q6-006 | Offset vanishing | 8 | exhaustive |
| KANI-Q4K_Q6-007 | bsum weight independence | 8 | stub_float |
| KANI-Q4K_Q6-008 | SIMD dequant equivalence | 8 | exhaustive |

## QA Gate

**Q4K/Q6K Superblock Contract** (F-QS-001)

Quantization layout quality gate

**Checks:** q4k_size, q6k_size, total_bytes_monotonic, dequant_finite, offset_vanishing, bsum_independence

**Pass criteria:** All 7 falsification tests pass

