# transpose-kernel-v1

**Version:** 1.0.0

Matrix transpose kernel — AVX2 8×8 in-register shuffle with cache blocking

## References

- Lam, Rothberg & Wolf (1991) Cache Performance of Blocked Algorithms. ASPLOS IV
- Intel 64 and IA-32 Architectures Optimization Reference Manual §11.12

## Equations

### transpose

$$
B[j * rows + i] = A[i * cols + j]
$$

**Domain:** $A \in \mathbb{R}^{rows×cols} (row-major)$

**Codomain:** $B \in \mathbb{R}^{cols×rows} (row-major)$

**Invariants:**

- $B has shape (cols, rows)$
- $Transpose is an involution: transpose(transpose(A)) = A$
- $trace(A) = trace(transpose(A)) for square A$
- $det(A) = det(transpose(A))$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Shape correctness | $shape(transpose(A[m,n])) = (n, m)$ |
| 2 | idempotency | Involution (self-inverse) | $transpose(transpose(A)) = A (bitwise exact)$ |
| 3 | equivalence | AVX2 matches scalar | $\|transpose_avx2(A) - transpose_scalar(A)\| = 0 (bitwise exact)$ |
| 4 | invariant | Element correctness | $B[j][i] = A[i][j] for all valid i,j$ |
| 5 | invariant | All elements transposed | $No element lost or duplicated — bijection on index pairs$ |

## Kernel Phases

1. **outer_blocking**: Partition matrix into 8×8 blocks for register-width transpose — *All blocks cover the full matrix (remainder handled separately)*
2. **avx2_8x8_microkernel**: In-register 8×8 transpose via 3-phase AVX2 shuffle/permute — *8 loads + 24 shuffles + 8 stores = 40 ops for 64 elements*
3. **remainder**: Scalar transpose for rows%8 and cols%8 edges — *Edge elements correct — no out-of-bounds access*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| transpose | avx2 | `transpose_avx2` |
| transpose | scalar | `transpose_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-TP-001 | Element correctness | transpose(A)[j][i] == A[i][j] for random A | Index computation error in micro-kernel |
| FALSIFY-TP-002 | Involution | transpose(transpose(A)) == A (bitwise exact) | Asymmetric load/store pattern in shuffle phases |
| FALSIFY-TP-003 | Non-8-aligned dimensions | Correct for 7×13, 17×3, 1×N, N×1 | Remainder handling bug — edge elements skipped |
| FALSIFY-TP-004 | AVX2 vs scalar parity | AVX2 output == scalar output (bitwise exact, no FP rounding) | Shuffle phase error — wrong permute immediate |
| FALSIFY-TP-005 | Identity matrix | transpose(I) == I for square identity | Off-by-one in diagonal element indexing |
| FALSIFY-TP-006 | Attention shape | 2048×128 transpose matches naive reference | Block boundary error at non-square dimensions |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-TP-001 | TP-INV-001 | 16 | exhaustive |
| KANI-TP-002 | TP-IDEMP-001 | 8 | bounded_int |
| KANI-TRANSP-003 | Shape correctness | 8 | exhaustive |
| KANI-TRANSP-004 | Involution (self-inverse) | 8 | exhaustive |
| KANI-TRANSP-005 | AVX2 matches scalar | 8 | exhaustive |
| KANI-TRANSP-006 | Element correctness | 8 | exhaustive |
| KANI-TRANSP-007 | All elements transposed | 8 | exhaustive |

## QA Gate

**Transpose Kernel Contract** (F-TP-001)

**Checks:** element_correctness, involution, simd_equivalence, remainder_handling

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

