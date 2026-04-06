# format-parity-v1

**Version:** 1.0.0

Format parity — cross-format tensor equivalence (GGUF, SafeTensors, APR)

## References

- APR-SPEC.md — APR format specification
- GGUF spec — GGML unified format
- SafeTensors spec — Hugging Face safe serialization
- contracts/tensor-layout-v1.yaml — layout contract (source of truth)

## Equations

### element_count

```
product(gguf_shape) == product(apr_shape)
```

**Domain:** $gguf_shape, apr_shape \in \mathbb{Z}^{+n}$

**Invariants:**

- $Total element count preserved across format conversion$
- $No data is lost or duplicated$

### identity_1d

```
1D tensors: apr_shape == gguf_shape (no transpose)
```

**Domain:** $shape \in \mathbb{Z}^{+1}$

**Invariants:**

- $Bias vectors and 1D tensors are identity-mapped$
- $Only 2D+ tensors require transpose$

### name_bijection

$$
tensor_template defines 1:1 mapping between format names
$$

**Domain:** $tensor names across formats$

**Invariants:**

- $Every GGUF tensor has exactly one APR counterpart$
- $Mapping is invertible$

### transpose_involution

```
swap(swap(shape)) == shape
```

**Domain:** $shape \in \mathbb{Z}^{+} × \mathbb{Z}^{+} (2D tensors)$

**Invariants:**

- $Transpose is its own inverse$
- $GGUF\to APR\to GGUF roundtrip preserves shape$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Transpose involution | `swap(swap([a, b])) == [a, b]` |
| 2 | invariant | Element count preserved | `product(gguf_shape) == product(apr_shape) for all tensors` |
| 3 | invariant | 1D no transpose | `len(shape) == 1 ⟹ apr_shape == gguf_shape` |
| 4 | equivalence | Roundtrip equivalence | $\|convert(convert(tensor, GGUF\to APR), APR\to GGUF) - tensor\| < \varepsilon$ |
| 5 | equivalence | SIMD format equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-FP-001 | Transpose involution | swap(swap(shape)) == shape for all 2D shapes | Double-transpose not identity |
| FALSIFY-FP-002 | Element count | product of shape dims preserved after swap | Shape transformation changes element count |
| FALSIFY-FP-003 | 1D identity | 1D shapes pass through unchanged | 1D tensors incorrectly transposed |
| FALSIFY-FP-004 | Roundtrip | GGUF→APR→GGUF roundtrip is lossless | Data corruption in format conversion |
| FALSIFY-FP-005 | SIMD format equivalence | SIMD format check matches scalar |  compare scalar vs SIMD format parity:SIMD format check diverges |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-FP-001 | FP-INV-001 | 4 | bounded_int |
| KANI-FORMAT-002 | Transpose involution | 8 | exhaustive |
| KANI-FORMAT-003 | Element count preserved | 8 | exhaustive |
| KANI-FORMAT-004 | 1D no transpose | 8 | exhaustive |
| KANI-FORMAT-005 | Roundtrip equivalence | 8 | exhaustive |
| KANI-FORMAT-006 | SIMD format equivalence | 8 | exhaustive |

## QA Gate

**Format Parity Contract** (F-FP-001)

Cross-format tensor equivalence quality gate

**Checks:** transpose_involution, element_count, identity_1d, roundtrip

**Pass criteria:** All 5 falsification tests pass

