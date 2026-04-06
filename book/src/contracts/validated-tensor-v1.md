# validated-tensor-v1

**Version:** 1.0.0

Validated tensor type invariants (embedding density, NaN/Inf rejection, L2 norm)

## References

- PMAT-235 Compile-time Poka-Yoke
- Qwen2.5-Coder Showcase Spec §15.3

## Dependency Graph

```mermaid
graph LR
    gguf_format_safety_v1["gguf-format-safety-v1"] --> validated_tensor_v1["validated-tensor-v1"]
    safetensors_format_safety_v1["safetensors-format-safety-v1"] --> validated_tensor_v1["validated-tensor-v1"]
```

## Equations

### density_gate

```
density(E) = count(E_ij != 0) / numel(E)
```

**Domain:** $E \in \mathbb{R}^{m×n}$

**Codomain:** $density \in [0, 1]$

**Invariants:**

- $density > 0.055 for valid embeddings (reject >= 94.5\% zeros)$
- $Fully dense matrix has density = 1.0$

### l2_norm_nondegeneracy

$$
forall i: ||E[i,:]||_2 > 0
$$

**Domain:** $E \in \mathbb{R}^{m×n}, m = vocabulary size$

**Invariants:**

- $No all-zero rows (every token has a non-trivial embedding)$

### nan_inf_rejection

```
count(isnan(E)) == 0 AND count(isinf(E)) == 0
```

**Domain:** $E \in \mathbb{R}^{m×n}$

**Invariants:**

- $No NaN values present$
- $No Inf values present$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Density gate | $density(E) > 0.055 for valid embeddings$ |
| 2 | invariant | NaN/Inf rejection | `count(isnan) == 0 AND count(isinf) == 0` |
| 3 | invariant | L2 norm non-degeneracy | $forall row i: \|\|E[i,:]\|\|_2 > 0$ |
| 4 | equivalence | SIMD validation equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-VT-001 | Density gate | Random normal embeddings have density >> 0.055 | Threshold too high or counting error |
| FALSIFY-VT-002 | NaN/Inf rejection | Injecting NaN is detected | Scanning misses NaN/Inf |
| FALSIFY-VT-003 | L2 norm | Zero rows are detected | Row norm calculation error |
| FALSIFY-VT-004 | SIMD validation equivalence | SIMD validation matches scalar |  compare scalar vs SIMD validation:SIMD validation rejects different inputs |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-VT-001 | VT-INV-001 | 4 | bounded_int |
| KANI-VALIDA-002 | Density gate | 8 | exhaustive |
| KANI-VALIDA-003 | NaN/Inf rejection | 8 | exhaustive |
| KANI-VALIDA-004 | L2 norm non-degeneracy | 8 | exhaustive |
| KANI-VALIDA-005 | SIMD validation equivalence | 8 | exhaustive |

## QA Gate

**Validated Tensor Contract** (F-VT-001)

Tensor validation quality gate

**Checks:** density_gate, nan_inf_rejection, l2_norm_nondegeneracy

**Pass criteria:** All 4 falsification tests pass

