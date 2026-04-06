# lora-algebra-v1

**Version:** 1.0.0

SVD LoRA extraction and merge strategy algebra

## References

- Hu et al. (2021) LoRA: Low-Rank Adaptation
- Eckart-Young-Mirsky theorem (1936)
- Yadav et al. (2023) TIES-Merging
- Yu et al. (2023) DARE: Language Models are Super Mario
- Qwen3.5 Fine-Tune Spec Phase 2

## Dependency Graph

```mermaid
graph LR
    classification_finetune_v1["classification-finetune-v1"] --> lora_algebra_v1["lora-algebra-v1"]
```

## Equations

### dare_unbiased

$$
E[DARE(delta, p)] = delta
$$

**Domain:** $p \in (0, 1), dropout probability$

**Codomain:** $Expected value preserves delta$

**Invariants:**

- $After drop with probability p, rescale by 1/(1-p)$
- $Unbiased estimator of delta$

### eckart_young

```
||delta - delta_r||_F <= sigma_{r+1}
```

**Domain:** $delta_r = U_r @ diag(sigma_1..r) @ V_r^T, rank-r truncated SVD$

**Codomain:** $||error||_F \in [0, sigma_{r+1}]$

**Invariants:**

- $Error bounded by (r+1)-th singular value$
- $Rank-r approximation is optimal in Frobenius norm$

### lora_shape

$$
A \in \mathbb{R}^{m×r}, B \in \mathbb{R}^{r×n}, A @ B \in \mathbb{R}^{m×n}
$$

**Domain:** $r << min(m, n)$

**Invariants:**

- $A @ B has same shape as original weight$
- $Storage: r*(m+n) << m*n for small r$

### shape_preservation

```
shape(merged[t]) == shape(base[t]) for all tensors t
```

**Domain:** $Any merge strategy$

**Invariants:**

- $Merge never changes tensor shapes$

### task_vector

$$
delta = W_fine - W_base
$$

**Domain:** $W_fine, W_base \in \mathbb{R}^{m×n}$

**Codomain:** $delta \in \mathbb{R}^{m×n}$

**Invariants:**

- `Additive: W_base + delta == W_fine (roundtrip)`

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Task vector roundtrip | `base + (fine - base) == fine within ULP` |
| 2 | bound | Eckart-Young bound | `\|\|M - M_r\|\|_F <= sigma_{r+1}` |
| 3 | invariant | LoRA shape compatibility | $A=[m,r], B=[r,n] => A@B=[m,n]$ |
| 4 | invariant | DARE unbiasedness | $E[DARE(delta, p)] = delta$ |
| 5 | invariant | Shape preservation | `merged shape == base shape` |
| 6 | equivalence | SIMD LoRA equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-LA-001 | Task vector roundtrip | base + delta == fine_tune for random matrices | Floating-point non-associativity in large matrices |
| FALSIFY-LA-002 | Eckart-Young | Truncation error <= next singular value | SVD implementation error |
| FALSIFY-LA-003 | Shape compatibility | LoRA decomposition preserves output shape | Dimension mismatch in matmul |
| FALSIFY-LA-004 | DARE unbiased | Mean of 1000 DARE samples ≈ delta | Rescaling factor wrong |
| FALSIFY-LA-005 | Shape preservation | W + BA has same shape as W | LoRA rank dimensions incompatible |
| FALSIFY-LA-006 | SIMD LoRA equivalence | SIMD LoRA application matches scalar | SIMD LoRA accumulation order differs |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-LA-001 | LA-INV-001 | 4 | bounded_int |
| KANI-LORA_A-002 | Task vector roundtrip | 8 | exhaustive |
| KANI-LORA_A-003 | Eckart-Young bound | 8 | stub_float |
| KANI-LORA_A-004 | LoRA shape compatibility | 8 | exhaustive |
| KANI-LORA_A-005 | DARE unbiasedness | 8 | exhaustive |
| KANI-LORA_A-006 | Shape preservation | 8 | exhaustive |
| KANI-LORA_A-007 | SIMD LoRA equivalence | 8 | exhaustive |

## QA Gate

**LoRA Algebra Contract** (F-LA-001)

SVD extraction and merge strategy quality gate

**Checks:** task_vector_roundtrip, eckart_young_bound, shape_compatibility, dare_unbiased, shape_preservation

**Pass criteria:** All 6 falsification tests pass

