# ica-v1

**Version:** 1.0.0

Independent Component Analysis — FastICA blind source separation

## References

- Hyvarinen & Oja (2000) Independent Component Analysis: Algorithms and Applications
- Hyvarinen (1999) Fast and Robust Fixed-Point Algorithms for ICA

## Equations

### fastica

$$
W = argmax_{W orthogonal} \sum_i |E[G(w_i^T z)]|² where z = whitened(X)
$$

**Domain:** $X \in \mathbb{R}^{n×d}, n_components \leq d$

**Codomain:** $S = X W^T \in \mathbb{R}^{n×k} (independent components)$

**Invariants:**

- $Output has n_components columns$
- $W is orthogonal: W W^T \approx I$
- $Components are maximally non-Gaussian$

### mixing

$$
X̂ = S A where A = W^{-1} (mixing matrix)
$$

**Domain:** $S \in \mathbb{R}^{n×k}, A \in \mathbb{R}^{k×d}$

**Codomain:** $X̂ \in \mathbb{R}^{n×d}$

**Invariants:**

- $Approximate reconstruction: X̂ \approx X when k = d$
- $A W \approx I (mixing · unmixing = identity)$

### unmixing

$$
S = X W^T where W is the unmixing matrix
$$

**Domain:** $X \in \mathbb{R}^{n×d}, W \in \mathbb{R}^{k×d}$

**Codomain:** $S \in \mathbb{R}^{n×k}$

**Invariants:**

- $Unmixing is linear$
- $Output shape = (n_samples, n_components)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Output shape | $ICA(X, k).shape = (n, k)$ |
| 2 | invariant | Deterministic output | $transform(X) = transform(X) for fixed model$ |
| 3 | invariant | Component count | $Number of output components = n_components$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-ICA-001 | Output shape | transformed shape = (n_samples, n_components) | Component extraction or projection error |
| FALSIFY-ICA-002 | Transform deterministic | transform(X) = transform(X) for fitted model | Random state leaking between calls |
| FALSIFY-ICA-003 | Finite output | All output values are finite for finite input | Division by zero in whitening or convergence failure |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-ICA-001 | ICA-INV-001 | 8 | stub_float |
| KANI-ICA_V1-002 | Output shape | 8 | exhaustive |
| KANI-ICA_V1-003 | Deterministic output | 8 | exhaustive |
| KANI-ICA_V1-004 | Component count | 8 | exhaustive |

## QA Gate

**ICA Contract** (F-ICA-001)

ICA correctness quality gate

**Checks:** output_shape, deterministic, finite_output

**Pass criteria:** All 3 falsification tests pass + Kani harnesses verify

