# pca-v1

**Version:** 1.0.0

Principal Component Analysis — eigendecomposition-based dimensionality reduction

## References

- Jolliffe (2002) Principal Component Analysis
- Bishop (2006) Pattern Recognition and Machine Learning, §12.1

## Equations

### explained_variance

```
explained_ratio_j = λ_j / Σ λ_i
```

**Domain:** $eigenvalues \lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d \geq 0$

**Codomain:** $ratio_j \in [0, 1], \sum ratio_j = 1$

**Invariants:**

- $Each ratio \in [0, 1]$
- $Ratios sum to 1$
- $Ratios are non-increasing (\lambda sorted descending)$
- $All eigenvalues \geq 0 (covariance matrix is PSD)$

### pca_transform

$$
Z = (X - μ) W_k where W_k = [w_1, ..., w_k] (top-k eigenvectors of Cov(X))
$$

**Domain:** $X \in \mathbb{R}^{n×d}, k \leq d$

**Codomain:** $Z \in \mathbb{R}^{n×k}$

**Invariants:**

- $Output has k columns (dimensionality reduction)$
- $Components are orthogonal: Z^T Z is diagonal$
- $First component captures maximum variance$

### reconstruction

$$
X̂ = Z W_k^T + μ (approximate reconstruction)
$$

**Domain:** $Z \in \mathbb{R}^{n×k}, W_k \in \mathbb{R}^{d×k}, μ \in \mathbb{R}^d$

**Codomain:** $X̂ \in \mathbb{R}^{n×d}$

**Invariants:**

- $||X - X̂|| decreases as k increases$
- $k = d ⟹ X̂ = X (perfect reconstruction)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Dimensionality reduction | $PCA(X, k).shape = (n, k)$ |
| 2 | bound | Explained variance bounded | $Each explained_ratio \in [0, 1]$ |
| 3 | invariant | Explained variance sums to 1 | $\sum explained_ratio = 1 (for all d components)$ |
| 4 | invariant | Perfect reconstruction at full rank | $k = d ⟹ \|\|X - reconstruct(PCA(X, d))\|\| < \varepsilon$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-PCA-001 | Dimensionality reduction | transformed shape = (n_samples, n_components) | Component selection or projection error |
| FALSIFY-PCA-002 | Explained variance bounded | each ratio ∈ [0, 1] and sum ≈ 1 | Eigenvalue normalization error |
| FALSIFY-PCA-003 | Variance ordering | explained_variance_ratio is non-increasing | Eigenvalues not sorted descending |
| FALSIFY-PCA-004 | Deterministic transform | transform(X) = transform(X) for same X | Non-deterministic random state |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-PCA-001 | PCA-INV-001 | 8 | stub_float |
| KANI-PCA-002 | PCA-BND-001 | 8 | stub_float |
| KANI-PCA_V1-003 | Dimensionality reduction | 8 | exhaustive |
| KANI-PCA_V1-004 | Explained variance bounded | 8 | stub_float |
| KANI-PCA_V1-005 | Explained variance sums to 1 | 8 | stub_float |
| KANI-PCA_V1-006 | Perfect reconstruction at full rank | 8 | exhaustive |

## QA Gate

**PCA Contract** (F-PCA-001)

PCA correctness quality gate

**Checks:** dim_reduction, variance_bounded, variance_ordering, deterministic

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

