# kmeans-kernel-v1

**Version:** 1.0.0

K-Means kernel — Lloyd's algorithm for cluster assignment

## References

- Lloyd (1982) Least Squares Quantization in PCM
- Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding

## Equations

### assignment

$$
c_i = argmin_j ||x_i - mu_j||^2
$$

**Domain:** $x in R^{N x d}, mu in R^{K x d}, K >= 1$

**Codomain:** $c in {0, ..., K-1}^N$

**Invariants:**

- $Each point assigned to nearest centroid$
- $Every cluster index in [0, K-1]$

### objective

```
J = sum_{i=1}^{N} ||x_i - mu_{c_i}||^2
```

**Domain:** $x in R^{N x d}, mu in R^{K x d}, c in {0..K-1}^N$

**Codomain:** $J in R_>=0$

**Invariants:**

- $J >= 0 (non-negative)$
- $J is non-increasing across iterations (monotone convergence)$

### update

$$
mu_j = (1/|S_j|) * sum_{i in S_j} x_i
$$

**Domain:** $x in R^{N x d}, S_j = {i : c_i = j}$

**Codomain:** $mu in R^{K x d}$

**Invariants:**

- $New centroid is mean of assigned points$
- $Empty cluster centroid unchanged$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Nearest centroid assignment | `\|\|x_i - mu_{c_i}\|\| <= \|\|x_i - mu_j\|\| for all j` |
| 2 | monotonicity | Objective non-increasing | $J_{t+1} <= J_t after each assignment+update step$ |
| 3 | bound | Objective non-negative | $J >= 0$ |
| 4 | invariant | Valid cluster indices | $c_i in {0, ..., K-1} for all i$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **initialize**: Initialize K centroids (random or k-means++) — *K distinct centroids*
2. **assign**: Assign each point to nearest centroid — *All assignments in [0, K-1]*
3. **update**: Recompute centroids as cluster means — *Centroid is mean of assigned points*
4. **check_convergence**: Check if assignments changed — *Terminates in finite iterations*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| kmeans_assign | avx2 | `kmeans_assign_avx2` |
| kmeans_assign | ptx | `kmeans_ptx` |
| kmeans_assign | scalar | `kmeans_assign_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-KM-001 | Nearest assignment | dist(x_i, mu_{c_i}) <= dist(x_i, mu_j) for all j | Distance comparison or argmin implementation incorrect |
| FALSIFY-KM-002 | Monotone convergence | J_{t+1} <= J_t for 100 random iterations | Centroid update not correctly computing mean |
| FALSIFY-KM-003 | Objective non-negativity | J >= 0 for any assignment | Squared distance computation has sign error |
| FALSIFY-KM-004 | Valid indices | All c_i in [0, K-1] | Out-of-bounds cluster index |
| FALSIFY-KM-005 | SIMD equivalence | \|assign_avx2(x) - assign_scalar(x)\| = 0 (exact match for argmin) | SIMD distance computation differs |
| FALSIFY-KM-006 | Boundary - K=1 | All points assigned to cluster 0, centroid = mean(x) | Single-cluster degenerate case not handled |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-KM-001 | KM-INV-001 | 4 | stub_float |
| KANI-KM-002 | KM-BND-001 | 4 | stub_float |
| KANI-KMEANS-003 | Nearest centroid assignment | 8 | exhaustive |
| KANI-KMEANS-004 | Objective non-increasing | 8 | exhaustive |
| KANI-KMEANS-005 | Objective non-negative | 8 | exhaustive |
| KANI-KMEANS-006 | Valid cluster indices | 8 | exhaustive |
| KANI-KMEANS-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**K-Means Contract** (F-KM-001)

Lloyd's algorithm cluster assignment quality gate

**Checks:** nearest_assignment, monotone_convergence, objective_bound, simd_equivalence

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

