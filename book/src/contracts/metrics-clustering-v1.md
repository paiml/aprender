# metrics-clustering-v1

**Version:** 1.0.0

Clustering metrics — evaluation measures for unsupervised cluster quality

## References

- Rousseeuw (1987) Silhouettes: a graphical aid to interpretation of cluster analysis
- Hubert & Arabie (1985) Comparing Partitions

## Equations

### inertia

$$
J = \sumᵢ ||xᵢ - μ_{cᵢ}||²
$$

**Domain:** $x \in \mathbb{R}^{n×d}, centroids \in \mathbb{R}^{K×d}, labels \in {0..K-1}ⁿ$

**Codomain:** $J \in [0, ∞)$

**Invariants:**

- $J \geq 0 (sum of squared distances)$
- $J = 0 iff all points equal their centroids$
- $J is non-increasing under k-means iteration$

### silhouette_coefficient

$$
s(i) = (b(i) - a(i)) / max(a(i), b(i)) where a(i) = mean intra-cluster dist, b(i) = min mean inter-cluster dist
$$

**Domain:** $a(i) \geq 0, b(i) \geq 0$

**Codomain:** $s(i) \in [-1, 1]$

**Invariants:**

- $s(i) \in [-1, 1]$
- $s(i) = 0 when a(i) = b(i)$
- $s(i) \to 1 when a(i) \to 0 and b(i) > 0$

### silhouette_score

$$
s(i) = (b(i) - a(i)) / max(a(i), b(i))
$$

**Domain:** $data \in \mathbb{R}^{n×d}, labels \in {0..K-1}ⁿ, K \geq 2, n \geq 2$

**Codomain:** $s̄ \in [-1, 1]$

**Invariants:**

- $s̄ \in [-1, 1] (bounded by construction)$
- $s̄ = 0 when single cluster (degenerate)$
- $s(i) > 0 means point i is well-clustered$
- $s(i) < 0 means point i is mis-clustered$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Silhouette score bounded | $s̄ \in [-1, 1] for all valid clusterings$ |
| 2 | bound | Inertia non-negative | $J \geq 0 for all data and assignments$ |
| 3 | invariant | Silhouette degenerate case | $s̄ = 0 when K < 2$ |
| 4 | invariant | Inertia zero at centroids | $J = 0 when xᵢ = μ_{cᵢ} for all i$ |
| 5 | bound | Per-point silhouette bounded | $s(i) \in [-1, 1] for each point$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CL-001 | Silhouette bounded | silhouette_score ∈ [-1, 1] for random clusterings | Division by zero when max(a,b) = 0 |
| FALSIFY-CL-002 | Inertia non-negative | inertia ≥ 0 for all data and centroids | Sign error in squared distance |
| FALSIFY-CL-003 | Silhouette degenerate | silhouette_score = 0 when all same label or single sample | Degenerate case not handled |
| FALSIFY-CL-004 | Well-separated clusters | silhouette > 0.5 for well-separated clusters (gap >> cluster radius) | Distance computation incorrect |
| FALSIFY-CL-005 | Inertia zero at centroids | inertia = 0 when each point equals its assigned centroid | Residual floating-point error or centroid assignment mismatch |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CL-001 | 1 | 4 | stub_float |
| KANI-CL-002 | 2 | 4 | stub_float |
| KANI-METRIC-003 | Silhouette score bounded | 8 | stub_float |
| KANI-METRIC-004 | Inertia non-negative | 8 | exhaustive |
| KANI-METRIC-005 | Silhouette degenerate case | 8 | exhaustive |
| KANI-METRIC-006 | Inertia zero at centroids | 8 | exhaustive |
| KANI-METRIC-007 | Per-point silhouette bounded | 8 | stub_float |

## QA Gate

**Clustering Metrics Contract** (F-CL-001)

Clustering metric correctness quality gate

**Checks:** silhouette_bounded, inertia_non_negative, silhouette_degenerate, well_separated, inertia_zero_at_centroids

**Pass criteria:** All 5 falsification tests pass

