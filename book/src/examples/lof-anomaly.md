# Case Study: Local Outlier Factor (LOF) Implementation

This chapter documents the complete EXTREME TDD implementation of aprender's Local Outlier Factor algorithm for density-based anomaly detection from Issue #20.

## Background

**GitHub Issue #20**: Implement Local Outlier Factor (LOF) for Anomaly Detection

**Requirements:**
- Density-based anomaly detection using local reachability density
- Detects outliers in varying density regions
- Parameters: n_neighbors, contamination
- Methods: fit(), predict(), score_samples(), negative_outlier_factor()
- LOF score interpretation: ≈1 = normal, >>1 = outlier

**Initial State:**
- Tests: 612 passing
- Existing anomaly detection: Isolation Forest
- No density-based anomaly detection

## Implementation Summary

### RED Phase

Created 16 comprehensive tests covering:
- Constructor and basic fitting
- LOF score calculation (higher = more anomalous)
- Anomaly prediction (1=normal, -1=anomaly)
- Contamination parameter (10%, 20%, 30%)
- n_neighbors parameter (local vs global context)
- Varying density clusters (key LOF advantage)
- negative_outlier_factor() for sklearn compatibility
- Error handling (predict/score before fit)
- Multidimensional data
- Edge cases (all normal points)

### GREEN Phase

Implemented complete LOF algorithm (352 lines):

**Core Components:**

1. **LocalOutlierFactor**: Public API
   - Builder pattern (with_n_neighbors, with_contamination)
   - fit/predict/score_samples methods
   - negative_outlier_factor for sklearn compatibility

2. **k-NN Search** (`compute_knn`):
   - Brute-force distance computation
   - Sort by distance
   - Extract k nearest neighbors

3. **Reachability Distance** (`reachability_distance`):
   - max(distance(A,B), k-distance(B))
   - Smooths density estimation

4. **Local Reachability Density** (`compute_lrd`):
   - LRD(A) = k / Σ(reachability_distance(A, neighbor))
   - Inverse of average reachability distance

5. **LOF Score** (`compute_lof_scores`):
   - LOF(A) = avg(LRD(neighbors)) / LRD(A)
   - Ratio of neighbor density to point density

**Key Algorithm Steps:**

1. **Fit**:
   - Compute k-NN for all training points
   - Compute LRD for all points
   - Compute LOF scores
   - Determine threshold from contamination

2. **Predict**:
   - Compute k-NN for query points against training
   - Compute LRD for query points
   - Compute LOF scores for query points
   - Apply threshold: LOF > threshold → anomaly

### REFACTOR Phase

- Removed unused variables
- Zero clippy warnings
- Exported in prelude
- Comprehensive documentation
- Varying density example showcasing LOF's key advantage

**Final State:**
- Tests: 612 → 628 (+16)
- Zero warnings
- All quality gates passing

## Algorithm Details

**Local Outlier Factor:**
- Compares local density to neighbors' densities
- Key advantage: Works with varying density regions
- LOF score interpretation:
  - LOF ≈ 1: Similar density (normal)
  - LOF >> 1: Lower density (outlier)
  - LOF < 1: Higher density (core point)

**Time Complexity:** O(n² log k)
- n = samples, k = n_neighbors
- Dominated by k-NN search

**Space Complexity:** O(n²)
- Distance matrix and k-NN storage

**Reachability Distance:**
```text
reach_dist(A, B) = max(dist(A, B), k_dist(B))
```
Where k_dist(B) is distance to B's k-th neighbor.

**Local Reachability Density:**
```text
LRD(A) = k / Σ_i reach_dist(A, neighbor_i)
```

**LOF Score:**
```text
LOF(A) = (Σ_i LRD(neighbor_i)) / (k * LRD(A))
```

## Parameters

- **n_neighbors** (default: 20): Number of neighbors for density estimation
  - Smaller k: More local, sensitive to local outliers
  - Larger k: More global context, stable but may miss local anomalies

- **contamination** (default: 0.1): Expected anomaly proportion
  - Range: 0.0 to 0.5
  - Sets classification threshold

## Example Highlights

The example demonstrates:
1. Basic anomaly detection
2. LOF score interpretation (≈1 vs >>1)
3. Varying density clusters (LOF's key advantage)
4. n_neighbors parameter effects
5. Contamination parameter
6. LOF vs Isolation Forest comparison
7. negative_outlier_factor for sklearn compatibility
8. Reproducibility

## Key Takeaways

1. **Density-Based**: LOF compares local densities, not global isolation
2. **Varying Density**: Excels where clusters have different densities
3. **Interpretable Scores**: LOF score has clear meaning
4. **Local Context**: n_neighbors controls locality
5. **Complementary**: Works well alongside Isolation Forest
6. **No Distance Metric Bias**: Uses relative densities

## Comparison: LOF vs Isolation Forest

| Feature | LOF | Isolation Forest |
|---------|-----|------------------|
| Approach | Density-based | Isolation-based |
| Varying Density | Excellent | Good |
| Global Outliers | Good | Excellent |
| Training Time | O(n²) | O(n log m) |
| Parameter Tuning | n_neighbors | n_estimators, max_samples |
| Interpretability | High (density ratio) | Medium (path length) |

**When to use LOF:**
- Data has regions with different densities
- Need to detect local outliers
- Want interpretable density-based scores

**When to use Isolation Forest:**
- Large datasets (faster training)
- Global outliers more important
- Don't know density structure

**Best practice:** Use both and ensemble the results!

## Use Cases

1. **Fraud Detection**: Transactions with unusual patterns relative to user's history
2. **Network Security**: Anomalous traffic in varying load conditions
3. **Manufacturing**: Defects in varying production speeds
4. **Sensor Networks**: Faulty sensors in varying environmental conditions
5. **Medical Diagnosis**: Unusual patient metrics relative to demographic group

## Testing Strategy

**Unit Tests** (16 implemented):
- Correctness: Detects clear outliers in varying densities
- API contracts: Panic before fit, return expected types
- Parameters: n_neighbors, contamination effects
- Edge cases: All normal, all anomalous, small k

**Property-Based Tests** (future work):
- LOF ≈ 1 for uniform density
- LOF monotonic in isolation degree
- Consistency: Same k → consistent relative ordering

## Related Topics

- [Isolation Forest](./isolation-forest-anomaly.md)
- [K-Means Clustering](./kmeans-clustering.md)
- [DBSCAN Clustering](./dbscan-clustering.md)
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
