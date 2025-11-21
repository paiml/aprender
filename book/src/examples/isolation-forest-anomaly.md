# Case Study: Isolation Forest Implementation

This chapter documents the complete EXTREME TDD implementation of aprender's Isolation Forest algorithm for anomaly detection from Issue #17.

## Background

**GitHub Issue #17**: Implement Isolation Forest for Anomaly Detection

**Requirements:**
- Ensemble of isolation trees using random partitioning
- O(n log n) training complexity
- Parameters: n_estimators, max_samples, contamination
- Methods: fit(), predict(), score_samples()
- predict() returns 1 for normal, -1 for anomaly
- score_samples() returns anomaly scores (lower = more anomalous)
- Use cases: fraud detection, network intrusion, quality control

**Initial State:**
- Tests: 596 passing
- Existing clustering: K-Means, DBSCAN, Hierarchical, GMM
- No anomaly detection support

## Implementation Summary

### RED Phase

Created 17 comprehensive tests covering:
- Constructor and basic fitting
- Anomaly prediction (1=normal, -1=anomaly)
- Anomaly score computation
- Contamination parameter (10%, 20%, 30%)
- Number of trees (ensemble size)
- Max samples (subsample size)
- Reproducibility with random seeds
- Multidimensional data (3+ features)
- Path length calculations
- Decision function consistency
- Error handling (predict/score before fit)
- Edge cases (all normal points)

### GREEN Phase

Implemented complete Isolation Forest (387 lines):

**Core Components:**

1. **IsolationNode**: Binary tree node structure
   - Split feature and value
   - Left/right children (Box for recursion)
   - Node size (for path length calculation)

2. **IsolationTree**: Single isolation tree
   - `build_tree()`: Recursive random partitioning
   - `path_length()`: Compute isolation path length
   - `c(n)`: Average BST path length for normalization

3. **IsolationForest**: Public API
   - Ensemble of isolation trees
   - Builder pattern (with_* methods)
   - fit(): Train ensemble on subsamples
   - predict(): Binary classification (1/-1)
   - score_samples(): Anomaly scores

**Key Algorithm Steps:**

1. **Training (fit)**:
   - For each of N trees:
     - Sample random subset (max_samples)
     - Build tree via random splits
     - Store tree in ensemble

2. **Tree Building (build_tree)**:
   - Terminal: depth >= max_depth OR n_samples <= 1
   - Pick random feature
   - Pick random split value between min/max
   - Recursively build left/right subtrees

3. **Scoring (score_samples)**:
   - For each sample:
     - Compute path length in each tree
     - Average across ensemble
     - Normalize: 2^(-avg_path / c_norm)
     - Invert (lower = more anomalous)

4. **Classification (predict)**:
   - Compute anomaly scores
   - Compare to threshold (from contamination)
   - Return 1 (normal) or -1 (anomaly)

**Numerical Considerations:**

- Random subsampling for efficiency
- Path length normalization via c(n) function
- Threshold computed from training data quantile
- Default max_samples: min(256, n_samples)

### REFACTOR Phase

- Removed unused imports
- Zero clippy warnings
- Exported in prelude for easy access
- Comprehensive documentation with examples
- Added fraud detection example scenario

**Final State:**
- Tests: 613 passing (596 → 613, +17)
- Zero warnings
- All quality gates passing

## Algorithm Details

**Isolation Forest:**
- Ensemble method for anomaly detection
- Intuition: Anomalies are easier to isolate than normal points
- Shorter path length → More anomalous

**Time Complexity:** O(n log m)
- n = samples, m = max_samples

**Space Complexity:** O(t * m * d)
- t = n_estimators, m = max_samples, d = features

**Average Path Length (c function):**
```text
c(n) = 2H(n-1) - 2(n-1)/n
where H(n) is harmonic number ≈ ln(n) + 0.5772
```

This normalizes path lengths by expected BST depth.

## Parameters

- **n_estimators** (default: 100): Number of trees in ensemble
  - More trees = more stable predictions
  - Diminishing returns after ~100 trees

- **max_samples** (default: min(256, n)): Subsample size per tree
  - Smaller = faster training
  - 256 is empirically good default
  - Full sample rarely needed

- **contamination** (default: 0.1): Expected anomaly proportion
  - Range: 0.0 to 0.5
  - Sets classification threshold
  - 0.1 = 10% anomalies expected

- **random_state** (optional): Seed for reproducibility

## Example Highlights

The example demonstrates:
1. Basic anomaly detection (8 normal + 2 outliers)
2. Anomaly score interpretation
3. Contamination parameter effects (10%, 20%, 30%)
4. Ensemble size comparison (10 vs 100 trees)
5. Credit card fraud detection scenario
6. Reproducibility with random seeds
7. Isolation path length concept
8. Max samples parameter

## Key Takeaways

1. **Unsupervised Anomaly Detection**: No labeled data required
2. **Fast Training**: O(n log m) makes it scalable
3. **Interpretable Scores**: Path length has clear meaning
4. **Few Parameters**: Easy to use with sensible defaults
5. **No Distance Metric**: Works with any feature types
6. **Handles High Dimensions**: Better than density-based methods
7. **Ensemble Benefits**: Averaging reduces variance

## Comparison with Other Methods

**vs K-Means:**
- K-Means: Finds clusters, requires distance threshold for anomalies
- Isolation Forest: Directly detects anomalies, no threshold needed

**vs DBSCAN:**
- DBSCAN: Density-based, requires eps/min_samples tuning
- Isolation Forest: Contamination parameter is intuitive

**vs GMM:**
- GMM: Probabilistic, assumes Gaussian distributions
- Isolation Forest: No distributional assumptions

**vs One-Class SVM:**
- SVM: O(n²) to O(n³) training time
- Isolation Forest: O(n log m) - much faster

## Use Cases

1. **Fraud Detection**: Credit card transactions, insurance claims
2. **Network Security**: Intrusion detection, anomalous traffic
3. **Quality Control**: Manufacturing defects, sensor anomalies
4. **System Monitoring**: Server metrics, application logs
5. **Healthcare**: Rare disease detection, unusual patient profiles

## Testing Strategy

**Property-Based Tests** (future work):
- Score ranges: All scores should be finite
- Contamination consistency: Higher contamination → more anomalies
- Reproducibility: Same seed → same results
- Path length bounds: 0 ≤ path ≤ log2(max_samples)

**Unit Tests** (17 implemented):
- Correctness: Detects clear outliers
- API contracts: Panic before fit, return expected types
- Parameters: All builder methods work
- Edge cases: All normal, all anomalous, small datasets

## Related Topics

- [K-Means Clustering](./kmeans-clustering.md)
- [DBSCAN Clustering](./dbscan-clustering.md)
- [GMM Clustering](./gmm-clustering.md)
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
- [Property-Based Testing](../advanced-testing/property-based-testing.md)
