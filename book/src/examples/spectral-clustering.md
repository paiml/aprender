# Case Study: Spectral Clustering Implementation

This chapter documents the complete EXTREME TDD implementation of aprender's Spectral Clustering algorithm for graph-based clustering from Issue #19.

## Background

**GitHub Issue #19**: Implement Spectral Clustering for Non-Convex Clustering

**Requirements:**
- Graph-based clustering using eigendecomposition
- Affinity matrix construction (RBF and k-NN)
- Normalized graph Laplacian
- Eigendecomposition for embedding
- K-Means clustering in eigenspace
- Parameters: n_clusters, affinity, gamma, n_neighbors

**Initial State:**
- Tests: 628 passing
- Existing clustering: K-Means, DBSCAN, Hierarchical, GMM
- No graph-based clustering

## Implementation Summary

### RED Phase

Created 12 comprehensive tests covering:
- Constructor and basic fitting
- Predict method and labels consistency
- Non-convex cluster shapes (moon-shaped clusters)
- RBF affinity matrix
- K-NN affinity matrix
- Gamma parameter effects
- Multiple clusters (3 clusters)
- Error handling (predict before fit)

### GREEN Phase

Implemented complete Spectral Clustering algorithm (352 lines):

**Core Components:**

1. **Affinity Enum**: RBF (Gaussian kernel) and KNN (k-nearest neighbors graph)

2. **SpectralClustering**: Public API with builder pattern
   - with_affinity, with_gamma, with_n_neighbors
   - fit/predict/is_fitted methods

3. **Affinity Matrix Construction**:
   - RBF: `W[i,j] = exp(-gamma * ||x_i - x_j||^2)`
   - K-NN: Connect each point to k nearest neighbors, symmetrize

4. **Graph Laplacian**: Normalized Laplacian `L = I - D^(-1/2) * W * D^(-1/2)`
   - D is degree matrix (diagonal)
   - Provides better numerical properties than unnormalized Laplacian

5. **Eigendecomposition** (`compute_embedding`):
   - Extract k smallest eigenvectors using nalgebra
   - Sort eigenvalues to find smallest k
   - Build embedding matrix in row-major order

6. **Row Normalization**: Critical for normalized spectral clustering
   - Normalize each row of embedding to unit length
   - Improves cluster separation in eigenspace

7. **K-Means Clustering**: Final clustering in eigenspace

**Key Algorithm Steps:**

```rust
1. Construct affinity matrix W (RBF or k-NN)
2. Compute degree matrix D
3. Compute normalized Laplacian L = I - D^(-1/2) * W * D^(-1/2)
4. Find k smallest eigenvectors of L
5. Normalize rows of eigenvector matrix
6. Apply K-Means clustering in eigenspace
```

### REFACTOR Phase

- Fixed unnecessary type cast warning
- Zero clippy warnings
- Exported Affinity and SpectralClustering in prelude
- Comprehensive documentation
- Example demonstrating RBF vs K-NN affinity

**Final State:**
- Tests: 628 → 640 (+12)
- Zero warnings
- All quality gates passing

## Algorithm Details

**Spectral Clustering:**
- Uses graph theory to find clusters
- Analyzes spectrum (eigenvalues) of graph Laplacian
- Effective for non-convex cluster shapes
- Based on graph cut optimization

**Time Complexity:** O(n² + n³)
- O(n²) for affinity matrix construction
- O(n³) for eigendecomposition
- Dominated by eigendecomposition

**Space Complexity:** O(n²)
- Affinity matrix storage
- Laplacian matrix storage

**RBF Affinity:**
```
W[i,j] = exp(-gamma * ||x_i - x_j||^2)
```
- Gamma controls locality (higher = more local)
- Full connectivity (dense graph)
- Good for globular clusters

**K-NN Affinity:**
```
W[i,j] = 1 if j in k-NN(i), 0 otherwise
Symmetrize: W[i,j] = max(W[i,j], W[j,i])
```
- Sparse connectivity
- Better for non-convex shapes
- Parameter k controls graph density

**Normalized Graph Laplacian:**
```
L = I - D^(-1/2) * W * D^(-1/2)
```
Where D is the degree matrix (diagonal, D[i,i] = sum of row i of W).

## Parameters

- **n_clusters** (required): Number of clusters to find

- **affinity** (default: RBF): Affinity matrix type
  - RBF: Gaussian kernel, good for globular clusters
  - KNN: k-nearest neighbors, good for non-convex shapes

- **gamma** (default: 1.0): RBF kernel coefficient
  - Higher gamma: More local similarity
  - Lower gamma: More global similarity
  - Only used for RBF affinity

- **n_neighbors** (default: 10): Number of neighbors for k-NN graph
  - Smaller k: Sparser graph, more clusters
  - Larger k: Denser graph, fewer clusters
  - Only used for KNN affinity

## Example Highlights

The example demonstrates:
1. Basic RBF affinity clustering
2. K-NN affinity for chain-like clusters
3. Gamma parameter effects (0.1, 1.0, 5.0)
4. Multiple clusters (k=3)
5. Spectral Clustering vs K-Means comparison
6. Affinity matrix interpretation

## Key Takeaways

1. **Graph-Based**: Uses graph theory and eigendecomposition
2. **Non-Convex**: Handles non-convex cluster shapes better than K-Means
3. **Affinity Choice**: RBF for globular, K-NN for non-convex
4. **Row Normalization**: Critical step after eigendecomposition
5. **Eigenvalue Sorting**: Must sort eigenvalues to find smallest k
6. **Computational Cost**: O(n³) eigendecomposition limits scalability

## Comparison: Spectral vs K-Means

| Feature | Spectral Clustering | K-Means |
|---------|---------------------|---------|
| Cluster Shape | Non-convex, arbitrary | Convex, spherical |
| Complexity | O(n³) | O(nki) |
| Scalability | Small to medium | Large datasets |
| Parameters | n_clusters, affinity, gamma/k | n_clusters, max_iter |
| Graph Structure | Yes (via affinity) | No |
| Initialization | Deterministic (eigenvectors) | Random (k-means++) |

**When to use Spectral Clustering:**
- Data has non-convex cluster shapes
- Clusters have varying densities
- Data has graph structure
- Dataset is small-to-medium sized

**When to use K-Means:**
- Clusters are roughly spherical
- Dataset is large (millions of points)
- Speed is critical
- Cluster sizes are similar

## Use Cases

1. **Image Segmentation**: Segment images by pixel similarity
2. **Social Network Analysis**: Find communities in social graphs
3. **Document Clustering**: Group documents by content similarity
4. **Gene Expression Analysis**: Cluster genes with similar expression patterns
5. **Anomaly Detection**: Identify outliers via cluster membership

## Testing Strategy

**Unit Tests** (12 implemented):
- Correctness: Separates well-separated clusters
- API contracts: Panic before fit, return expected types
- Parameters: affinity, gamma, n_neighbors effects
- Edge cases: Multiple clusters, non-convex shapes

**Property-Based Tests** (future work):
- Connected components: k eigenvalues near 0 → k clusters
- Affinity symmetry: W[i,j] = W[j,i]
- Laplacian positive semi-definite

## Technical Challenges Solved

### Challenge 1: Eigenvalue Ordering
**Problem**: nalgebra's SymmetricEigen doesn't sort eigenvalues.
**Solution**: Manual sorting of eigenvalue-index pairs, take k smallest indices.

### Challenge 2: Row-Major vs Column-Major
**Problem**: Embedding matrix constructed in column-major order but Matrix expects row-major.
**Solution**: Iterate rows first, then columns when extracting eigenvectors.

### Challenge 3: Row Normalization
**Problem**: Without row normalization, clustering quality was poor.
**Solution**: Normalize each row of embedding matrix to unit length.

### Challenge 4: Concentric Circles
**Problem**: Original test used concentric circles, fundamentally challenging for spectral clustering.
**Solution**: Replaced with more realistic moon-shaped clusters.

## Related Topics

- [K-Means Clustering](./kmeans-clustering.md)
- [DBSCAN Clustering](./dbscan-clustering.md)
- [Hierarchical Clustering](./hierarchical-clustering.md)
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)

## References

1. Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). On spectral clustering: Analysis and an algorithm. NIPS.
2. Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17(4), 395-416.
3. Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. IEEE TPAMI.
