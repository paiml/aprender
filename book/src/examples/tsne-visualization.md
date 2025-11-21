# Case Study: t-SNE Implementation

This chapter documents the complete EXTREME TDD implementation of aprender's t-SNE algorithm for dimensionality reduction and visualization from Issue #18.

## Background

**GitHub Issue #18**: Implement t-SNE for Dimensionality Reduction and Visualization

**Requirements:**
- Non-linear dimensionality reduction (2D/3D)
- Perplexity-based similarity computation
- KL divergence minimization via gradient descent
- Parameters: n_components, perplexity, learning_rate, n_iter
- Reproducibility with random_state

**Initial State:**
- Tests: 640 passing
- Existing dimensionality reduction: PCA (linear)
- No non-linear dimensionality reduction

## Implementation Summary

### RED Phase

Created 12 comprehensive tests covering:
- Constructor and basic fitting
- Transform and fit_transform methods
- Perplexity parameter effects
- Learning rate and iteration count
- 2D and 3D embeddings
- Reproducibility with random_state
- Error handling (transform before fit)
- Local structure preservation
- Embedding finite values

### GREEN Phase

Implemented complete t-SNE algorithm (~400 lines):

**Core Components:**

1. **TSNE**: Public API with builder pattern
   - with_perplexity, with_learning_rate, with_n_iter, with_random_state
   - fit/transform/fit_transform methods

2. **Pairwise Distances** (`compute_pairwise_distances`):
   - Squared Euclidean distances in high-D
   - O(n²) computation

3. **Conditional Probabilities** (`compute_p_conditional`):
   - Binary search for sigma to match perplexity
   - Gaussian kernel: P(j|i) ∝ exp(-||x_i - x_j||² / (2σ_i²))
   - Target entropy: H = log₂(perplexity)

4. **Joint Probabilities** (`compute_p_joint`):
   - Symmetrize: P_{ij} = (P(j|i) + P(i|j)) / (2N)
   - Numerical stability with max(1e-12)

5. **Q Matrix** (`compute_q`):
   - Student's t-distribution in low-D
   - Q_{ij} ∝ (1 + ||y_i - y_j||²)^{-1}
   - Heavy-tailed distribution avoids crowding

6. **Gradient Computation** (`compute_gradient`):
   - ∇KL(P||Q) = 4Σ_j (p_ij - q_ij) · (y_i - y_j) / (1 + ||y_i - y_j||²)

7. **Optimization**:
   - Gradient descent with momentum (0.5 → 0.8)
   - Small random initialization (±0.00005)
   - Reproducible LCG random number generator

**Key Algorithm Steps:**

```rust
1. Compute pairwise distances in high-D
2. Binary search for sigma to match perplexity
3. Compute conditional probabilities P(j|i)
4. Symmetrize to joint probabilities P_{ij}
5. Initialize embedding randomly (small values)
6. For each iteration:
   a. Compute Q matrix (Student's t in low-D)
   b. Compute gradient of KL divergence
   c. Update embedding with momentum
7. Return final embedding
```

### REFACTOR Phase

- Fixed legacy numeric constants (f32::INFINITY)
- Zero clippy warnings
- Exported TSNE in prelude
- Comprehensive documentation
- Example demonstrating all key features

**Final State:**
- Tests: 640 → 652 (+12)
- Zero warnings
- All quality gates passing

## Algorithm Details

**Time Complexity:** O(n² · iterations)
- Dominated by pairwise distance computation each iteration
- Typical: 1000 iterations × O(n²) = impractical for n > 10,000

**Space Complexity:** O(n²)
- Distance matrix, P matrix, Q matrix all n×n

**Binary Search for Perplexity:**
- Target: H(P_i) = log₂(perplexity)
- Search for beta = 1/(2σ²) in range [0, ∞)
- 50 iterations max for convergence
- Tolerance: |H - target| < 1e-5

**Momentum Optimization:**
- Initial momentum: 0.5 (first 250 iterations)
- Final momentum: 0.8 (after iteration 250)
- Helps escape local minima and speed convergence

## Parameters

- **n_components** (default: 2): Output dimensions (usually 2 or 3)

- **perplexity** (default: 30.0): Balance local/global structure
  - Low (5-10): Very local, reveals fine clusters
  - Medium (20-30): Balanced (recommended)
  - High (50+): More global structure
  - Rule of thumb: perplexity < n_samples / 3

- **learning_rate** (default: 200.0): Gradient descent step size
  - Too low: Slow convergence
  - Too high: Unstable/divergence
  - Typical range: 10-1000

- **n_iter** (default: 1000): Number of gradient descent iterations
  - Minimum: 250 for reasonable results
  - Recommended: 1000 for convergence
  - More iterations: Better but slower

- **random_state** (default: None): Random seed for reproducibility

## Example Highlights

The example demonstrates:
1. Basic 4D → 2D reduction
2. Perplexity effects (2.0 vs 5.0)
3. 3D embedding
4. Learning rate effects (50.0 vs 500.0)
5. Reproducibility with random_state
6. t-SNE vs PCA comparison

## Key Takeaways

1. **Non-Linear**: Captures manifolds that PCA cannot
2. **Local Preservation**: Excellent at preserving neighborhoods
3. **Visualization**: Best for 2D/3D plots
4. **Perplexity Critical**: Try multiple values (5, 10, 30, 50)
5. **Stochastic**: Different runs give different embeddings
6. **Slow**: O(n²) limits scalability
7. **No Transform**: Cannot embed new data points

## Comparison: t-SNE vs PCA

| Feature | t-SNE | PCA |
|---------|-------|-----|
| Type | Non-linear | Linear |
| Preserves | Local structure | Global variance |
| Speed | O(n²·iter) | O(n·d·k) |
| Transform New Data | No | Yes |
| Deterministic | No (stochastic) | Yes |
| Best For | Visualization | Preprocessing |

**When to use t-SNE:**
- Visualizing high-dimensional data
- Exploratory data analysis
- Finding hidden clusters
- Presentations (2D plots)

**When to use PCA:**
- Feature reduction before modeling
- Large datasets (n > 10,000)
- Need to transform new data
- Need deterministic results

## Use Cases

1. **MNIST Visualization**: Visualize 784D digit images in 2D
2. **Word Embeddings**: Explore word2vec/GloVe embeddings
3. **Single-Cell RNA-seq**: Cluster cell types
4. **Image Features**: Visualize CNN features
5. **Customer Segmentation**: Explore behavioral clusters

## Testing Strategy

**Unit Tests** (12 implemented):
- Correctness: Embeddings have correct shape
- Reproducibility: Same random_state → same result
- Parameters: Perplexity, learning rate, n_iter effects
- Edge cases: Transform before fit, finite values

**Property-Based Tests** (future work):
- Local structure: Nearby points in high-D → nearby in low-D
- Perplexity monotonicity: Higher perplexity → smoother embedding
- Convergence: More iterations → lower KL divergence

## Technical Challenges Solved

### Challenge 1: Perplexity Matching
**Problem**: Finding sigma to match target perplexity.
**Solution**: Binary search on beta = 1/(2σ²) with entropy target.

### Challenge 2: Numerical Stability
**Problem**: Very small probabilities cause log(0) errors.
**Solution**: Clamp probabilities to max(p, 1e-12).

### Challenge 3: Reproducibility
**Problem**: std::random is non-deterministic.
**Solution**: Custom LCG random generator with seed.

### Challenge 4: Large Embedding Values
**Problem**: Embeddings can have very large absolute values.
**Solution**: This is expected - t-SNE preserves relative distances, not absolute positions.

## Related Topics

- [PCA Implementation](./pca-iris.md)
- [K-Means Clustering](./kmeans-clustering.md)
- [Spectral Clustering](./spectral-clustering.md)
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)

## References

1. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. JMLR.
2. Wattenberg, et al. (2016). How to Use t-SNE Effectively. Distill.
3. Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. Nature Communications.
