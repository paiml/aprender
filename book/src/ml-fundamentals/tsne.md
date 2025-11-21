# t-SNE Theory

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear dimensionality reduction technique optimized for visualizing high-dimensional data in 2D or 3D space.

## Core Idea

t-SNE preserves local structure by:
1. Computing pairwise similarities in high-dimensional space (Gaussian kernel)
2. Computing pairwise similarities in low-dimensional space (Student's t-distribution)
3. Minimizing the KL divergence between these two distributions

## Algorithm

### Step 1: High-Dimensional Similarities

Compute conditional probabilities using Gaussian kernel:

```
P(j|i) = exp(-||x_i - x_j||² / (2σ_i²)) / Σ_k exp(-||x_i - x_k||² / (2σ_i²))
```

Where σ_i is chosen such that the perplexity equals a target value.

**Perplexity** controls the effective number of neighbors:
```
Perplexity(P_i) = 2^H(P_i)
where H(P_i) = -Σ_j P(j|i) log₂ P(j|i)
```

Typical range: 5-50 (default: 30)

### Step 2: Symmetric Joint Probabilities

Make probabilities symmetric:
```
P_{ij} = (P(j|i) + P(i|j)) / (2N)
```

### Step 3: Low-Dimensional Similarities

Use Student's t-distribution (heavy-tailed) to avoid "crowding problem":
```
Q_{ij} = (1 + ||y_i - y_j||²)^{-1} / Σ_{k≠l} (1 + ||y_k - y_l||²)^{-1}
```

### Step 4: Minimize KL Divergence

Minimize Kullback-Leibler divergence:
```
KL(P||Q) = Σ_i Σ_j P_{ij} log(P_{ij} / Q_{ij})
```

Using gradient descent with momentum:
```
∂KL/∂y_i = 4 Σ_j (P_{ij} - Q_{ij}) · (y_i - y_j) · (1 + ||y_i - y_j||²)^{-1}
```

## Parameters

- **n_components** (default: 2): Embedding dimensions (usually 2 or 3 for visualization)
- **perplexity** (default: 30.0): Balance between local and global structure
  - Low (5-10): Very local, reveals fine clusters
  - Medium (20-30): Balanced
  - High (50+): More global structure
- **learning_rate** (default: 200.0): Gradient descent step size
- **n_iter** (default: 1000): Number of optimization iterations
  - More iterations → better convergence but slower

## Time and Space Complexity

- **Time**: O(n²) per iteration for pairwise distances
  - Total: O(n² · iterations)
  - Impractical for n > 10,000
- **Space**: O(n²) for distance and probability matrices

## Advantages

✓ **Non-linear**: Captures complex manifolds
✓ **Local Structure**: Preserves neighborhoods excellently
✓ **Visualization**: Best for 2D/3D plots
✓ **Cluster Revelation**: Makes clusters visually obvious

## Disadvantages

✗ **Slow**: O(n²) doesn't scale to large datasets
✗ **Stochastic**: Different runs give different embeddings
✗ **No Transform**: Cannot embed new data points
✗ **Global Structure**: Distances between clusters not meaningful
✗ **Tuning**: Sensitive to perplexity, learning rate, iterations

## Comparison with PCA

| Feature | t-SNE | PCA |
|---------|-------|-----|
| Type | Non-linear | Linear |
| Preserves | Local structure | Global variance |
| Speed | O(n²·iter) | O(n·d·k) |
| New Data | No | Yes |
| Stochastic | Yes | No |
| Use Case | Visualization | Preprocessing |

## When to Use

**Use t-SNE for:**
- Visualizing high-dimensional data (>3D)
- Exploratory data analysis
- Finding hidden clusters
- Presentations and reports (2D plots)

**Don't use t-SNE for:**
- Large datasets (n > 10,000)
- Feature reduction before modeling (use PCA instead)
- When you need to transform new data
- When global structure matters

## Best Practices

1. **Normalize data** before t-SNE (different scales affect distances)
2. **Try multiple perplexity values** (5, 10, 30, 50) to see different structures
3. **Run multiple times** with different random seeds (stochastic)
4. **Use enough iterations** (500-1000 minimum)
5. **Don't over-interpret** distances between clusters
6. **Consider PCA first** if dataset > 50 dimensions (reduce to ~50D first)

## Example Usage

```rust
use aprender::prelude::*;

// High-dimensional data
let data = Matrix::from_vec(100, 50, high_dim_data)?;

// Reduce to 2D for visualization
let mut tsne = TSNE::new(2)
    .with_perplexity(30.0)
    .with_n_iter(1000)
    .with_random_state(42);

let embedding = tsne.fit_transform(&data)?;

// Plot embedding[i, 0] vs embedding[i, 1]
```

## References

1. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. JMLR, 9, 2579-2605.
2. Wattenberg, et al. (2016). How to Use t-SNE Effectively. Distill.
3. Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. Nature Communications, 10, 5416.
