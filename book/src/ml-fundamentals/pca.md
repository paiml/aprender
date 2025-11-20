# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a fundamental dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving as much variance as possible. This chapter covers the theory, implementation, and practical considerations for using PCA in aprender.

## Why Dimensionality Reduction?

High-dimensional data presents several challenges:

- **Curse of dimensionality**: Distance metrics become less meaningful in high dimensions
- **Visualization**: Impossible to visualize data beyond 3D
- **Computational cost**: Training time grows with dimensionality
- **Overfitting**: More features increase risk of spurious correlations
- **Storage**: High-dimensional data requires more memory

PCA addresses these challenges by finding a lower-dimensional subspace that captures most of the data's variance.

## Mathematical Foundation

### Core Idea

PCA finds orthogonal directions (principal components) along which data varies the most. These directions are the eigenvectors of the covariance matrix.

**Steps**:
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors
4. Project data onto top-k eigenvectors

### Covariance Matrix

For centered data matrix **X** (n samples × p features):

```text
Σ = (X^T X) / (n - 1)
```

The covariance matrix Σ is:
- Symmetric: Σ = Σ^T
- Positive semi-definite: all eigenvalues ≥ 0
- Size: p × p (independent of n)

### Eigendecomposition

The eigenvectors of Σ form the principal components:

```text
Σ v_i = λ_i v_i
```

where:
- `v_i` = i-th principal component (eigenvector)
- `λ_i` = variance explained by v_i (eigenvalue)

**Key properties**:
- Eigenvectors are orthogonal: `v_i ⊥ v_j` for i ≠ j
- Eigenvalues sum to total variance: `Σ λ_i = trace(Σ)`
- Components ordered by decreasing eigenvalue

### Projection

To project data onto k principal components:

```text
X_pca = (X - μ) W_k

where:
  μ = column means
  W_k = [v_1, v_2, ..., v_k]  (p × k matrix)
```

### Reconstruction

To reconstruct original space from reduced dimensions:

```text
X_reconstructed = X_pca W_k^T + μ
```

Perfect reconstruction when k = p (all components kept).

## Implementation in Aprender

### Basic Usage

```rust,ignore
use aprender::preprocessing::{PCA, StandardScaler};
use aprender::traits::Transformer;
use aprender::primitives::Matrix;

// Always standardize first (PCA is scale-sensitive)
let mut scaler = StandardScaler::new();
let scaled_data = scaler.fit_transform(&data)?;

// Reduce from 4D to 2D
let mut pca = PCA::new(2);
let reduced = pca.fit_transform(&scaled_data)?;

// Analyze explained variance
let var_ratio = pca.explained_variance_ratio().unwrap();
println!("PC1 explains {:.1}%", var_ratio[0] * 100.0);
println!("PC2 explains {:.1}%", var_ratio[1] * 100.0);

// Reconstruct original space
let reconstructed = pca.inverse_transform(&reduced)?;
```

### Transformer Trait

PCA implements the `Transformer` trait:

```rust,ignore
pub trait Transformer {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<(), &'static str>;
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str>;
    fn fit_transform(&mut self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str> {
        self.fit(x)?;
        self.transform(x)
    }
}
```

This enables:
- **Fit on training data** → Learn components
- **Transform test data** → Apply same projection
- **Pipeline compatibility** → Chain with other transformers

### Explained Variance

```rust,ignore
let explained_var = pca.explained_variance().unwrap();
let explained_ratio = pca.explained_variance_ratio().unwrap();

// Cumulative variance
let mut cumsum = 0.0;
for (i, ratio) in explained_ratio.iter().enumerate() {
    cumsum += ratio;
    println!("PC{}: {:.2}% (cumulative: {:.2}%)",
             i+1, ratio*100.0, cumsum*100.0);
}
```

**Rule of thumb**: Keep components until 90-95% variance explained.

### Principal Components (Loadings)

```rust,ignore
let components = pca.components().unwrap();
let (n_components, n_features) = components.shape();

for i in 0..n_components {
    println!("PC{} loadings:", i+1);
    for j in 0..n_features {
        println!("  Feature {}: {:.4}", j, components.get(i, j));
    }
}
```

**Interpretation**:
- Larger absolute values = more important for that component
- Sign indicates direction of influence
- Orthogonal components capture different variation patterns

## Time and Space Complexity

### Computational Cost

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Center data | O(n · p) | O(n · p) |
| Covariance matrix | O(p² · n) | O(p²) |
| Eigendecomposition | O(p³) | O(p²) |
| Transform | O(n · k · p) | O(n · k) |
| Inverse transform | O(n · k · p) | O(n · p) |

where:
- n = number of samples
- p = number of features
- k = number of components

**Bottleneck**: Eigendecomposition is O(p³), making PCA impractical for p > 10,000 without specialized methods (truncated SVD, randomized PCA).

### Memory Requirements

**During fit**:
- Centered data: 4n·p bytes (f32)
- Covariance matrix: 4p² bytes
- Eigenvectors: 4k·p bytes (stored components)
- **Total**: ~4(n·p + p²) bytes

**Example** (1000 samples, 100 features):
- 0.4 MB centered data
- 0.04 MB covariance
- **Total**: ~0.44 MB

**Scaling**: Memory dominated by n·p term for large datasets.

## Choosing the Number of Components

### Methods

1. **Variance threshold**: Keep components explaining ≥ 90% variance

```rust,ignore
let ratios = pca.explained_variance_ratio().unwrap();
let mut cumsum = 0.0;
let mut k = 0;
for ratio in ratios {
    cumsum += ratio;
    k += 1;
    if cumsum >= 0.90 {
        break;
    }
}
println!("Need {} components for 90% variance", k);
```

2. **Scree plot**: Look for "elbow" where eigenvalues plateau

3. **Kaiser criterion**: Keep components with eigenvalue > 1.0

4. **Domain knowledge**: Use as many components as interpretable

### Tradeoffs

| Fewer Components | More Components |
|-----------------|----------------|
| Faster training | Better reconstruction |
| Less overfitting risk | Preserves subtle patterns |
| Simpler models | Higher computational cost |
| Information loss | Potential overfitting |

## When to Use PCA

### Good Use Cases

✓ **Visualization**: Reduce to 2D/3D for plotting
✓ **Preprocessing**: Remove correlated features before ML
✓ **Compression**: Reduce storage for large datasets
✓ **Denoising**: Remove low-variance (noisy) dimensions
✓ **Regularization**: Prevent overfitting in high dimensions

### When PCA Fails

✗ **Non-linear structure**: PCA only captures linear relationships
✗ **Outliers**: Covariance sensitive to extreme values
✗ **Sparse data**: Text/categorical data better handled by other methods
✗ **Interpretability required**: Principal components are linear combinations
✗ **Class separation not along high-variance directions**: Use LDA instead

## Algorithm Details

### Eigendecomposition Implementation

Aprender uses **nalgebra's SymmetricEigen** for covariance matrix eigendecomposition:

```rust,ignore
use nalgebra::{DMatrix, SymmetricEigen};

let cov_matrix = DMatrix::from_row_slice(n_features, n_features, &cov);
let eigen = SymmetricEigen::new(cov_matrix);

let eigenvalues = eigen.eigenvalues;   // sorted ascending by default
let eigenvectors = eigen.eigenvectors; // corresponding eigenvectors
```

**Why SymmetricEigen?**
- Covariance matrices are symmetric positive semi-definite
- Specialized algorithms (Jacobi, LAPACK SYEV) exploit symmetry
- Guarantees real eigenvalues and orthogonal eigenvectors
- More numerically stable than general eigendecomposition

### Numerical Stability

**Potential issues**:
1. **Catastrophic cancellation**: Subtracting nearly-equal numbers in covariance
2. **Eigenvalue precision**: Small eigenvalues may be computed inaccurately
3. **Degeneracy**: Multiple eigenvalues ≈ λ lead to non-unique eigenvectors

**Aprender's approach**:
- Use f32 (single precision) for memory efficiency
- Center data before covariance to reduce magnitude differences
- Sort eigenvalues/vectors explicitly (not relying on solver ordering)
- Components normalized to unit length (‖v_i‖ = 1)

## Standardization Best Practice

**Always standardize before PCA**:

```rust,ignore
let mut scaler = StandardScaler::new();
let scaled = scaler.fit_transform(&data)?;
let mut pca = PCA::new(n_components);
let reduced = pca.fit_transform(&scaled)?;
```

**Why?**
- Features with larger scales dominate variance
- Example: Age (0-100) vs Income ($0-$1M) → Income dominates
- Standardization ensures each feature contributes equally

**When not to standardize**:
- Features already on same scale (e.g., all pixel intensities 0-255)
- Domain knowledge suggests unequal weighting is correct

## Comparison with Other Methods

| Method | Linear? | Supervised? | Preserves | Use Case |
|--------|---------|-------------|-----------|----------|
| PCA | Yes | No | Variance | Unsupervised, visualization |
| LDA | Yes | Yes | Class separation | Classification preprocessing |
| t-SNE | No | No | Local structure | Visualization only |
| Autoencoders | No | No | Reconstruction | Non-linear compression |
| Feature selection | N/A | Optional | Original features | Interpretability |

**PCA advantages**:
- Fast (closed-form solution)
- Deterministic (no random initialization)
- Interpretable components (linear combinations)
- Mathematical guarantees (optimal variance preservation)

## Example: Iris Dataset

Complete example from `examples/pca_iris.rs`:

```rust,ignore
use aprender::preprocessing::{PCA, StandardScaler};
use aprender::traits::Transformer;

// 1. Standardize
let mut scaler = StandardScaler::new();
let scaled = scaler.fit_transform(&iris_data)?;

// 2. Apply PCA (4D → 2D)
let mut pca = PCA::new(2);
let reduced = pca.fit_transform(&scaled)?;

// 3. Analyze results
let var_ratio = pca.explained_variance_ratio().unwrap();
println!("Variance captured: {:.1}%",
         var_ratio.iter().sum::<f32>() * 100.0);

// 4. Reconstruct
let reconstructed_scaled = pca.inverse_transform(&reduced)?;
let reconstructed = scaler.inverse_transform(&reconstructed_scaled)?;

// 5. Compute reconstruction error
let rmse = compute_rmse(&iris_data, &reconstructed);
println!("Reconstruction RMSE: {:.4}", rmse);
```

**Typical results**:
- PC1 + PC2 capture ~96% of Iris variance
- 2D projection enables visualization of 3 species
- RMSE ≈ 0.18 (small reconstruction error)

## Further Reading

- **Foundations**: Jolliffe, I.T. "Principal Component Analysis" (2002)
- **SVD connection**: PCA via SVD instead of covariance eigendecomposition
- **Kernel PCA**: Non-linear extension using kernel trick
- **Incremental PCA**: Online algorithm for streaming data
- **Randomized PCA**: Approximate PCA for very high dimensions (p > 10,000)

## API Reference

```rust,ignore
// Constructor
pub fn new(n_components: usize) -> Self

// Transformer trait
fn fit(&mut self, x: &Matrix<f32>) -> Result<(), &'static str>
fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str>

// Accessors
pub fn explained_variance(&self) -> Option<&[f32]>
pub fn explained_variance_ratio(&self) -> Option<&[f32]>
pub fn components(&self) -> Option<&Matrix<f32>>

// Reconstruction
pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str>
```

**See also**:
- `preprocessing::StandardScaler` - Always use before PCA
- `examples/pca_iris.rs` - Complete walkthrough
- `traits::Transformer` - Composable preprocessing pipeline
