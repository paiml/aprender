# Case Study: PCA Iris

This case study demonstrates Principal Component Analysis (PCA) for dimensionality reduction on the famous Iris dataset, reducing 4D flower measurements to 2D while preserving 96% of variance.

## Overview

We'll apply PCA to Iris flower data to:
- Reduce 4 features (sepal/petal dimensions) to 2 principal components
- Analyze explained variance (how much information is preserved)
- Reconstruct original data and measure reconstruction error
- Understand principal component loadings (feature importance)

## Running the Example

```bash
cargo run --example pca_iris
```

Expected output: Step-by-step PCA analysis including standardization, dimensionality reduction, explained variance analysis, transformed data samples, reconstruction quality, and principal component loadings.

## Dataset

### Iris Flower Measurements (30 samples)

```rust,ignore
// Features: [sepal_length, sepal_width, petal_length, petal_width]
// 10 samples each from: Setosa, Versicolor, Virginica

let data = Matrix::from_vec(30, 4, vec![
    // Setosa (small petals, large sepals)
    5.1, 3.5, 1.4, 0.2,
    4.9, 3.0, 1.4, 0.2,
    ...
    // Versicolor (medium petals and sepals)
    7.0, 3.2, 4.7, 1.4,
    6.4, 3.2, 4.5, 1.5,
    ...
    // Virginica (large petals and sepals)
    6.3, 3.3, 6.0, 2.5,
    5.8, 2.7, 5.1, 1.9,
    ...
])?;
```

**Dataset characteristics**:
- 30 samples (10 per species)
- 4 features (all measurements in centimeters)
- 3 species with distinct morphological patterns

## Step 1: Standardizing Features

### Why Standardize?

PCA is sensitive to feature scales. Without standardization:
- Features with larger values dominate variance
- Example: Sepal length (4-8 cm) would dominate petal width (0.1-2.5 cm)
- Result: Principal components biased toward large-scale features

### Implementation

```rust,no_run
use aprender::preprocessing::{StandardScaler, PCA};
use aprender::traits::Transformer;

let mut scaler = StandardScaler::new();
let scaled_data = scaler.fit_transform(&data)?;
```

**StandardScaler transforms each feature** to zero mean and unit variance:
```text
X_scaled = (X - mean) / std
```

After standardization, all features contribute equally to PCA.

## Step 2: Applying PCA (4D → 2D)

### Dimensionality Reduction

```rust,no_run
let mut pca = PCA::new(2); // Keep 2 principal components
let transformed = pca.fit_transform(&scaled_data)?;

println!("Original shape: {:?}", data.shape());       // (30, 4)
println!("Reduced shape: {:?}", transformed.shape()); // (30, 2)
```

**What happens during fit**:
1. Compute covariance matrix: Σ = (X^T X) / (n-1)
2. Eigendecomposition: Σ v_i = λ_i v_i
3. Sort eigenvectors by eigenvalue (descending)
4. Keep top 2 eigenvectors as principal components

**Transform** projects data onto principal components:
```text
X_pca = (X - mean) @ components^T
```

## Step 3: Explained Variance Analysis

### Results

```text
Explained Variance by Component:
   PC1: 2.9501 (71.29%) ███████████████████████████████████
   PC2: 1.0224 (24.71%) ████████████

Total Variance Captured: 96.00%
Information Lost:        4.00%
```

### Interpretation

**PC1 (71.29% variance)**:
- Captures overall flower size
- Dominant direction of variation
- Likely separates Setosa (small) from Virginica (large)

**PC2 (24.71% variance)**:
- Captures petal vs sepal differences
- Secondary variation pattern
- Likely separates Versicolor from other species

**96% total variance**: Excellent dimensionality reduction
- Only 4% information loss
- 2D representation sufficient for visualization
- Suitable for downstream ML tasks

### Variance Ratios

```rust,no_run
let explained_var = pca.explained_variance()?;
let explained_ratio = pca.explained_variance_ratio()?;

for (i, (&var, &ratio)) in explained_var.iter()
                             .zip(explained_ratio.iter()).enumerate() {
    println!("PC{}: variance={:.4}, ratio={:.2}%",
             i+1, var, ratio*100.0);
}
```

**Eigenvalues (explained_variance)**:
- PC1: 2.9501 (variance captured)
- PC2: 1.0224
- Sum ≈ 4.0 (total variance of standardized data)

**Ratios sum to 1.0**: All variance accounted for.

## Step 4: Transformed Data

### Sample Output

```text
Sample      Species        PC1        PC2
────────────────────────────────────────────
     0       Setosa    -2.2055    -0.8904
     1       Setosa    -2.0411     0.4635
    10   Versicolor     0.9644    -0.8293
    11   Versicolor     0.6384    -0.6166
    20    Virginica     1.7447    -0.8603
    21    Virginica     1.0657     0.8717
```

### Visual Separation

**PC1 axis** (horizontal):
- Setosa: Negative values (~-2.2)
- Versicolor: Slightly positive (~0.8)
- Virginica: Positive values (~1.5)

**PC2 axis** (vertical):
- All species: Values range from -1 to +1
- Less separable than PC1

**Conclusion**: 2D projection enables easy visualization and classification of species.

## Step 5: Reconstruction (2D → 4D)

### Implementation

```rust,no_run
let reconstructed_scaled = pca.inverse_transform(&transformed)?;
let reconstructed = scaler.inverse_transform(&reconstructed_scaled)?;
```

**Inverse transform**:
```text
X_reconstructed = X_pca @ components^T + mean
```

### Reconstruction Error

```text
Reconstruction Error Metrics:
   MSE:        0.033770
   RMSE:       0.183767
   Max Error:  0.699232
```

**Sample Reconstruction**:
```text
Feature   Original  Reconstructed
──────────────────────────────────
Sample 0:
 Sepal L     5.1000         5.0208  (error: -0.08 cm)
 Sepal W     3.5000         3.5107  (error: +0.01 cm)
 Petal L     1.4000         1.4504  (error: +0.05 cm)
 Petal W     0.2000         0.2462  (error: +0.05 cm)
```

### Interpretation

**RMSE = 0.184**:
- Average reconstruction error is 0.184 cm
- Small compared to feature ranges (0.2-10 cm)
- Demonstrates 2D representation preserves most information

**Max error = 0.70 cm**:
- Worst-case reconstruction error
- Still reasonable for biological measurements
- Validates 96% variance capture claim

**Why not perfect reconstruction?**
- 2 components < 4 original features
- 4% variance discarded
- Trade-off: compression vs accuracy

## Step 6: Principal Component Loadings

### Feature Importance

```text
 Component    Sepal L    Sepal W    Petal L    Petal W
──────────────────────────────────────────────────────
       PC1     0.5310    -0.2026     0.5901     0.5734
       PC2    -0.3407    -0.9400     0.0033    -0.0201
```

### Interpretation

**PC1 (overall size)**:
- Positive loadings: Sepal L (0.53), Petal L (0.59), Petal W (0.57)
- Negative loading: Sepal W (-0.20)
- **Meaning**: Larger flowers score high on PC1
- Separates Setosa (small) vs Virginica (large)

**PC2 (petal vs sepal differences)**:
- Strong negative: Sepal W (-0.94)
- Near-zero: Petal L (0.003), Petal W (-0.02)
- **Meaning**: Captures sepal width variation
- Separates species by sepal shape

### Mathematical Properties

**Orthogonality**: PC1 ⊥ PC2
```rust,no_run
let components = pca.components()?;
let dot_product = (0..4).map(|k| {
    components.get(0, k) * components.get(1, k)
}).sum::<f32>();
assert!(dot_product.abs() < 1e-6); // ≈ 0
```

**Unit length**: ‖v_i‖ = 1
```rust,no_run
let norm_sq = (0..4).map(|k| {
    let val = components.get(0, k);
    val * val
}).sum::<f32>();
assert!((norm_sq.sqrt() - 1.0).abs() < 1e-6); // ≈ 1
```

## Performance Metrics

### Time Complexity

| Operation | Iris Dataset | General (n×p) |
|-----------|-------------|---------------|
| Standardization | 0.12 ms | O(n·p) |
| Covariance | 0.05 ms | O(p²·n) |
| Eigendecomposition | 0.03 ms | O(p³) |
| Transform | 0.02 ms | O(n·k·p) |
| **Total** | **0.22 ms** | **O(p³ + p²·n)** |

**Bottleneck**: Eigendecomposition O(p³)
- Iris: p=4, very fast (0.03 ms)
- High-dimensional: p>10,000, use truncated SVD

### Memory Usage

**Iris example**:
- Centered data: 30×4×4 = 480 bytes
- Covariance matrix: 4×4×4 = 64 bytes
- Components stored: 2×4×4 = 32 bytes
- **Total**: ~576 bytes

**General formula**: 4(n·p + p²) bytes

## Key Takeaways

### When to Use PCA

✓ **Visualization**: Reduce to 2D/3D for plotting\
✓ **Preprocessing**: Remove correlated features before ML\
✓ **Compression**: Reduce storage by 50%+ with minimal information loss\
✓ **Denoising**: Discard low-variance (noisy) components

### PCA Assumptions

1. **Linear relationships**: PCA captures linear structure only
2. **Variance = importance**: High-variance directions are informative
3. **Standardization required**: Features must be on similar scales
4. **Orthogonal components**: Each PC independent of others

### Best Practices

1. **Always standardize** before PCA (unless features already scaled)
2. **Check explained variance**: Aim for 90-95% cumulative
3. **Interpret loadings**: Understand what each PC represents
4. **Validate reconstruction**: Low RMSE confirms quality
5. **Visualize 2D projection**: Verify species separation

## Full Code

```rust,no_run
use aprender::preprocessing::{StandardScaler, PCA};
use aprender::primitives::Matrix;
use aprender::traits::Transformer;

// 1. Load data
let data = Matrix::from_vec(30, 4, iris_data)?;

// 2. Standardize
let mut scaler = StandardScaler::new();
let scaled = scaler.fit_transform(&data)?;

// 3. Apply PCA
let mut pca = PCA::new(2);
let reduced = pca.fit_transform(&scaled)?;

// 4. Analyze variance
let var_ratio = pca.explained_variance_ratio().unwrap();
println!("Variance: {:.1}%", var_ratio.iter().sum::<f32>() * 100.0);

// 5. Reconstruct
let reconstructed_scaled = pca.inverse_transform(&reduced)?;
let reconstructed = scaler.inverse_transform(&reconstructed_scaled)?;

// 6. Compute error
let rmse = compute_rmse(&data, &reconstructed);
println!("RMSE: {:.4}", rmse);
```

## Further Exploration

**Try different n_components**:
```rust,no_run
let mut pca1 = PCA::new(1);  // ~71% variance
let mut pca3 = PCA::new(3);  // ~99% variance
let mut pca4 = PCA::new(4);  // 100% variance (perfect reconstruction)
```

**Analyze per-species variance**:
- Compute PCA separately for each species
- Compare principal directions
- Identify species-specific variation patterns

**Compare with other methods**:
- LDA: Supervised dimensionality reduction (uses labels)
- t-SNE: Non-linear visualization (preserves local structure)
- UMAP: Non-linear, faster than t-SNE

## Related Examples

- [`examples/iris_clustering.rs`](./iris-clustering.md) - K-Means on same dataset
- [`book/src/ml-fundamentals/pca.md`](../ml-fundamentals/pca.md) - Full PCA theory
