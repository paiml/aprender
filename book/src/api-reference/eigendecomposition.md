# Eigendecomposition

The `SymmetricEigen` type provides eigendecomposition for symmetric matrices, essential for PCA, spectral clustering, and scientific computing.

## Basic Usage

```rust
use trueno::{Matrix, SymmetricEigen};

// Create a symmetric matrix
let m = Matrix::<f32>::from_vec(3, 3, vec![
    4.0, 2.0, 0.0,
    2.0, 5.0, 3.0,
    0.0, 3.0, 6.0,
])?;

// Compute eigendecomposition
let eigen = SymmetricEigen::new(&m)?;

// Access results
let eigenvalues = eigen.eigenvalues();     // Sorted descending
let eigenvectors = eigen.eigenvectors();   // As matrix (columns = eigenvectors)
```

## Eigenvalues

Eigenvalues are returned in **descending order** (PCA convention).

```rust
let eigen = SymmetricEigen::new(&covariance_matrix)?;

// Largest eigenvalue first
let principal = eigen.eigenvalues()[0];

// Variance explained by first PC
let total_variance: f32 = eigen.eigenvalues().iter().sum();
let explained = eigen.eigenvalues()[0] / total_variance;
println!("First PC explains {:.1}% of variance", explained * 100.0);
```

## Eigenvectors

Eigenvectors form an orthonormal basis.

```rust
let eigen = SymmetricEigen::new(&m)?;

// Get i-th eigenvector as a Vector
let v0 = eigen.eigenvector(0)?;

// Eigenvectors are orthonormal
let dot = v0.dot(&eigen.eigenvector(1)?)?;
assert!(dot.abs() < 1e-5); // ≈ 0

// Unit length
let norm = v0.norm_l2()?;
assert!((norm - 1.0).abs() < 1e-5); // ≈ 1
```

## Verification

Verify A × v = λ × v for each eigenpair.

```rust
let eigen = SymmetricEigen::new(&m)?;

for i in 0..eigen.len() {
    let lambda = eigen.eigenvalues()[i];
    let v = eigen.eigenvector(i)?;

    let av = m.matvec(&v)?;
    let lambda_v = v.scale(lambda)?;

    let error: f32 = av.sub(&lambda_v)?
        .as_slice()
        .iter()
        .map(|x| x.abs())
        .sum();

    assert!(error < 1e-5, "Eigenpair {} invalid", i);
}
```

## Reconstruction

Reconstruct the original matrix: A = V × D × Vᵀ

```rust
let eigen = SymmetricEigen::new(&m)?;

// V * diag(eigenvalues) * V^T should equal original matrix
let reconstructed = eigen.reconstruct();
let error = m.frobenius_distance(&reconstructed);
assert!(error < 1e-5);
```

## GPU Acceleration

For large matrices, use GPU backend.

```rust
use trueno::GpuBackend;

let mut gpu = GpuBackend::new();
let large = Matrix::<f32>::from_vec(256, 256, /* ... */)?;

let (eigenvalues, eigenvectors) = gpu.symmetric_eigen(
    large.as_slice(),
    256
)?;
```

## Algorithm Details

Trueno uses the **Jacobi eigenvalue algorithm**:

- **Numerically stable**: Based on Golub & Van Loan formulation
- **Convergence**: Quadratic convergence for well-conditioned matrices
- **SIMD-optimized**: Jacobi rotations use SIMD where beneficial
- **Accuracy**: Results match nalgebra to 1e-5 tolerance

## Performance

| Matrix Size | Trueno | nalgebra | Speedup |
|-------------|--------|----------|---------|
| 64×64 | 12ms | 18ms | 1.5x |
| 128×128 | 378µs | 491µs | 1.3x |
| 256×256 | 1.28ms | 2.80ms | 2.2x |

## Use Cases

1. **PCA (Principal Component Analysis)**
   ```rust
   let cov = compute_covariance(&data);
   let eigen = SymmetricEigen::new(&cov)?;
   let top_k_components = &eigen.eigenvalues()[0..k];
   ```

2. **Spectral Clustering**
   ```rust
   let laplacian = compute_graph_laplacian(&adjacency);
   let eigen = SymmetricEigen::new(&laplacian)?;
   let fiedler_vector = eigen.eigenvector(1)?; // 2nd smallest
   ```

3. **Vibration Analysis**
   ```rust
   let stiffness = compute_stiffness_matrix(&structure);
   let eigen = SymmetricEigen::new(&stiffness)?;
   let natural_frequencies: Vec<f32> = eigen.eigenvalues()
       .iter()
       .map(|&λ| λ.sqrt())
       .collect();
   ```
