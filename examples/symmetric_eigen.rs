#![allow(clippy::disallowed_methods)]
//! Symmetric Eigendecomposition Example
//!
//! Demonstrates trueno's SymmetricEigen for PCA and spectral analysis.
//! This replaces nalgebra dependency for eigendecomposition tasks.
//!
//! Run with: cargo run --example symmetric_eigen

use trueno::{Matrix, SymmetricEigen};

fn main() {
    println!("=== Trueno SymmetricEigen Demo ===\n");

    // Example 1: Simple 2x2 symmetric matrix
    println!("1. Simple 2x2 Eigendecomposition");
    println!("   Matrix: [[3, 1], [1, 3]]");

    let m = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]).expect("valid matrix");
    let eigen = SymmetricEigen::new(&m).expect("eigendecomposition");

    println!("   Eigenvalues (descending): {:?}", eigen.eigenvalues());
    println!("   Expected: [4.0, 2.0]");
    println!();

    // Example 2: PCA-style covariance matrix
    println!("2. PCA Covariance Matrix (3x3)");
    println!("   Simulating data with strong first principal component");

    // Covariance matrix with dominant first eigenvalue
    #[rustfmt::skip]
    let cov = Matrix::from_vec(3, 3, vec![
        5.0, 2.0, 1.0,
        2.0, 3.0, 0.5,
        1.0, 0.5, 1.0,
    ]).expect("valid matrix");

    let pca_eigen = SymmetricEigen::new(&cov).expect("eigendecomposition");

    println!("   Eigenvalues: {:?}", pca_eigen.eigenvalues());
    println!(
        "   First PC explains {:.1}% of variance",
        100.0 * pca_eigen.eigenvalues()[0] / pca_eigen.eigenvalues().iter().sum::<f32>()
    );
    println!();

    // Example 3: Verify A*v = lambda*v
    println!("3. Verification: A*v = lambda*v");
    for (i, (lambda, v)) in pca_eigen.iter().take(2).enumerate() {
        let av = cov.matvec(&v).expect("matvec");
        let lambda_v: Vec<f32> = v.as_slice().iter().map(|x| x * lambda).collect();

        let error: f32 =
            av.as_slice().iter().zip(lambda_v.iter()).map(|(a, b)| (a - b).abs()).sum();

        println!("   Eigenpair {}: lambda={:.4}, error={:.2e}", i, lambda, error);
    }
    println!();

    // Example 4: Reconstruction accuracy
    println!("4. Reconstruction: V * D * V^T = A");
    let reconstructed = pca_eigen.reconstruct().expect("reconstruction");

    let mut max_error = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let orig = cov.get(i, j).unwrap();
            let recon = reconstructed.get(i, j).unwrap();
            max_error = max_error.max((orig - recon).abs());
        }
    }
    println!("   Max reconstruction error: {:.2e}", max_error);
    println!();

    // Example 5: Larger matrix for performance
    println!("5. Performance: 100x100 matrix");
    let n = 100;
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let val = 1.0 / (1.0 + (i as f32 - j as f32).abs());
            data[i * n + j] = val;
            data[j * n + i] = val;
        }
        data[i * n + i] += n as f32; // Diagonal dominance
    }

    let large = Matrix::from_vec(n, n, data).expect("valid matrix");

    let start = std::time::Instant::now();
    let large_eigen = SymmetricEigen::new(&large).expect("eigendecomposition");
    let elapsed = start.elapsed();

    println!("   Computed {} eigenvalues in {:?}", large_eigen.len(), elapsed);
    println!("   Largest eigenvalue: {:.4}", large_eigen.eigenvalues()[0]);
    println!("   Smallest eigenvalue: {:.4}", large_eigen.eigenvalues()[n - 1]);
    println!();

    println!("=== Demo Complete ===");
    println!("trueno::SymmetricEigen provides:");
    println!("  - SIMD-accelerated Jacobi algorithm");
    println!("  - Eigenvalues sorted descending (PCA convention)");
    println!("  - Orthonormal eigenvectors");
    println!("  - No external dependencies (replaces nalgebra)");
}
