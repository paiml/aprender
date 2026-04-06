#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Eigendecomposition benchmarks comparing trueno vs nalgebra
//!
//! Issue: https://github.com/paiml/trueno/issues/63
//! Goal: Demonstrate that trueno's SymmetricEigen can replace nalgebra dependency

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Create a symmetric positive definite matrix for benchmarking
fn create_spd_matrix(n: usize) -> Vec<f32> {
    // Create a well-conditioned SPD matrix using: A = I + (1/n) * B * B^T
    // where B has small off-diagonal elements
    let mut result = vec![0.0f32; n * n];

    // Start with scaled identity matrix (strongly diagonal dominant)
    for i in 0..n {
        result[i * n + i] = (n as f32) + 1.0;
    }

    // Add small symmetric perturbations
    for i in 0..n {
        for j in 0..i {
            // Small off-diagonal values that won't dominate
            let val = 0.1 * ((i + j) % 5) as f32 / (n as f32);
            result[i * n + j] = val;
            result[j * n + i] = val;
        }
    }

    result
}

fn bench_eigen_trueno(c: &mut Criterion) {
    use trueno::{Matrix, SymmetricEigen};

    let mut group = c.benchmark_group("eigen_trueno");

    // Test various matrix sizes
    let sizes = vec![
        16,  // Small
        32,  // Medium
        64,  // Above SIMD threshold
        128, // Large
        256, // Very large
    ];

    for n in sizes {
        let data = create_spd_matrix(n);
        let matrix = Matrix::from_vec(n, n, data).expect("valid matrix");

        group.bench_with_input(BenchmarkId::from_parameter(n), &matrix, |bench, matrix| {
            bench.iter(|| {
                let eigen = SymmetricEigen::new(black_box(matrix)).expect("eigen should succeed");
                black_box(eigen);
            });
        });
    }

    group.finish();
}

fn bench_eigen_nalgebra(c: &mut Criterion) {
    use nalgebra::DMatrix;

    let mut group = c.benchmark_group("eigen_nalgebra");

    let sizes = vec![16, 32, 64, 128, 256];

    for n in sizes {
        let data = create_spd_matrix(n);
        // nalgebra uses column-major, but we created row-major
        // For symmetric matrices, this doesn't matter
        let matrix = DMatrix::from_row_slice(n, n, &data);

        group.bench_with_input(BenchmarkId::from_parameter(n), &matrix, |bench, matrix| {
            bench.iter(|| {
                let eigen = black_box(matrix).clone().symmetric_eigen();
                black_box(eigen);
            });
        });
    }

    group.finish();
}

fn bench_eigen_comparison(c: &mut Criterion) {
    use nalgebra::DMatrix;
    use trueno::{Matrix, SymmetricEigen};

    let mut group = c.benchmark_group("eigen_comparison");

    // Focus on the critical sizes for PCA/ML workloads
    let sizes = vec![64, 128, 256];

    for n in sizes {
        let data = create_spd_matrix(n);

        // Trueno matrix
        let trueno_matrix = Matrix::from_vec(n, n, data.clone()).expect("valid matrix");

        // Nalgebra matrix
        let nalgebra_matrix = DMatrix::from_row_slice(n, n, &data);

        // Benchmark trueno
        group.bench_with_input(BenchmarkId::new("trueno", n), &trueno_matrix, |bench, matrix| {
            bench.iter(|| {
                let eigen = SymmetricEigen::new(black_box(matrix)).expect("eigen should succeed");
                black_box(eigen);
            });
        });

        // Benchmark nalgebra
        group.bench_with_input(
            BenchmarkId::new("nalgebra", n),
            &nalgebra_matrix,
            |bench, matrix| {
                bench.iter(|| {
                    let eigen = black_box(matrix).clone().symmetric_eigen();
                    black_box(eigen);
                });
            },
        );
    }

    group.finish();
}

fn bench_eigen_reconstruction(c: &mut Criterion) {
    use trueno::{Matrix, SymmetricEigen};

    let mut group = c.benchmark_group("eigen_reconstruction");

    let sizes = vec![32, 64, 128];

    for n in sizes {
        let data = create_spd_matrix(n);
        let matrix = Matrix::from_vec(n, n, data).expect("valid matrix");
        let eigen = SymmetricEigen::new(&matrix).expect("eigen should succeed");

        group.bench_with_input(BenchmarkId::from_parameter(n), &eigen, |bench, eigen| {
            bench.iter(|| {
                let reconstructed = black_box(eigen).reconstruct().expect("reconstruct");
                black_box(reconstructed);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_eigen_trueno,
    bench_eigen_nalgebra,
    bench_eigen_comparison,
    bench_eigen_reconstruction
);
criterion_main!(benches);
