#![allow(clippy::disallowed_methods, clippy::float_cmp)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno::Matrix;

/// Generate a non-trivial f32 data vector for benchmarking.
/// Uses `(i * stride) % modulus` to produce repeatable, non-zero patterns.
fn gen_data(len: usize, stride: usize, modulus: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * stride) % modulus) as f32).collect()
}

/// Create a pair of matrices `(m x n)` and `(n x p)` for matmul benchmarks.
fn make_matmul_pair(m: usize, n: usize, p: usize, b_stride: usize) -> (Matrix<f32>, Matrix<f32>) {
    let a = Matrix::from_vec(m, n, gen_data(m * n, 1, 100)).unwrap();
    let b = Matrix::from_vec(n, p, gen_data(n * p, b_stride, 100)).unwrap();
    (a, b)
}

/// Benchmark matmul across a list of `(m, n, p)` shapes within a named group.
fn bench_matmul_group(
    c: &mut Criterion,
    group_name: &str,
    shapes: &[(usize, usize, usize)],
    b_stride: usize,
) {
    let mut group = c.benchmark_group(group_name);
    for &(m, n, p) in shapes {
        let id = format!("{m}x{n}_x_{n}x{p}");
        let (a, b) = make_matmul_pair(m, n, p, b_stride);
        group.bench_with_input(BenchmarkId::from_parameter(&id), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(black_box(a).matmul(black_box(b)).unwrap()));
        });
    }
    group.finish();
}

/// Benchmark convolve2d across a list of configs within a named group.
/// Each config is `(rows, cols, description, kernel)`.
fn bench_convolve2d_group(
    c: &mut Criterion,
    group_name: &str,
    configs: &[(usize, usize, &str, Matrix<f32>)],
) {
    let mut group = c.benchmark_group(group_name);
    for (rows, cols, desc, kernel) in configs {
        let input_data: Vec<f32> = (0..rows * cols).map(|i| ((i % 256) as f32) / 255.0).collect();
        let input = Matrix::from_vec(*rows, *cols, input_data).unwrap();
        let id = format!("{rows}x{cols}_k{}", kernel.rows());
        group.bench_with_input(
            BenchmarkId::new(*desc, &id),
            &(&input, kernel),
            |bench, (input, kernel)| {
                bench.iter(|| black_box(black_box(input).convolve2d(black_box(kernel)).unwrap()));
            },
        );
    }
    group.finish();
}

/// Build a uniform averaging kernel of the given size.
fn averaging_kernel(size: usize) -> Matrix<f32> {
    let val = 1.0 / (size * size) as f32;
    Matrix::from_vec(size, size, vec![val; size * size]).unwrap()
}

fn bench_matmul_sizes(c: &mut Criterion) {
    // Test various matrix sizes to show SIMD threshold effect
    let sizes = [
        (16, 16, 16),       // Small: below SIMD threshold (64)
        (32, 32, 32),       // Medium: below threshold
        (64, 64, 64),       // At threshold
        (128, 128, 128),    // Large: above threshold (SIMD should shine)
        (256, 256, 256),    // Very large: maximum SIMD benefit
        (512, 512, 512),    // Phase 3: Large matrix baseline
        (1024, 1024, 1024), // Phase 3: Very large matrix baseline
    ];
    bench_matmul_group(c, "matmul", &sizes, 2);
}

fn bench_matmul_rectangular(c: &mut Criterion) {
    // Common ML shapes (batch processing)
    let shapes = [
        (32, 128, 64),   // Small batch
        (64, 256, 128),  // Medium batch
        (128, 512, 256), // Large batch (neural network layer)
    ];
    bench_matmul_group(c, "matmul_rectangular", &shapes, 3);
}

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    let sizes = [(64, 64), (128, 128), (256, 256), (128, 256)];

    for (rows, cols) in sizes {
        let id = format!("{rows}x{cols}");
        let m = Matrix::from_vec(rows, cols, gen_data(rows * cols, 1, 100)).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(&id), &m, |bench, m| {
            bench.iter(|| black_box(black_box(m).transpose()));
        });
    }

    group.finish();
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec");

    let sizes = [(64, 64), (128, 128), (256, 256), (512, 512)];

    for (rows, cols) in sizes {
        let id = format!("{rows}x{cols}_x_{cols}");
        let m = Matrix::from_vec(rows, cols, gen_data(rows * cols, 1, 100)).unwrap();
        let v = trueno::Vector::from_slice(&gen_data(cols, 1, 100));
        group.bench_with_input(BenchmarkId::from_parameter(&id), &(&m, &v), |bench, (m, v)| {
            bench.iter(|| black_box(black_box(m).matvec(black_box(v)).unwrap()));
        });
    }

    group.finish();
}

fn bench_convolve2d(c: &mut Criterion) {
    // Test various image and kernel sizes for CNN/image processing workloads
    let configs: Vec<(usize, usize, &str, Matrix<f32>)> = vec![
        (32, 32, "small_3x3", averaging_kernel(3)),
        (64, 64, "medium_3x3", averaging_kernel(3)),
        (128, 128, "large_3x3", averaging_kernel(3)),
        (256, 256, "xlarge_3x3", averaging_kernel(3)),
        (128, 128, "large_5x5", averaging_kernel(5)),
        (256, 256, "xlarge_5x5", averaging_kernel(5)),
        (512, 512, "xxlarge_3x3", averaging_kernel(3)),
    ];
    bench_convolve2d_group(c, "convolve2d", &configs);
}

fn bench_convolve2d_edge_detection(c: &mut Criterion) {
    // Sobel horizontal kernel
    #[rustfmt::skip]
    let sobel_h = Matrix::from_vec(3, 3, vec![
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ]).unwrap();

    let configs: Vec<(usize, usize, &str, Matrix<f32>)> = vec![
        (128, 128, "small", sobel_h.clone()),
        (256, 256, "medium", sobel_h.clone()),
        (512, 512, "large", sobel_h),
    ];
    bench_convolve2d_group(c, "convolve2d_edge_detection", &configs);
}

criterion_group!(
    benches,
    bench_matmul_sizes,
    bench_matmul_rectangular,
    bench_transpose,
    bench_matvec,
    bench_convolve2d,
    bench_convolve2d_edge_detection
);
criterion_main!(benches);
