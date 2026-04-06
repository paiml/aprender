#![allow(clippy::disallowed_methods)]
//! Benchmark vector-matrix multiply (vocab projection pattern)
use std::time::Instant;
use trueno::{Matrix, Vector};

/// Compute GFLOPS from matrix dimensions and elapsed time in ms.
fn compute_gflops(rows: usize, inner: usize, cols: usize, time_ms: f64) -> f64 {
    (2.0 * rows as f64 * inner as f64 * cols as f64) / (time_ms / 1000.0) / 1e9
}

/// Benchmark Matrix::matmul approach.
fn bench_matmul(ma: &Matrix<f32>, mb: &Matrix<f32>, iterations: usize) -> f64 {
    // Warmup
    for _ in 0..3 {
        let _ = ma.matmul(mb);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ma.matmul(mb).unwrap();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark transposed dot-product approach.
fn bench_transposed_dots(
    va: &Vector<f32>,
    b_t: &[f32],
    inner: usize,
    cols: usize,
    iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..3 {
        let mut result = vec![0.0_f32; cols];
        for (j, result_elem) in result.iter_mut().enumerate() {
            let col_start = j * inner;
            let vb = Vector::from_slice(&b_t[col_start..col_start + inner]);
            *result_elem = va.dot(&vb).unwrap();
        }
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let mut result = vec![0.0_f32; cols];
        for (j, result_elem) in result.iter_mut().enumerate() {
            let col_start = j * inner;
            let vb = Vector::from_slice(&b_t[col_start..col_start + inner]);
            *result_elem = va.dot(&vb).unwrap();
        }
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark naive scalar approach.
fn bench_scalar(a: &[f32], b: &[f32], inner: usize, cols: usize, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        let mut result = vec![0.0_f32; cols];
        for j in 0..cols {
            let mut sum = 0.0_f32;
            for i in 0..inner {
                sum += a[i] * b[i * cols + j];
            }
            result[j] = sum;
        }
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

fn main() {
    // Whisper vocab projection: 1x384 @ 384x51865
    let rows = 1;
    let inner = 384;
    let cols = 51865;
    let iterations = 10;

    println!("Benchmarking vocab projection pattern: {rows}x{inner} @ {inner}x{cols}");
    println!("Total ops: {} million", (rows * inner * cols) as f64 / 1e6);

    // Create test data
    let a: Vec<f32> = (0..rows * inner).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..inner * cols).map(|i| (i as f32) * 0.0001).collect();

    // Method 1: Full matmul via Matrix
    let ma = Matrix::from_vec(rows, inner, a.clone()).unwrap();
    let mb = Matrix::from_vec(inner, cols, b.clone()).unwrap();
    let matmul_time = bench_matmul(&ma, &mb, iterations);
    let matmul_gflops = compute_gflops(rows, inner, cols, matmul_time);
    println!("\nMatrix::matmul: {:.1}ms ({:.2} GFLOPS)", matmul_time, matmul_gflops);

    // Method 2: Transpose B to column-major, then use dots
    let mut b_t = vec![0.0_f32; inner * cols];
    for i in 0..inner {
        for j in 0..cols {
            b_t[j * inner + i] = b[i * cols + j];
        }
    }
    let va = Vector::from_slice(&a);
    let transposed_time = bench_transposed_dots(&va, &b_t, inner, cols, iterations);
    let transposed_gflops = compute_gflops(rows, inner, cols, transposed_time);
    println!(
        "Vector dots (transposed): {:.1}ms ({:.2} GFLOPS)",
        transposed_time, transposed_gflops
    );

    // Method 3: Direct scalar (baseline)
    let scalar_time = bench_scalar(&a, &b, inner, cols, iterations);
    let scalar_gflops = compute_gflops(rows, inner, cols, scalar_time);
    println!("Scalar (naive): {:.1}ms ({:.2} GFLOPS)", scalar_time, scalar_gflops);

    println!("\n=== ANALYSIS ===");
    println!("Matrix::matmul vs scalar: {:.1}x", scalar_time / matmul_time);
    println!("Transposed dots vs scalar: {:.1}x", scalar_time / transposed_time);
    println!(
        "Potential speedup for matmul: {:.1}x",
        matmul_time / transposed_time.min(scalar_time)
    );
}
