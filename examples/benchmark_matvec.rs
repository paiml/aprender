#![allow(clippy::disallowed_methods)]
use std::time::Instant;
use trueno::Matrix;

fn main() {
    println!("Matrix-Vector Multiplication Benchmark\n");

    // Test different matrix sizes
    let sizes = vec![(100, 100), (500, 500), (1000, 1000), (2000, 2000)];

    for (rows, cols) in sizes {
        println!("=== Matrix {}×{} × Vector {} ===", rows, cols, cols);

        // Create matrix and vector
        let matrix = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols).map(|i| ((i % 100) as f32) / 10.0).collect(),
        )
        .expect("Failed to create matrix");

        let vector_data: Vec<f32> = (0..cols).map(|i| ((i % 50) as f32) / 5.0).collect();
        let vector = trueno::Vector::from_slice(&vector_data);

        // Warmup
        for _ in 0..3 {
            let _ = matrix.matvec(&vector).expect("Warmup matvec failed");
        }

        // Benchmark
        let iterations = if rows <= 500 { 100 } else { 20 };
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = matrix.matvec(&vector).expect("Benchmark matvec failed");
        }

        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;

        println!("Average time: {:.2} µs ({} iterations)", avg_time_us, iterations);
        println!(
            "Throughput: {:.2} GFLOPS\n",
            (2.0 * rows as f64 * cols as f64) / (avg_time_us * 1000.0)
        );
    }

    println!("Note: SIMD-optimized implementation using AVX2/SSE2 dot products");
    println!("Expected: 2-4× speedup vs naive scalar implementation");
}
