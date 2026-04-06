#![allow(clippy::disallowed_methods)]
use std::time::Instant;
use trueno::{Matrix, Vector};

fn main() {
    println!("=======================================================");
    println!("    Matrix-Vector Multiplication: Parallel Benchmark");
    println!("=======================================================\n");

    // Test sizes: only ≥4096 rows triggers parallel execution
    let test_configs = vec![
        (1024, 512, "1024×512 (sequential)"),
        (2048, 512, "2048×512 (sequential)"),
        (4096, 512, "4096×512 (parallel threshold)"),
        (8192, 512, "8192×512 (2× parallel threshold)"),
    ];

    for (rows, cols, desc) in test_configs {
        println!("=== {} ===", desc);

        // Create test data
        let matrix = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols).map(|i| ((i % 100) as f32) / 10.0).collect(),
        )
        .expect("Failed to create matrix");

        let vector =
            Vector::from_slice(&(0..cols).map(|i| ((i % 50) as f32) / 5.0).collect::<Vec<f32>>());

        // Warmup
        for _ in 0..3 {
            let _ = matrix.matvec(&vector).expect("Warmup matvec failed");
        }

        // Benchmark
        let iterations = if rows <= 2048 { 50 } else { 20 };
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = matrix.matvec(&vector).expect("Benchmark matvec failed");
        }

        let elapsed = start.elapsed();
        let avg_time_ms = elapsed.as_micros() as f64 / (iterations as f64 * 1000.0);
        let ops = 2.0 * rows as f64 * cols as f64; // Multiply-add for each element
        let gflops = ops / (avg_time_ms * 1e6);

        println!("  Rows: {}, Cols: {}", rows, cols);
        println!("  Average time: {:.3} ms ({} iterations)", avg_time_ms, iterations);
        println!("  Throughput: {:.2} GFLOPS", gflops);

        #[cfg(feature = "parallel")]
        {
            if rows >= 4096 {
                println!("  Execution: Parallel (Rayon) + SIMD");
            } else {
                println!("  Execution: Sequential + SIMD (below parallel threshold)");
            }
        }
        #[cfg(not(feature = "parallel"))]
        println!("  Execution: Sequential + SIMD");

        println!();
    }

    println!("=======================================================");
    println!("Build with:");
    println!("  cargo run --release --features parallel --example benchmark_matvec_parallel");
    println!("\nExpected Performance:");
    println!("  • Parallel: 2-3× speedup vs sequential for large matrices");
    println!("  • SIMD: 2-4× speedup from AVX2/SSE2 dot products");
    println!("  • Combined: 4-12× total speedup vs naive scalar");
    println!("=======================================================");
}
