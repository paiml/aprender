#![allow(clippy::disallowed_methods)]
use std::time::Instant;
use trueno::Matrix;

fn main() {
    let size = 1024;
    println!("Creating {}×{} matrices...", size, size);

    let a =
        Matrix::from_vec(size, size, (0..size * size).map(|i| ((i % 100) as f32) / 10.0).collect())
            .expect("Failed to create matrix A");

    let b = Matrix::from_vec(
        size,
        size,
        (0..size * size).map(|i| (((i * 7) % 100) as f32) / 10.0).collect(),
    )
    .expect("Failed to create matrix B");

    // Warmup
    println!("Warmup...");
    for _ in 0..3 {
        let _ = a.matmul(&b).expect("Warmup matmul failed");
    }

    // Benchmark
    println!("Benchmarking matmul {}×{}×{}...", size, size, size);
    let iterations = 10;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = a.matmul(&b).expect("Benchmark matmul failed");
    }

    let elapsed = start.elapsed();
    let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("\n=== Results ===");
    println!("Total time: {:.2}ms ({} iterations)", elapsed.as_millis(), iterations);
    println!("Average time per matmul: {:.2}ms", avg_time_ms);

    #[cfg(feature = "parallel")]
    println!("Parallel feature: ENABLED");
    #[cfg(not(feature = "parallel"))]
    println!("Parallel feature: DISABLED");
}
