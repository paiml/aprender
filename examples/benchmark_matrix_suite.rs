#![allow(clippy::disallowed_methods)]
use std::time::Instant;
use trueno::{Matrix, Vector};

/// Benchmark Q4K GEMV at inference-relevant dimensions.
/// Output format matches cgp quant profiler parsing:
///   "Q4K GEMV (MxK)...     X.XX us  (Y.Y GFLOPS)"
fn bench_q4k_gemv() {
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Q4K Quantized GEMV Benchmark");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Qwen2.5-1.5B dimensions: embed=1536, ffn=8960
    let configs: &[(usize, usize, &str)] = &[
        (1536, 8960, "ffn_up/gate"),
        (8960, 1536, "ffn_down"),
        (1536, 1536, "attn_qkv"),
        (4096, 4096, "generic_4K"),
    ];

    for &(out_dim, in_dim, label) in configs {
        // Validate in_dim is multiple of 256 (super-block requirement)
        if in_dim % 256 != 0 {
            println!("  Skipping {label}: in_dim={in_dim} not multiple of 256");
            continue;
        }

        let num_sb = (out_dim * in_dim) / 256;
        let q4k_bytes = num_sb * 144;

        // Generate test data (pattern doesn't matter for timing)
        let q4k_data: Vec<u8> = (0..q4k_bytes).map(|i| (i % 256) as u8).collect();
        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

        // Warmup
        for _ in 0..3 {
            let _ =
                trueno::backends::q4k::matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);
        }

        // Timed runs — enough iterations for stable measurement
        let iterations = if out_dim * in_dim > 5_000_000 { 50 } else { 200 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ =
                trueno::backends::q4k::matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);
        }
        let elapsed = start.elapsed();
        let per_us = elapsed.as_secs_f64() * 1e6 / iterations as f64;
        let total_ops = 2.0 * out_dim as f64 * in_dim as f64;
        let gflops = total_ops / (per_us / 1e6) / 1e9;
        let bw_gbps = q4k_bytes as f64 / (per_us / 1e6) / 1e9;

        println!(
            "  Q4K GEMV ({out_dim}x{in_dim}, {label})...{:>10.2} us  ({gflops:.2} GFLOPS, {bw_gbps:.1} GB/s)",
            per_us
        );
    }
}

fn create_test_matrix(rows: usize, cols: usize, seed_multiplier: usize) -> Matrix<f32> {
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols).map(|i| (((i * seed_multiplier) % 100) as f32) / 10.0).collect(),
    )
    .expect("Failed to create matrix")
}

fn iterations_for_size(rows: usize, threshold: usize, fast: usize, slow: usize) -> usize {
    if rows <= threshold {
        fast
    } else {
        slow
    }
}

fn bench_matmul(matrix_a: &Matrix<f32>, matrix_b: &Matrix<f32>, rows: usize, cols: usize) {
    if rows > 1024 {
        return;
    }
    print!("  Matrix Multiplication ({}x{}x{})... ", rows, cols, cols);

    // Warmup
    for _ in 0..2 {
        let _ = matrix_a.matmul(matrix_b).expect("Warmup matmul failed");
    }

    let iterations = iterations_for_size(rows, 512, 10, 5);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = matrix_a.matmul(matrix_b).expect("Benchmark matmul failed");
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let gflops = (2.0 * rows as f64 * cols as f64 * cols as f64) / (avg_ms * 1e6);

    println!("{:>8.2} ms  ({:.2} GFLOPS)", avg_ms, gflops);
}

fn bench_matvec(matrix_a: &Matrix<f32>, vector: &Vector<f32>, rows: usize, cols: usize) {
    print!("  Matrix-Vector ({}x{} x {})... ", rows, cols, cols);

    // Warmup
    for _ in 0..3 {
        let _ = matrix_a.matvec(vector).expect("Warmup matvec failed");
    }

    let iterations = iterations_for_size(rows, 512, 100, 20);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = matrix_a.matvec(vector).expect("Benchmark matvec failed");
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let gflops = (2.0 * rows as f64 * cols as f64) / (avg_us * 1000.0);

    println!("{:>8.2} us  ({:.2} GFLOPS)", avg_us, gflops);
}

fn bench_vecmat(matrix_a: &Matrix<f32>, vector: &Vector<f32>, rows: usize, cols: usize) {
    print!("  Vector-Matrix ({} x {}x{})... ", rows, rows, cols);

    // Warmup
    for _ in 0..3 {
        let _ = Matrix::vecmat(vector, matrix_a).expect("Warmup vecmat failed");
    }

    let iterations = iterations_for_size(rows, 512, 100, 20);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Matrix::vecmat(vector, matrix_a).expect("Benchmark vecmat failed");
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let gflops = (2.0 * rows as f64 * cols as f64) / (avg_us * 1000.0);

    println!("{:>8.2} us  ({:.2} GFLOPS)", avg_us, gflops);
}

fn bench_transpose(matrix_a: &Matrix<f32>, rows: usize, cols: usize) {
    print!("  Transpose ({}x{})... ", rows, cols);

    // Warmup
    for _ in 0..3 {
        let _ = matrix_a.transpose();
    }

    let iterations = iterations_for_size(rows, 512, 100, 20);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = matrix_a.transpose();
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let bandwidth_gb = (rows as f64 * cols as f64 * 8.0) / (avg_us * 1000.0); // Read + Write

    println!("{:>8.2} us  ({:.2} GB/s bandwidth)", avg_us, bandwidth_gb);
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Trueno Matrix Operations Performance Benchmark Suite        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test sizes
    let sizes = vec![(256, 256), (512, 512), (1024, 1024), (2048, 2048)];

    for (rows, cols) in sizes {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  Matrix Size: {}x{}", rows, cols);
        println!("═══════════════════════════════════════════════════════════════════\n");

        let matrix_a = create_test_matrix(rows, cols, 1);
        let matrix_b = create_test_matrix(rows, cols, 7);

        let vector_data: Vec<f32> = (0..cols).map(|i| ((i % 50) as f32) / 5.0).collect();
        let vector = Vector::from_slice(&vector_data);

        bench_matmul(&matrix_a, &matrix_b, rows, cols);
        bench_matvec(&matrix_a, &vector, rows, cols);
        bench_vecmat(&matrix_a, &vector, rows, cols);
        bench_transpose(&matrix_a, rows, cols);
    }

    // Q4K GEMV benchmark — Qwen2.5-1.5B FFN dimensions
    bench_q4k_gemv();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Optimizations Applied:");
    println!("  * Matrix Multiplication: SIMD + 3-level cache blocking + parallel");
    println!("  * Matrix-Vector: SIMD dot products (AVX2/SSE2)");
    println!("  * Vector-Matrix: SIMD row accumulation (cache-friendly)");
    println!("  * Transpose: Cache-blocked (64x64 blocks)");
    println!("  * Q4K GEMV: Fused dequant+dot (AVX2), parallel for large layers");
    println!("═══════════════════════════════════════════════════════════════════\n");

    #[cfg(feature = "parallel")]
    println!("  [Parallel feature: ENABLED for matrices >=1024x1024]\n");
    #[cfg(not(feature = "parallel"))]
    println!("  [Parallel feature: DISABLED]\n");
}
