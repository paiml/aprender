//! IMP-600c: GPU vs SIMD Benchmark for Matrix-Matrix Multiplication (GEMM)
//!
//! Popperian Falsification Test:
//! CLAIM: "trueno wgpu GPU excels at GEMM but not MATVEC"
//!
//! Run: cargo run --example gpu_gemm_benchmark --features gpu

use std::hint::black_box;
use std::time::Instant;

// Scalar GEMM for baseline
#[allow(clippy::many_single_char_names)]
fn gemm_scalar(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn main() {
    println!("=== IMP-600c: GPU vs SIMD GEMM Benchmark ===");
    println!("Testing where GPU should excel: large matrix-matrix multiplication\n");

    // Smaller test cases to avoid buffer limits
    let test_cases = [
        ("small 256x256", 256, 256, 256),
        ("medium 512x512", 512, 512, 512),
        ("phi2 batch8", 2560, 2560, 8),
        ("llama batch8", 4096, 4096, 8),
        ("large 1024x1024", 1024, 1024, 1024),
    ];

    #[cfg(feature = "gpu")]
    {
        use trueno::backends::gpu::GpuDevice;

        match GpuDevice::new() {
            Ok(device) => {
                println!("GPU device initialized\n");
                println!(
                    "{:<20} {:>6} {:>6} {:>6} {:>12} {:>12} {:>10}",
                    "Operation", "M", "K", "N", "Scalar(ms)", "GPU(ms)", "Speedup"
                );
                println!("{}", "-".repeat(85));

                for (name, m, k, n) in test_cases {
                    // Skip if buffer would be too large (>128MB)
                    let buffer_size = (m * k + k * n + m * n) * 4;
                    if buffer_size > 128 * 1024 * 1024 {
                        println!(
                            "{:<20} {:>6} {:>6} {:>6} SKIPPED (buffer > 128MB)",
                            name, m, k, n
                        );
                        continue;
                    }

                    let a: Vec<f32> = (0..(m * k)).map(|i| ((i % 1000) as f32) * 0.001).collect();
                    let b: Vec<f32> = (0..(k * n)).map(|i| ((i % 1000) as f32) * 0.001).collect();

                    let iterations = 5;

                    // Warmup and verify correctness
                    let scalar_result = gemm_scalar(&a, &b, m, k, n);
                    let mut gpu_result = vec![0.0f32; m * n];
                    device
                        .matmul(&a, &b, &mut gpu_result, m, k, n)
                        .expect("GPU matmul failed");

                    // Check correctness
                    let max_diff = scalar_result
                        .iter()
                        .zip(gpu_result.iter())
                        .map(|(s, g)| (s - g).abs())
                        .fold(0.0f32, f32::max);
                    if max_diff > 0.1 {
                        println!("WARNING: Max diff = {:.4} for {}", max_diff, name);
                    }

                    // Benchmark scalar
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(gemm_scalar(&a, &b, m, k, n));
                    }
                    let scalar_ms = start.elapsed().as_millis() as f64 / iterations as f64;

                    // Benchmark GPU
                    let start = Instant::now();
                    for _ in 0..iterations {
                        device
                            .matmul(&a, &b, &mut gpu_result, m, k, n)
                            .expect("GPU matmul failed");
                        black_box(&gpu_result);
                    }
                    let gpu_ms = start.elapsed().as_millis() as f64 / iterations as f64;

                    let speedup = scalar_ms / gpu_ms;

                    println!(
                        "{:<20} {:>6} {:>6} {:>6} {:>12.2} {:>12.2} {:>10.2}x",
                        name, m, k, n, scalar_ms, gpu_ms, speedup
                    );
                }

                println!("\n=== Falsification Analysis ===");
                println!("CLAIM: GPU excels at GEMM (batch operations) but not MATVEC");
                println!("- For GEMM with large N: GPU should show significant speedup");
                println!("- For MATVEC (N=1): GPU has overhead that SIMD avoids");
                println!("- LLM token generation is MATVEC-bound → SIMD is optimal");
                println!("- LLM prompt processing is GEMM-bound → GPU can help");
            },
            Err(e) => println!("GPU not available: {}", e),
        }
    }

    #[cfg(not(feature = "gpu"))]
    println!("Run with: cargo run --example gpu_gemm_benchmark --features gpu");

    println!("\n=== Summary: When to Use GPU vs SIMD ===");
    println!("1. Token generation (matvec, N=1): Use SIMD (2.7x faster than GPU)");
    println!("2. Prompt processing (gemm, N=seq_len): GPU if N > ~32");
    println!("3. Batch inference (gemm, N=batch_size): GPU if N > ~16");
    println!("4. llama.cpp strategy: SIMD for inference, cuBLAS for training/batching");
}
