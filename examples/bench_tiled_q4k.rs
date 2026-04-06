//! Benchmark tiled Q4K CUDA kernel performance
//!
//! Tests the TiledQ4KGemvKernel with shared memory caching
//! Run: cargo run --release --features cuda --example bench_tiled_q4k

#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;
use std::time::Instant;

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !CudaExecutor::is_available() {
            println!("CUDA not available");
            return;
        }

        let mut executor = CudaExecutor::new(0).expect("Failed to create CUDA executor");
        println!("GPU: {}", executor.device_name().unwrap_or_default());
        println!();

        // Test dimensions matching Qwen2.5-1.5B attention projection
        // K=1536 (hidden_dim), N=1536 (hidden_dim)
        // 1536 % 256 = 0, so tiled kernel should be used
        let test_cases = [
            (1536usize, 1536usize, "1.5B attention Q/K/V projection"),
            (1536, 8960, "1.5B FFN up projection"),
            (4096, 4096, "7B attention projection"),
        ];

        println!("=== Q4K GEMV Kernel Benchmark ===\n");

        for (k, n, desc) in test_cases {
            let use_tiled = k % 256 == 0 && k <= 10240;
            println!("--- {} ---", desc);
            println!("  K={}, N={}, Tiled={}", k, n, use_tiled);

            // Create Q4K weight data
            // Q4K format: 144 bytes per 256-element super-block
            let num_superblocks = (k * n) / 256;
            let weight_size = num_superblocks * 144;
            let weights = vec![0x55u8; weight_size];

            // Create input activations
            let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001).sin()).collect();
            let mut output = vec![0.0f32; n];

            // Warmup
            for _ in 0..5 {
                executor
                    .q4k_gemv(&weights, &input, &mut output, n as u32, k as u32)
                    .expect("warmup failed");
            }
            executor.synchronize().expect("sync");

            // Benchmark
            let iterations = 100;
            let start = Instant::now();
            for _ in 0..iterations {
                executor
                    .q4k_gemv(&weights, &input, &mut output, n as u32, k as u32)
                    .expect("benchmark failed");
            }
            executor.synchronize().expect("sync");
            let elapsed = start.elapsed();

            let ms_per_op = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
            let ops_per_sec = 1000.0 / ms_per_op;

            // For transformer inference:
            // - 6 GEMVs per layer (Q, K, V, O projections + up/down projections)
            // - 28 layers for 1.5B model
            let gemvs_per_token = 28 * 6;
            let est_tok_s = ops_per_sec / gemvs_per_token as f64;

            println!("  Time: {:.3} ms/op ({:.1} ops/s)", ms_per_op, ops_per_sec);
            println!("  Est tok/s (28 layers): {:.1}", est_tok_s);

            // Calculate memory bandwidth utilization
            // Q4K: 144 bytes per 256 elements = 0.5625 bytes/element
            // Read: weights (K*N * 0.5625) + input (K * 4) bytes
            // Write: output (N * 4) bytes
            let weight_bytes = (k as f64 * n as f64) * 0.5625;
            let input_bytes = k as f64 * 4.0;
            let output_bytes = n as f64 * 4.0;
            let total_bytes = weight_bytes + input_bytes + output_bytes;
            let bandwidth_gb_s = total_bytes / (ms_per_op * 1e6);
            println!(
                "  Memory bandwidth: {:.1} GB/s (RTX 4090 peak: ~1000 GB/s)",
                bandwidth_gb_s
            );
            println!();
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled");
    }
}
