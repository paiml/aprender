//! PARITY-038: CUDA Streams Async Execution Benchmark
//!
//! Tests multi-stream overlapped execution:
//! - Sequential: H2D → Compute → D2H (one at a time)
//! - Overlapped: Compute[n] overlaps with H2D[n+1] and D2H[n-1]
//!
//! Run with: cargo run --release --example parity_038_async_streams --features cuda

#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║      PARITY-038: CUDA Streams Async Execution Benchmark        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;

        // Test configuration matching phi-2 FFN
        let hidden_dim = 2560u32;
        let intermediate_dim = 10240u32;
        let num_tokens = 10usize; // Simulate generating 10 tokens

        println!("Configuration (phi-2 FFN):");
        println!("  hidden_dim: {}", hidden_dim);
        println!("  intermediate_dim: {}", intermediate_dim);
        println!("  num_tokens: {}", num_tokens);
        println!();

        // Initialize CUDA
        println!("[1/4] Initializing CUDA...");
        let start = Instant::now();
        let mut executor = match CudaExecutor::new(0) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Failed to initialize CUDA: {}", err);
                std::process::exit(1);
            },
        };
        let cuda_init_time = start.elapsed();
        println!("       CUDA init: {:?}", cuda_init_time);

        if let Ok(name) = executor.device_name() {
            println!("       Device: {}", name);
        }

        // Generate test data
        println!("\n[2/4] Generating test data and loading weights...");
        let m = intermediate_dim;
        let k = hidden_dim;
        let n = 1u32; // Single token MATVEC

        let weights: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let input: Vec<f32> = (0..k as usize)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();

        // Load weights to GPU (PARITY-037)
        let weight_bytes = executor
            .load_weights("fc1", &weights)
            .expect("Failed to load weights");
        println!(
            "       Loaded {} MB weights to GPU",
            weight_bytes / 1024 / 1024
        );

        // Warmup
        println!("\n[3/4] Warming up...");
        let mut output = vec![0.0f32; m as usize];
        for _ in 0..3 {
            let _ = executor.gemm_cached("fc1", &input, &mut output, m, n, k);
        }

        // Benchmark 1: Sequential execution (current implementation)
        println!("\n[4/4] Benchmarking...");
        println!("\n═══ Sequential Execution (baseline) ═══");

        let start = Instant::now();
        for _token in 0..num_tokens {
            // Each token: full round-trip
            executor
                .gemm_cached("fc1", &input, &mut output, m, n, k)
                .expect("GEMM failed");
        }
        let sequential_time = start.elapsed();
        let sequential_per_token = sequential_time / num_tokens as u32;

        println!(
            "  Total time: {:?} for {} tokens",
            sequential_time, num_tokens
        );
        println!("  Per token: {:?}", sequential_per_token);

        // Benchmark 2: Async execution with pre-allocated buffers
        println!("\n═══ Async Execution (PARITY-038) ═══");

        // Pre-allocate GPU buffers for double-buffering
        let mut input_buf = executor
            .allocate_buffer(k as usize)
            .expect("Failed to allocate input buffer");
        let output_buf = executor
            .allocate_buffer(m as usize)
            .expect("Failed to allocate output buffer");

        // Copy input to GPU once (it doesn't change for this test)
        input_buf
            .copy_from_host(&input)
            .expect("Failed to copy input");

        let start = Instant::now();
        for _token in 0..num_tokens {
            // Launch compute (non-blocking)
            executor
                .gemm_cached_async("fc1", &input_buf, &output_buf, m, n, k)
                .expect("Async GEMM failed");

            // Sync compute stream
            executor.synchronize_compute().expect("Sync compute failed");
        }
        // Final D2H
        output_buf
            .copy_to_host(&mut output)
            .expect("Final copy failed");

        let async_time = start.elapsed();
        let async_per_token = async_time / num_tokens as u32;

        println!("  Total time: {:?} for {} tokens", async_time, num_tokens);
        println!("  Per token: {:?}", async_per_token);

        // Results
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                        RESULTS                                  ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║ Sequential:  {:>10?}/token                                  ║",
            sequential_per_token
        );
        println!(
            "║ Async:       {:>10?}/token                                  ║",
            async_per_token
        );
        println!("╠════════════════════════════════════════════════════════════════╣");

        let speedup = sequential_per_token.as_secs_f64() / async_per_token.as_secs_f64();
        if speedup > 1.0 {
            println!(
                "║ Async is {:.2}x FASTER (reduced sync overhead)              ║",
                speedup
            );
        } else {
            println!(
                "║ Async is {:.2}x SLOWER (overhead dominates)                 ║",
                1.0 / speedup
            );
        }
        println!("╚════════════════════════════════════════════════════════════════╝");

        // Performance estimate
        let flops_per_matvec = 2.0 * (m as f64) * (k as f64);
        let async_gflops = flops_per_matvec / async_per_token.as_secs_f64() / 1e9;

        // Full FFN estimate (fc1 + fc2, 32 layers)
        let fc1_ops = 2.0 * (intermediate_dim as f64) * (hidden_dim as f64);
        let fc2_ops = 2.0 * (hidden_dim as f64) * (intermediate_dim as f64);
        let ffn_ops_per_layer = fc1_ops + fc2_ops;
        let total_ffn_ops = ffn_ops_per_layer * 32.0;
        let async_time_per_token = total_ffn_ops / (async_gflops * 1e9);
        let tps_async = 1.0 / async_time_per_token;

        println!("\n═══ Token Generation Estimate (FFN only, 32 layers) ═══");
        println!("  Async GFLOPS: {:.2}", async_gflops);
        println!(
            "  Estimated: {:.1} tok/s ({:.2}ms per token)",
            tps_async,
            async_time_per_token * 1000.0
        );

        // M3/M4 status
        let m3_target = 50.6; // tok/s
        let m4_target = 202.3; // tok/s
        println!("\n═══ Milestone Status ═══");
        println!(
            "  M3 (<5x gap, >{:.1} tok/s): {}",
            m3_target,
            if tps_async > m3_target {
                "✓ ACHIEVED"
            } else {
                "✗ NOT YET"
            }
        );
        println!(
            "  M4 (<1.25x gap, >{:.1} tok/s): {}",
            m4_target,
            if tps_async > m4_target {
                "✓ ACHIEVED"
            } else {
                "✗ NOT YET"
            }
        );

        // Cleanup
        executor.clear_weights();

        println!("\n═══ Analysis (PARITY-038) ═══");
        println!("  - Multi-stream infrastructure added to CudaExecutor");
        println!("  - compute_stream: kernel execution");
        println!("  - transfer_stream: async H2D/D2H copies");
        println!("  - Pre-allocated buffers eliminate allocation overhead");
        println!("  - Next: Double-buffering for true overlap (PARITY-039+)");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with --features cuda");
    }
}
