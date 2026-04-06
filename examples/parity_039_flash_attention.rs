//! PARITY-039: FlashAttention Performance Benchmark
//!
//! Tests FlashAttention fused kernel vs standard attention:
//! - Memory: O(N) vs O(N²)
//! - Performance for various sequence lengths
//!
//! Run with: cargo run --release --example parity_039_flash_attention --features cuda

#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║      PARITY-039: FlashAttention Performance Benchmark          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;

        // Test configurations
        let configs = [
            (64u32, 64u32, "Small (64x64)"),
            (128, 64, "Medium (128x64)"),
            (256, 64, "phi-2 like (256x64)"),
            (512, 64, "Large (512x64)"),
        ];
        let iterations = 10;

        // Initialize CUDA
        println!("[1/3] Initializing CUDA...");
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

        // Memory comparison
        println!("\n[2/3] Memory Analysis (O(N) vs O(N²))...");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║ seq_len │ Standard (O(N²)) │ Flash (O(N)) │ Savings              ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        for (seq_len, head_dim, _) in &configs {
            let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(*seq_len, *head_dim);
            let savings = naive as f64 / flash as f64;
            println!(
                "║ {:>7} │ {:>12} KB │ {:>8} KB │ {:>6.1}x savings       ║",
                seq_len,
                naive / 1024,
                flash / 1024,
                savings
            );
        }
        println!("╚══════════════════════════════════════════════════════════════════╝");

        // Performance benchmark
        println!("\n[3/3] Performance Benchmark...");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║ Configuration     │ Time/iter │ GFLOPS │ Status                 ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");

        let mut total_gflops = 0.0f64;
        let mut total_configs = 0;

        for (seq_len, head_dim, name) in &configs {
            let size = (*seq_len * *head_dim) as usize;

            // Generate test data
            let q: Vec<f32> = (0..size).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
            let k: Vec<f32> = (0..size).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
            let v: Vec<f32> = (0..size).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
            let mut output = vec![0.0f32; size];

            let scale = 1.0 / (*head_dim as f32).sqrt();

            // Warmup
            for _ in 0..3 {
                let _ = executor.flash_attention(
                    &q,
                    &k,
                    &v,
                    &mut output,
                    *seq_len,
                    *head_dim,
                    scale,
                    false,
                );
            }

            // Benchmark
            let start = Instant::now();
            for _ in 0..iterations {
                executor
                    .flash_attention(&q, &k, &v, &mut output, *seq_len, *head_dim, scale, false)
                    .expect("FlashAttention failed");
            }
            let total_time = start.elapsed();
            let per_iter = total_time / iterations as u32;

            // FLOPs: attention is ~4 * N * N * d (Q@K, softmax, @V)
            let flops = 4.0 * (*seq_len as f64) * (*seq_len as f64) * (*head_dim as f64);
            let gflops = flops / per_iter.as_secs_f64() / 1e9;
            total_gflops += gflops;
            total_configs += 1;

            let status = if per_iter.as_micros() < 1000 {
                "✓ <1ms"
            } else if per_iter.as_micros() < 5000 {
                "⚠ <5ms"
            } else {
                "✗ slow"
            };

            println!(
                "║ {:<17} │ {:>9?} │ {:>6.1} │ {}                 ║",
                name, per_iter, gflops, status
            );
        }
        println!("╚══════════════════════════════════════════════════════════════════╝");

        let avg_gflops = total_gflops / total_configs as f64;

        // Token generation estimate
        // phi-2: 32 layers, 32 heads, head_dim=80
        let layers = 32;
        let num_heads = 32;
        let head_dim = 80u32;
        let seq_len = 128u32; // Typical context length during generation

        // FLOPs per attention per layer: 4 * seq_len * seq_len * head_dim * num_heads
        let attn_flops_per_layer =
            4.0 * (seq_len as f64) * (seq_len as f64) * (head_dim as f64) * (num_heads as f64);
        let total_attn_flops = attn_flops_per_layer * layers as f64;

        // Time estimate based on average GFLOPS
        let attn_time_per_token = total_attn_flops / (avg_gflops * 1e9);

        // Compare to PARITY-038 FFN time (6.53ms from benchmark)
        let ffn_time_per_token = 0.00653; // seconds
        let total_time_per_token = attn_time_per_token + ffn_time_per_token;
        let tps = 1.0 / total_time_per_token;

        println!("\n═══ Token Generation Estimate (phi-2 like, 32 layers) ═══");
        println!("  Attention GFLOPS (avg): {:.1}", avg_gflops);
        println!(
            "  Attention time/token: {:.2}ms",
            attn_time_per_token * 1000.0
        );
        println!(
            "  FFN time/token (PARITY-038): {:.2}ms",
            ffn_time_per_token * 1000.0
        );
        println!("  Total time/token: {:.2}ms", total_time_per_token * 1000.0);
        println!("  Estimated: {:.1} tok/s", tps);

        // M3/M4 status
        let m3_target = 50.6;
        let m4_target = 202.3;
        println!("\n═══ Milestone Status ═══");
        println!(
            "  M3 (<5x gap, >{:.1} tok/s): {}",
            m3_target,
            if tps > m3_target {
                "✓ ACHIEVED"
            } else {
                "✗ NOT YET"
            }
        );
        println!(
            "  M4 (<1.25x gap, >{:.1} tok/s): {}",
            m4_target,
            if tps > m4_target {
                "✓ ACHIEVED"
            } else {
                "✗ NOT YET"
            }
        );

        println!("\n═══ Analysis (PARITY-039) ═══");
        println!("  - FlashAttention kernel operational");
        println!("  - O(N) memory vs O(N²) for standard attention");
        println!("  - Enables longer context without OOM");
        println!("  - Next: FP16 Tensor Cores for higher throughput");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with --features cuda");
    }
}
