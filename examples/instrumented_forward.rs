//! Instrumented forward pass to find the 3.6x overhead source
//!
//! Measures time spent in each operation category.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use std::time::Instant;

fn main() {
    // Set thread count to physical cores (not hyperthreads)
    // This reduces thread synchronization overhead
    let num_physical_cores = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12); // Default to 12 physical cores

    if let Err(e) = realizar::inference::configure_thread_pool(num_physical_cores) {
        eprintln!("Note: Thread pool already configured: {e}");
    }
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    println!("Loading model: {model_path}");
    let mapped = MappedGGUFModel::from_path(&model_path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");

    let config = model.config();
    println!("\nModel config:");
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  intermediate_dim: {}", config.intermediate_dim);
    println!("  num_layers: {}", config.num_layers);

    // Encode a prompt
    let prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    let tokens = mapped.model.encode(prompt).unwrap();
    println!("Prompt tokens: {}", tokens.len());

    // Generation config
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 50,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![151645, 151643],
        trace: false,
        ..Default::default()
    };

    // Warmup
    println!("\nWarmup...");
    let _ = model.generate_with_cache(&tokens, &gen_config);

    // Measure individual operations by running the model multiple times
    // and using perf counters
    let iters = 3;
    println!("\nRunning {} iterations...", iters);

    let overall_start = Instant::now();
    let mut total_tokens = 0;
    for _ in 0..iters {
        let output = model
            .generate_with_cache(&tokens, &gen_config)
            .expect("gen");
        total_tokens += output.len() - tokens.len();
    }
    let overall_elapsed = overall_start.elapsed();

    let tok_s = total_tokens as f64 / overall_elapsed.as_secs_f64();
    let per_token_us = overall_elapsed.as_micros() as f64 / total_tokens as f64;

    println!("\n=== Overall Performance ===");
    println!("Tokens generated: {}", total_tokens);
    println!("Throughput: {:.1} tok/s", tok_s);
    println!("Per token: {:.1} µs", per_token_us);

    // Calculate FLOPs breakdown
    let h = config.hidden_dim as f64;
    let i = config.intermediate_dim as f64;
    let l = config.num_layers as f64;

    // Per-token FLOPs by operation category
    let qkv_flops = 3.0 * h * h * 2.0 * l; // QKV projection (simplified, ignores GQA)
    let _attn_flops = 0.0; // ~O(seq) for decode, negligible
    let proj_flops = h * h * 2.0 * l; // Attention output projection
    let ffn_up_gate_flops = 2.0 * h * i * 2.0 * l; // FFN up + gate
    let ffn_down_flops = i * h * 2.0 * l; // FFN down
    let _total_flops = qkv_flops + proj_flops + ffn_up_gate_flops + ffn_down_flops;

    // Theoretical time at 123.4 GFLOP/s kernel throughput
    let kernel_gflops = 123.4;
    let qkv_us = (qkv_flops / 1e9) / kernel_gflops * 1e6;
    let proj_us = (proj_flops / 1e9) / kernel_gflops * 1e6;
    let ffn_up_gate_us = (ffn_up_gate_flops / 1e9) / kernel_gflops * 1e6;
    let ffn_down_us = (ffn_down_flops / 1e9) / kernel_gflops * 1e6;
    let total_theoretical_us = qkv_us + proj_us + ffn_up_gate_us + ffn_down_us;

    println!("\n=== Theoretical Breakdown (at kernel speed) ===");
    println!(
        "QKV projection:     {:>8.1} µs  ({:.1}%)",
        qkv_us,
        100.0 * qkv_us / total_theoretical_us
    );
    println!(
        "Attn output proj:   {:>8.1} µs  ({:.1}%)",
        proj_us,
        100.0 * proj_us / total_theoretical_us
    );
    println!(
        "FFN up+gate:        {:>8.1} µs  ({:.1}%)",
        ffn_up_gate_us,
        100.0 * ffn_up_gate_us / total_theoretical_us
    );
    println!(
        "FFN down:           {:>8.1} µs  ({:.1}%)",
        ffn_down_us,
        100.0 * ffn_down_us / total_theoretical_us
    );
    println!("Total theoretical:  {:>8.1} µs", total_theoretical_us);
    println!("Actual:             {:>8.1} µs", per_token_us);
    println!(
        "Overhead:           {:>8.1}x",
        per_token_us / total_theoretical_us
    );

    // Estimate where overhead comes from
    // - RMSNorm: 28 layers × 2 norms × (5 FLOPs/elem × 1536) = ~430K FLOPs
    // - Attention scores: O(seq_len) per head per layer - negligible for short seq
    // - SiLU: 28 layers × 8960 = ~250K FLOPs
    // - Memory alloc/dealloc: ?
    // - Thread sync: ?
    let norm_flops = 2.0 * l * 5.0 * h; // 5 FLOPs per element for RMSNorm
    let silu_flops = l * i; // SiLU: x * sigmoid(x)
    let overhead_flops = norm_flops + silu_flops;
    let _overhead_us = (overhead_flops / 1e9) / kernel_gflops * 1e6;

    println!("\n=== Non-Matmul Operations ===");
    println!(
        "RMSNorm FLOPs:      {:>8.1}K ({:.1} µs at kernel speed)",
        norm_flops / 1e3,
        (norm_flops / 1e9) / kernel_gflops * 1e6
    );
    println!(
        "SiLU FLOPs:         {:>8.1}K ({:.1} µs at kernel speed)",
        silu_flops / 1e3,
        (silu_flops / 1e9) / kernel_gflops * 1e6
    );

    // Gap analysis
    let gap_us = per_token_us - total_theoretical_us;
    println!("\n=== Gap Analysis ===");
    println!("Matmul theoretical: {:>8.1} µs", total_theoretical_us);
    println!("Actual:             {:>8.1} µs", per_token_us);
    println!(
        "Gap:                {:>8.1} µs ({:.1}%)",
        gap_us,
        100.0 * gap_us / per_token_us
    );
    println!();
    println!("If gap is mostly in:");
    println!("  - RMSNorm/SiLU:   Scalar ops need SIMD optimization");
    println!("  - Thread sync:    Rayon overhead, consider batching");
    println!("  - Memory alloc:   Hot path allocations, use scratch buffers");
    println!("  - Cache misses:   Working set > L3, need better tiling");

    // Rayon thread info
    println!("\n=== Rayon Configuration ===");
    println!("Configured threads: {}", num_physical_cores);
    println!("Actual thread pool: {}", rayon::current_num_threads());
}
