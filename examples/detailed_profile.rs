//! Detailed forward pass profiling
//! Measures time spent in each operation category to identify the 1.75x overhead source.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use std::time::Instant;

fn main() {
    // Use 16 threads (optimal from previous profiling)
    if let Err(e) = realizar::inference::configure_thread_pool(16) {
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
    println!("  num_heads: {}", config.num_heads);
    println!("  num_kv_heads: {}", config.num_kv_heads);

    // Encode a prompt
    let prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    let tokens = mapped.model.encode(prompt).unwrap();
    println!("\nPrompt tokens: {}", tokens.len());

    // Generation config for decode-only measurement
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 30,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![151645, 151643],
        trace: false,
        ..Default::default()
    };

    // Warmup
    println!("\nWarmup...");
    let _ = model.generate_with_cache(&tokens, &gen_config);

    // Profile multiple iterations
    let iters = 5;
    println!("\nProfiling {} generation runs...", iters);

    let start = Instant::now();
    let mut total_generated = 0;
    for _ in 0..iters {
        let output = model
            .generate_with_cache(&tokens, &gen_config)
            .expect("gen");
        total_generated += output.len() - tokens.len();
    }
    let total = start.elapsed();
    let per_token_us = total.as_micros() as f64 / total_generated as f64;
    let tok_s = total_generated as f64 / total.as_secs_f64();

    println!("\n=== Timing Results ===");
    println!("Total tokens: {}", total_generated);
    println!("Throughput: {:.1} tok/s", tok_s);
    println!("Per token: {:.1} µs", per_token_us);

    // Calculate theoretical breakdown
    let h = config.hidden_dim as f64;
    let i = config.intermediate_dim as f64;
    let l = config.num_layers as f64;
    let num_kv_heads = config.num_kv_heads as f64;
    let head_dim = h / config.num_heads as f64;
    let kv_dim = num_kv_heads * head_dim;

    // FLOPs per token decode (single token)
    // QKV projection (GQA: Q=hidden, K=kv_dim, V=kv_dim)
    let qkv_flops = (h * h + h * kv_dim + h * kv_dim) * 2.0 * l; // Q + K + V
                                                                 // Attention output projection
    let attn_out_flops = h * h * 2.0 * l;
    // FFN gate + up (SwiGLU)
    let ffn_gate_flops = h * i * 2.0 * l;
    let ffn_up_flops = h * i * 2.0 * l;
    // FFN down
    let ffn_down_flops = i * h * 2.0 * l;
    // LM head (once)
    let lm_head_flops = h * 151936.0 * 2.0; // vocab size

    let total_matmul_flops =
        qkv_flops + attn_out_flops + ffn_gate_flops + ffn_up_flops + ffn_down_flops + lm_head_flops;

    // Non-matmul FLOPs (per token)
    // RMSNorm: 2 per layer × (variance + div) × hidden_dim
    let rmsnorm_flops = 2.0 * l * 5.0 * h; // 5 ops per element (mean_sq, rsqrt, mul)
                                           // SiLU: intermediate_dim × layers
    let silu_flops = i * l * 4.0; // x * sigmoid(x) ≈ 4 ops
                                  // Attention softmax: O(seq_len) - negligible for decode
    let softmax_flops = 1.0 * h * l; // simplified
                                     // Q8K quantization: 3-4 times per layer
    let quant_flops = 3.0 * h * 2.0 * l; // quantize + dequant

    let total_non_matmul_flops = rmsnorm_flops + silu_flops + softmax_flops + quant_flops;

    // Theoretical time at kernel speed (123.4 GFLOP/s for matmul)
    let kernel_gflops = 123.4;
    let matmul_time_us = (total_matmul_flops / 1e9) / kernel_gflops * 1e6;

    // Non-matmul at ~50 GFLOP/s (memory-bound elementwise)
    let non_matmul_gflops = 50.0;
    let non_matmul_time_us = (total_non_matmul_flops / 1e9) / non_matmul_gflops * 1e6;

    let theoretical_us = matmul_time_us + non_matmul_time_us;

    println!("\n=== Theoretical Breakdown ===");
    println!("Matmul FLOPs:      {:.2}B", total_matmul_flops / 1e9);
    println!(
        "  QKV projection:  {:.2}B ({:.1}%)",
        qkv_flops / 1e9,
        100.0 * qkv_flops / total_matmul_flops
    );
    println!(
        "  Attn output:     {:.2}B ({:.1}%)",
        attn_out_flops / 1e9,
        100.0 * attn_out_flops / total_matmul_flops
    );
    println!(
        "  FFN gate:        {:.2}B ({:.1}%)",
        ffn_gate_flops / 1e9,
        100.0 * ffn_gate_flops / total_matmul_flops
    );
    println!(
        "  FFN up:          {:.2}B ({:.1}%)",
        ffn_up_flops / 1e9,
        100.0 * ffn_up_flops / total_matmul_flops
    );
    println!(
        "  FFN down:        {:.2}B ({:.1}%)",
        ffn_down_flops / 1e9,
        100.0 * ffn_down_flops / total_matmul_flops
    );
    println!(
        "  LM head:         {:.2}B ({:.1}%)",
        lm_head_flops / 1e9,
        100.0 * lm_head_flops / total_matmul_flops
    );
    println!();
    println!("Non-matmul FLOPs:  {:.2}M", total_non_matmul_flops / 1e6);
    println!("  RMSNorm:         {:.2}M", rmsnorm_flops / 1e6);
    println!("  SiLU:            {:.2}M", silu_flops / 1e6);
    println!("  Softmax:         {:.2}M", softmax_flops / 1e6);
    println!("  Q8K quant:       {:.2}M", quant_flops / 1e6);

    println!("\n=== Theoretical vs Actual ===");
    println!("Matmul theoretical:     {:>8.1} µs", matmul_time_us);
    println!("Non-matmul theoretical: {:>8.1} µs", non_matmul_time_us);
    println!("Total theoretical:      {:>8.1} µs", theoretical_us);
    println!("Actual:                 {:>8.1} µs", per_token_us);
    println!(
        "Overhead:               {:>8.2}x",
        per_token_us / theoretical_us
    );

    // Gap analysis
    let gap_us = per_token_us - theoretical_us;
    let gap_pct = 100.0 * gap_us / per_token_us;

    println!("\n=== Gap Analysis ===");
    println!("Gap: {:.1} µs ({:.1}% of actual)", gap_us, gap_pct);
    println!();
    println!("Possible sources of gap:");
    println!("  - Parallel region spawn overhead (rayon::join, par_iter)");
    println!("  - Memory allocation (Vec::with_capacity, extend)");
    println!("  - Cache effects (L3 thrashing, DRAM latency)");
    println!("  - Function call overhead");
    println!("  - Q8K quantization not at kernel speed");

    // Target analysis
    let target_tok_s = 42.0;
    let target_us_per_tok = 1_000_000.0 / target_tok_s;

    println!("\n=== Target Analysis ===");
    println!(
        "Current:    {:.1} tok/s ({:.1} µs/tok)",
        tok_s, per_token_us
    );
    println!(
        "Target:     {:.1} tok/s ({:.1} µs/tok)",
        target_tok_s, target_us_per_tok
    );
    println!("Gap factor: {:.2}x", per_token_us / target_us_per_tok);
    println!();

    // What efficiency would we need?
    let needed_gflops = (total_matmul_flops / 1e9) / (target_us_per_tok / 1e6);
    println!("To reach target:");
    println!("  Need {:.1} GFLOP/s effective throughput", needed_gflops);
    println!(
        "  Current: {:.1} GFLOP/s ({:.1}% of kernel)",
        (total_matmul_flops / 1e9) / (per_token_us / 1e6),
        100.0 * (total_matmul_flops / 1e9) / (per_token_us / 1e6) / kernel_gflops
    );
    println!(
        "  Target:  {:.1} GFLOP/s ({:.1}% of kernel)",
        needed_gflops,
        100.0 * needed_gflops / kernel_gflops
    );
}
