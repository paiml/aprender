//! Layer-by-layer forward pass profiling
//! Identifies where the 2.6x overhead is hiding

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use std::time::Instant;

fn main() {
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
    println!("  head_dim: {}", config.hidden_dim / config.num_heads);

    // Encode a prompt
    let prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    let tokens = mapped.model.encode(prompt).unwrap();
    println!("\nPrompt tokens: {}", tokens.len());

    // Generation config
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![151645, 151643],
        trace: false,
        ..Default::default()
    };

    // Warmup with actual generation
    println!("\nWarmup...");
    let _ = model.generate_with_cache(&tokens, &gen_config);

    // Profile token generation (uses optimized Q8K path)
    let iters = 5;
    println!(
        "\nProfiling {} generation runs ({} tokens each)...",
        iters, gen_config.max_tokens
    );

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

    println!("\n=== Generation Timing ===");
    println!("Total time: {} ms", total.as_millis());
    println!("Tokens generated: {}", total_generated);
    println!(
        "Per token: {:.1} µs ({:.2} ms)",
        per_token_us,
        per_token_us / 1000.0
    );
    println!("Throughput: {:.1} tok/s", tok_s);

    // Calculate theoretical matmul time
    let h = config.hidden_dim as f64;
    let i = config.intermediate_dim as f64;
    let l = config.num_layers as f64;

    // FLOPs per token (single token decode):
    // QKV: 3 * h * h * 2 (but with GQA, K/V are smaller)
    // Attn out: h * h * 2
    // FFN gate: h * i * 2
    // FFN up: h * i * 2
    // FFN down: i * h * 2
    let qkv_flops = 3.0 * h * h * 2.0; // simplified - actual is GQA
    let attn_out_flops = h * h * 2.0;
    let ffn_gate_flops = h * i * 2.0;
    let ffn_up_flops = h * i * 2.0;
    let ffn_down_flops = i * h * 2.0;
    let layer_flops = qkv_flops + attn_out_flops + ffn_gate_flops + ffn_down_flops + ffn_up_flops;
    let total_flops = layer_flops * l;

    // Measured GFLOPS from micro_profile
    let matmul_gflops = 123.4; // Q4K×Q8K from micro_profile
    let theoretical_time_us = (total_flops / 1e9) / matmul_gflops * 1e6;

    let achieved_gflops = (total_flops / 1e9) / (per_token_us / 1e6);

    println!("\n=== Theoretical vs Actual ===");
    println!("FLOPs per token: {:.2}B", total_flops / 1e9);
    println!("Theoretical matmul time: {:.1} µs", theoretical_time_us);
    println!("Actual per token: {:.1} µs", per_token_us);
    println!(
        "Overhead factor: {:.2}x",
        per_token_us / theoretical_time_us
    );
    println!(
        "Achieved: {:.1} GFLOP/s (of {:.1} kernel)",
        achieved_gflops, matmul_gflops
    );
    println!(
        "Efficiency: {:.1}%",
        100.0 * achieved_gflops / matmul_gflops
    );

    // Memory bandwidth analysis
    // Weight bytes per token (Q4K = 4.5 bits/weight)
    let weights_per_layer = 4 * config.hidden_dim * config.hidden_dim // QKV + out
        + 3 * config.hidden_dim * config.intermediate_dim; // gate, up, down
    let total_weights = weights_per_layer * config.num_layers;
    let weight_bytes = total_weights as f64 * 4.5 / 8.0; // Q4K_M bits per weight

    let mem_bandwidth_achieved = weight_bytes / (per_token_us / 1e6) / 1e9;

    println!("\n=== Memory Bandwidth ===");
    println!("Weight bytes per token: {:.1} MB", weight_bytes / 1e6);
    println!(
        "Memory bandwidth achieved: {:.1} GB/s",
        mem_bandwidth_achieved
    );
    println!("DDR5-4800 4-channel peak: ~154 GB/s");
    println!(
        "Bandwidth efficiency: {:.1}%",
        100.0 * mem_bandwidth_achieved / 154.0
    );

    // Target analysis
    let target_tok_s = 42.0;
    let target_us_per_tok = 1_000_000.0 / target_tok_s;
    println!("\n=== Target Analysis ===");
    println!("Current: {:.1} tok/s", tok_s);
    println!("Target: {:.1} tok/s (2x Ollama)", target_tok_s);
    println!("Gap: {:.1}x slower", target_tok_s / tok_s);
    println!(
        "Need to reduce per-token from {:.1} µs to {:.1} µs",
        per_token_us, target_us_per_tok
    );
}
