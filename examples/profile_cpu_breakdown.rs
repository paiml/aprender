//! CPU inference timing breakdown
//! Identifies which operations consume the most time in forward pass

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

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let num_layers = model.config().num_layers;

    println!("Model config:");
    println!("  hidden_dim: {hidden_dim}");
    println!("  intermediate_dim: {intermediate_dim}");
    println!("  num_layers: {num_layers}");
    println!();

    // Generate tokens and measure time
    let prompt =
        "<|im_start|>user\nWrite a hello world program.<|im_end|>\n<|im_start|>assistant\n";
    let tokens = mapped.model.encode(prompt).unwrap();
    println!("Prompt: {} tokens", tokens.len());

    let config = QuantizedGenerateConfig {
        max_tokens: 50,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![151645, 151643],
        trace: false,
        ..Default::default()
    };

    // Warmup
    println!("Warming up...");
    let _ = model.generate_with_cache(&tokens, &config);

    // Timed run
    println!("Generating...\n");
    let start = Instant::now();
    let output = model.generate_with_cache(&tokens, &config).expect("gen");
    let total_time = start.elapsed();

    let new_tokens = output.len() - tokens.len();
    let per_token_ms = total_time.as_millis() as f64 / new_tokens as f64;
    let tok_s = new_tokens as f64 / total_time.as_secs_f64();

    println!("=== Results ===");
    println!("Generated: {} tokens", new_tokens);
    println!("Total time: {:.1}ms", total_time.as_millis());
    println!("Per token: {:.1}ms", per_token_ms);
    println!("Throughput: {:.1} tok/s", tok_s);
    println!();

    // Theoretical analysis
    // Per-token FLOPs estimate:
    // - QKV projection: 3 * hidden_dim * hidden_dim * 2
    // - Attention output: hidden_dim * hidden_dim * 2
    // - FFN gate/up: 2 * hidden_dim * intermediate_dim * 2
    // - FFN down: intermediate_dim * hidden_dim * 2
    // - Per layer: ~10 * hidden_dim^2 + 6 * hidden_dim * intermediate_dim
    let flops_per_layer =
        10.0 * (hidden_dim as f64).powi(2) + 6.0 * hidden_dim as f64 * intermediate_dim as f64;
    let flops_per_token = flops_per_layer * num_layers as f64;
    let gflops_per_token = flops_per_token / 1e9;

    let achieved_gflops = gflops_per_token / (per_token_ms / 1000.0);

    println!("=== Compute Analysis ===");
    println!("FLOPs per token: {:.2}B", gflops_per_token);
    println!("Achieved: {:.2} GFLOP/s", achieved_gflops);
    println!();

    // Memory analysis
    // Weight bytes per token (Q4K = 4.5 bits/weight)
    let weights_per_layer = 4 * hidden_dim * hidden_dim // QKV + out
        + 3 * hidden_dim * intermediate_dim; // gate, up, down
    let total_weights = weights_per_layer * num_layers;
    let weight_bytes = (total_weights as f64 * 4.5 / 8.0) as usize; // Q4K_M

    // Arithmetic intensity = FLOPs / bytes
    let ai = gflops_per_token * 1e9 / weight_bytes as f64;

    println!("=== Memory Analysis ===");
    println!("Weight bytes per token: {:.1}MB", weight_bytes as f64 / 1e6);
    println!("Arithmetic intensity: {:.2} FLOP/byte", ai);
    println!();

    // Roofline comparison
    // DDR4-3200: ~25 GB/s single channel, ~50 GB/s dual channel
    let mem_bandwidth_gbs = 50.0; // Assume dual channel DDR4
    let mem_bound_peak = mem_bandwidth_gbs * ai;
    let compute_peak = 200.0; // Assume ~200 GFLOP/s for AVX2 FMA

    println!("=== Roofline Model ===");
    println!(
        "Memory bandwidth: {:.0} GB/s (DDR4 dual)",
        mem_bandwidth_gbs
    );
    println!("Memory-bound peak: {:.2} GFLOP/s", mem_bound_peak);
    println!("Compute peak: {:.0} GFLOP/s (AVX2 FMA)", compute_peak);
    println!("Achieved: {:.2} GFLOP/s", achieved_gflops);
    println!(
        "Efficiency: {:.1}% of memory-bound peak",
        100.0 * achieved_gflops / mem_bound_peak
    );

    let decoded = mapped.model.decode(&output[tokens.len()..]);
    println!("\n=== Output ===\n{decoded}");
}
