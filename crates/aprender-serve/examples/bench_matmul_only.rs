//! Benchmark pure matmul performance, isolated from forward pass
//!
//! Measures just the Q4_K matmul kernel to identify true bottleneck

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let num_layers = model.config().num_layers;

    println!(
        "hidden_dim: {}, intermediate_dim: {}, layers: {}",
        hidden_dim, intermediate_dim, num_layers
    );

    // Create random activations
    let activations: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017).sin()).collect();

    // Get first layer for testing
    let layer = &model.layers()[0];
    let ffn_up_weight = &layer.ffn_up_weight;

    println!(
        "\nFFN Up weight: {}x{} (Q4_K)",
        ffn_up_weight.out_dim, ffn_up_weight.in_dim
    );
    println!("Weight data size: {} bytes", ffn_up_weight.data.len());

    // Warmup
    for _ in 0..10 {
        let _ = realizar::quantize::fused_q4k_parallel_matvec(
            &ffn_up_weight.data,
            &activations,
            hidden_dim,
            intermediate_dim,
        )?;
    }

    // Benchmark matmul
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = realizar::quantize::fused_q4k_parallel_matvec(
            &ffn_up_weight.data,
            &activations,
            hidden_dim,
            intermediate_dim,
        )?;
    }
    let elapsed = start.elapsed();

    let us_per_matmul = elapsed.as_micros() as f64 / iterations as f64;
    let matmuls_per_sec = 1_000_000.0 / us_per_matmul;

    // Calculate bytes processed
    // Q4_K: 4.5 bits per element = 0.5625 bytes
    let weight_bytes = (intermediate_dim * hidden_dim) as f64 * 0.5625;
    let activation_bytes = hidden_dim as f64 * 4.0; // f32
    let total_bytes = weight_bytes + activation_bytes;
    let bandwidth_gbs = (total_bytes * matmuls_per_sec) / 1e9;

    println!("\n=== FFN Up Matmul Performance ===");
    println!("Time per matmul: {:.1} µs", us_per_matmul);
    println!("Matmuls per sec: {:.0}", matmuls_per_sec);
    println!("Bytes per matmul: {:.2} MB", total_bytes / 1e6);
    println!("Effective bandwidth: {:.1} GB/s", bandwidth_gbs);

    // Calculate theoretical tok/s if only matmuls mattered
    // Per token: 22 layers × (QKV + O + Up + Gate + Down) matmuls
    // Attention: 3 QKV + 1 O = 4 matmuls (smaller)
    // FFN: 1 up + 1 gate + 1 down = 3 matmuls
    // Total: 7 matmuls per layer × 22 layers = 154 matmuls + 1 LM head
    let matmuls_per_token = 155;
    let us_per_token = us_per_matmul * matmuls_per_token as f64;
    let toks_per_sec = 1_000_000.0 / us_per_token;

    println!("\n=== Theoretical Token Performance ===");
    println!("Matmuls per token: {}", matmuls_per_token);
    println!(
        "Time per token (matmul only): {:.1} ms",
        us_per_token / 1000.0
    );
    println!("Theoretical tok/s: {:.1}", toks_per_sec);
    println!("\nActual tok/s: ~12");
    println!("Gap: {:.1}x (other overhead)", toks_per_sec / 12.0);

    // Now test with QKV matmul (smaller)
    let realizar::gguf::OwnedQKVWeights::Fused(qkv_weight) = &layer.qkv_weight else {
        println!("\nQKV weight is not fused, skipping");
        return Ok(());
    };

    println!("\n=== QKV Matmul Performance ===");
    println!("QKV weight: {}x{}", qkv_weight.out_dim, qkv_weight.in_dim);

    // Warmup
    for _ in 0..10 {
        let _ = realizar::quantize::fused_q4k_parallel_matvec(
            &qkv_weight.data,
            &activations,
            qkv_weight.in_dim,
            qkv_weight.out_dim,
        )?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = realizar::quantize::fused_q4k_parallel_matvec(
            &qkv_weight.data,
            &activations,
            qkv_weight.in_dim,
            qkv_weight.out_dim,
        )?;
    }
    let elapsed = start.elapsed();

    let us_per_matmul = elapsed.as_micros() as f64 / iterations as f64;
    let weight_bytes = (qkv_weight.out_dim * qkv_weight.in_dim) as f64 * 0.5625;
    let total_bytes = weight_bytes + activation_bytes;
    let bandwidth_gbs = (total_bytes / us_per_matmul) * 1_000_000.0 / 1e9;

    println!("Time per matmul: {:.1} µs", us_per_matmul);
    println!("Bytes per matmul: {:.2} MB", total_bytes / 1e6);
    println!("Effective bandwidth: {:.1} GB/s", bandwidth_gbs);

    Ok(())
}
