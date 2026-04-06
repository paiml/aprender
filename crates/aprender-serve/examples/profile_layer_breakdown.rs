//! PAR-126: Profile time breakdown within a single layer
//! Identify which operations dominate the forward pass

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, GGUF_TYPE_Q4_K};
use realizar::quantize::fused_q4k_parallel_matvec_into;
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;

    println!("Model config:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  intermediate_dim: {}", intermediate_dim);
    println!("  num_heads: {}", num_heads);
    println!("  num_kv_heads: {}", num_kv_heads);
    println!("  head_dim: {}", head_dim);
    println!("  layers: {}", model.config().num_layers);

    // Get first layer for profiling
    let layer = &model.layers()[0];

    // Test inputs
    let hidden: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0)
        .collect();
    let q_dim = hidden_dim;
    let k_dim = num_kv_heads * head_dim;
    let v_dim = k_dim;
    let qkv_dim = q_dim + k_dim + v_dim;

    let iterations = 100;

    // Profile individual operations
    println!(
        "\n=== Operation Breakdown (avg of {} iterations) ===",
        iterations
    );

    // 1. RMSNorm (inline implementation for profiling)
    let mut normed = vec![0.0f32; hidden_dim];
    let norm_weight = &layer.attn_norm_weight;
    let eps = model.config().eps;
    let start = Instant::now();
    for _ in 0..iterations {
        // Compute variance
        let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        // Normalize and scale
        for (i, (out, inp)) in normed.iter_mut().zip(hidden.iter()).enumerate() {
            *out = inp * inv_rms * norm_weight[i];
        }
    }
    let rmsnorm_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("RMSNorm:        {:>8.0} us", rmsnorm_us);

    // 2. QKV matmul (main bottleneck)
    // Handle different QKV weight formats
    let mut qkv = vec![0.0f32; qkv_dim];
    let start = Instant::now();
    if let realizar::gguf::OwnedQKVWeights::Fused(ref weight) = &layer.qkv_weight {
        if weight.qtype == GGUF_TYPE_Q4_K {
            for _ in 0..iterations {
                fused_q4k_parallel_matvec_into(
                    &weight.data,
                    &normed,
                    weight.in_dim,
                    weight.out_dim,
                    &mut qkv,
                )?;
            }
        }
    }
    let qkv_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "QKV matmul:     {:>8.0} us ({}x{})",
        qkv_us, qkv_dim, hidden_dim
    );

    // 3. Attention output projection
    let mut attn_proj = vec![0.0f32; hidden_dim];
    let attn_out = vec![0.1f32; hidden_dim];
    let attn_weight = &layer.attn_output_weight;
    let start = Instant::now();
    for _ in 0..iterations {
        if attn_weight.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_parallel_matvec_into(
                &attn_weight.data,
                &attn_out,
                attn_weight.in_dim,
                attn_weight.out_dim,
                &mut attn_proj,
            )?;
        }
    }
    let attn_out_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Attn out:       {:>8.0} us ({}x{})",
        attn_out_us, attn_weight.out_dim, attn_weight.in_dim
    );

    // 4. FFN up
    let mut ffn_up = vec![0.0f32; intermediate_dim];
    let up_weight = &layer.ffn_up_weight;
    let start = Instant::now();
    for _ in 0..iterations {
        if up_weight.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_parallel_matvec_into(
                &up_weight.data,
                &normed,
                up_weight.in_dim,
                up_weight.out_dim,
                &mut ffn_up,
            )?;
        }
    }
    let ffn_up_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "FFN up:         {:>8.0} us ({}x{})",
        ffn_up_us, up_weight.out_dim, up_weight.in_dim
    );

    // 5. FFN gate
    let mut ffn_gate = vec![0.0f32; intermediate_dim];
    let mut ffn_gate_us = 0.0;
    if let Some(ref gate_weight) = layer.ffn_gate_weight {
        let start = Instant::now();
        for _ in 0..iterations {
            if gate_weight.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_parallel_matvec_into(
                    &gate_weight.data,
                    &normed,
                    gate_weight.in_dim,
                    gate_weight.out_dim,
                    &mut ffn_gate,
                )?;
            }
        }
        ffn_gate_us = start.elapsed().as_micros() as f64 / iterations as f64;
        println!(
            "FFN gate:       {:>8.0} us ({}x{})",
            ffn_gate_us, gate_weight.out_dim, gate_weight.in_dim
        );
    }

    // 6. FFN down
    let mut ffn_down = vec![0.0f32; hidden_dim];
    let ffn_hidden = vec![0.1f32; intermediate_dim];
    let down_weight = &layer.ffn_down_weight;
    let start = Instant::now();
    for _ in 0..iterations {
        if down_weight.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_parallel_matvec_into(
                &down_weight.data,
                &ffn_hidden,
                down_weight.in_dim,
                down_weight.out_dim,
                &mut ffn_down,
            )?;
        }
    }
    let ffn_down_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "FFN down:       {:>8.0} us ({}x{})",
        ffn_down_us, down_weight.out_dim, down_weight.in_dim
    );

    // Summary
    let layer_matmul_us = qkv_us + attn_out_us + ffn_up_us + ffn_gate_us + ffn_down_us;
    let layer_total_us = rmsnorm_us * 2.0 + layer_matmul_us;

    println!("\n=== Per-Layer Summary ===");
    println!("Matmuls only:   {:>8.0} us", layer_matmul_us);
    println!(
        "Layer total:    {:>8.0} us (estimated, excludes attention)",
        layer_total_us
    );

    let full_model_ms = layer_total_us * model.config().num_layers as f64 / 1000.0;
    println!(
        "\n=== Full Model Estimate ({} layers) ===",
        model.config().num_layers
    );
    println!("Matmuls+norms:  {:>8.1} ms", full_model_ms);

    let actual_ms = 55.0; // From measurement
    let overhead_ms = actual_ms - full_model_ms;
    println!("Actual (24t):   {:>8.1} ms", actual_ms);
    println!(
        "Overhead:       {:>8.1} ms ({:.0}%)",
        overhead_ms,
        overhead_ms / actual_ms * 100.0
    );

    // Compare to Ollama
    let ollama_ms = 14.2; // 70.59 tok/s
    println!("\n=== vs Ollama ===");
    println!("Ollama:         {:>8.1} ms (70.59 tok/s)", ollama_ms);
    println!("realizar:       {:>8.1} ms (18.2 tok/s)", actual_ms);
    println!("Ratio:          {:>8.1}x slower", actual_ms / ollama_ms);

    // Matmul GFLOPS analysis
    let qkv_flops = 2.0 * qkv_dim as f64 * hidden_dim as f64;
    let attn_out_flops = 2.0 * hidden_dim as f64 * hidden_dim as f64;
    let ffn_up_flops = 2.0 * intermediate_dim as f64 * hidden_dim as f64;
    let ffn_down_flops = 2.0 * hidden_dim as f64 * intermediate_dim as f64;

    println!("\n=== GFLOPS Analysis ===");
    println!("QKV:      {:.2} GFLOPS", qkv_flops / (qkv_us * 1000.0));
    println!(
        "Attn out: {:.2} GFLOPS",
        attn_out_flops / (attn_out_us * 1000.0)
    );
    println!(
        "FFN up:   {:.2} GFLOPS",
        ffn_up_flops / (ffn_up_us * 1000.0)
    );
    println!(
        "FFN down: {:.2} GFLOPS",
        ffn_down_flops / (ffn_down_us * 1000.0)
    );

    Ok(())
}
