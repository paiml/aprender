//! PAR-126: Profile matmul cold vs warm to measure cache effects

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};
use realizar::quantize::{fused_q4k_parallel_matvec_into, fused_q6k_parallel_matvec_into};
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

    println!("=== PAR-126 Cold vs Warm Matmul Profiler ===\n");
    println!(
        "Model: {} layers, hidden={}",
        model.config().num_layers,
        hidden_dim
    );

    let layer = &model.layers()[0];

    // Test: Measure FIRST iteration (cold cache) vs average of 100 (warm)
    println!("\n=== Cold vs Warm Timing ===\n");

    // FFN down (Q6_K - largest operation)
    let w = &layer.ffn_down_weight;
    let ffn_input: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 / intermediate_dim as f32) * 2.0 - 1.0)
        .collect();
    let mut output = vec![0.0f32; hidden_dim];

    // Drop page cache to simulate cold start
    // Note: This requires root, so we'll just use fresh allocation
    let mut output_cold: Vec<f32> = vec![0.0f32; hidden_dim];

    // Measure cold (first iteration)
    let start = Instant::now();
    fused_q6k_parallel_matvec_into(&w.data, &ffn_input, w.in_dim, w.out_dim, &mut output_cold)?;
    let cold_us = start.elapsed().as_micros() as f64;

    // Measure warm (100 iterations)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q6k_parallel_matvec_into(&w.data, &ffn_input, w.in_dim, w.out_dim, &mut output)?;
    }
    let warm_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("FFN down (Q6_K {}x{}):", w.out_dim, w.in_dim);
    println!("  Cold (1st iter): {:.0} µs", cold_us);
    println!("  Warm (avg 100):  {:.0} µs", warm_us);
    println!("  Cold/Warm ratio: {:.2}x", cold_us / warm_us);

    // FFN gate (Q4_K)
    if let Some(ref w) = layer.ffn_gate_weight {
        let hidden_input: Vec<f32> = (0..hidden_dim)
            .map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0)
            .collect();
        let mut gate_out_cold = vec![0.0f32; intermediate_dim];
        let mut gate_out = vec![0.0f32; intermediate_dim];

        let start = Instant::now();
        fused_q4k_parallel_matvec_into(
            &w.data,
            &hidden_input,
            w.in_dim,
            w.out_dim,
            &mut gate_out_cold,
        )?;
        let cold_us = start.elapsed().as_micros() as f64;

        let start = Instant::now();
        for _ in 0..iterations {
            fused_q4k_parallel_matvec_into(
                &w.data,
                &hidden_input,
                w.in_dim,
                w.out_dim,
                &mut gate_out,
            )?;
        }
        let warm_us = start.elapsed().as_micros() as f64 / iterations as f64;

        println!("\nFFN gate (Q4_K {}x{}):", w.out_dim, w.in_dim);
        println!("  Cold (1st iter): {:.0} µs", cold_us);
        println!("  Warm (avg 100):  {:.0} µs", warm_us);
        println!("  Cold/Warm ratio: {:.2}x", cold_us / warm_us);
    }

    // Now test with PRE-ALLOCATED buffers (like the actual forward pass)
    println!("\n=== With Pre-Allocated Buffers (Like Actual Forward Pass) ===\n");

    // Pre-allocate all buffers
    let mut q_out = vec![0.0f32; hidden_dim];
    let mut k_out = vec![0.0f32; hidden_dim / 6]; // num_kv_heads * head_dim
    let mut v_out = vec![0.0f32; hidden_dim / 6];
    let mut attn_out_buf = vec![0.0f32; hidden_dim];
    let mut up_out = vec![0.0f32; intermediate_dim];
    let mut gate_out = vec![0.0f32; intermediate_dim];
    let mut down_out = vec![0.0f32; hidden_dim];
    let hidden_input: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0)
        .collect();
    let attn_input: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0)
        .collect();

    let runs = 20;
    let mut total_preallocated_us = 0.0f64;

    for _ in 0..runs {
        let start = Instant::now();

        // Simulate one layer's matmuls with pre-allocated buffers
        // QKV
        if let realizar::gguf::OwnedQKVWeights::Separate { q, k, v } = &layer.qkv_weight {
            if q.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_parallel_matvec_into(
                    &q.data,
                    &hidden_input,
                    q.in_dim,
                    q.out_dim,
                    &mut q_out,
                )?;
            }
            if k.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_parallel_matvec_into(
                    &k.data,
                    &hidden_input,
                    k.in_dim,
                    k.out_dim,
                    &mut k_out,
                )?;
            } else if k.qtype == GGUF_TYPE_Q6_K {
                fused_q6k_parallel_matvec_into(
                    &k.data,
                    &hidden_input,
                    k.in_dim,
                    k.out_dim,
                    &mut k_out,
                )?;
            }
            if v.qtype == GGUF_TYPE_Q6_K {
                fused_q6k_parallel_matvec_into(
                    &v.data,
                    &hidden_input,
                    v.in_dim,
                    v.out_dim,
                    &mut v_out,
                )?;
            }
        }

        // Attn out
        let w = &layer.attn_output_weight;
        if w.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_parallel_matvec_into(
                &w.data,
                &attn_input,
                w.in_dim,
                w.out_dim,
                &mut attn_out_buf,
            )?;
        }

        // FFN up, gate
        let w = &layer.ffn_up_weight;
        if w.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_parallel_matvec_into(
                &w.data,
                &hidden_input,
                w.in_dim,
                w.out_dim,
                &mut up_out,
            )?;
        }

        if let Some(ref w) = layer.ffn_gate_weight {
            if w.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_parallel_matvec_into(
                    &w.data,
                    &hidden_input,
                    w.in_dim,
                    w.out_dim,
                    &mut gate_out,
                )?;
            }
        }

        // FFN down
        let w = &layer.ffn_down_weight;
        if w.qtype == GGUF_TYPE_Q6_K {
            fused_q6k_parallel_matvec_into(
                &w.data,
                &ffn_input,
                w.in_dim,
                w.out_dim,
                &mut down_out,
            )?;
        }

        total_preallocated_us += start.elapsed().as_micros() as f64;
    }

    let preallocated_per_layer_us = total_preallocated_us / runs as f64;
    let preallocated_full_model_ms =
        preallocated_per_layer_us * model.config().num_layers as f64 / 1000.0;

    println!(
        "Pre-allocated matmuls per layer: {:.0} µs",
        preallocated_per_layer_us
    );
    println!(
        "Full model (28 layers): {:.1} ms",
        preallocated_full_model_ms
    );

    // Compare all variations
    let isolated_per_layer_us = 525.0; // From profile_all_matmuls (warm, isolated)
    println!("\n=== Summary Comparison ===");
    println!(
        "Isolated (warm):    {:.0} µs/layer ({:.1} ms/model)",
        isolated_per_layer_us,
        isolated_per_layer_us * 28.0 / 1000.0
    );
    println!(
        "Pre-allocated:      {:.0} µs/layer ({:.1} ms/model)",
        preallocated_per_layer_us, preallocated_full_model_ms
    );
    println!(
        "Overhead vs warm:   {:.2}x",
        preallocated_per_layer_us / isolated_per_layer_us
    );

    // Gap to actual forward pass
    let actual_per_token_ms = 33.5;
    let attention_ms = 1.2;
    let other_ops_ms = 0.3;
    let matmul_and_ops = preallocated_full_model_ms + attention_ms + other_ops_ms;
    println!("\n=== Gap Analysis ===");
    println!("Matmuls:        {:.1} ms", preallocated_full_model_ms);
    println!("Attention:      {:.1} ms", attention_ms);
    println!("Other ops:      {:.1} ms", other_ops_ms);
    println!("Total estimated: {:.1} ms", matmul_and_ops);
    println!("Actual:         {:.1} ms", actual_per_token_ms);
    println!(
        "Still unexplained: {:.1} ms ({:.0}%)",
        actual_per_token_ms - matmul_and_ops,
        (actual_per_token_ms - matmul_and_ops) / actual_per_token_ms * 100.0
    );

    Ok(())
}
