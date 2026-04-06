//! PAR-126: Profile matmuls through ALL 28 layers (not just layer 0)
//! This simulates the actual forward pass cache behavior

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
    let num_layers = model.config().num_layers;

    println!("=== PAR-126: Profile ALL {} Layers ===\n", num_layers);
    println!(
        "Model: hidden={}, intermediate={}",
        hidden_dim, intermediate_dim
    );

    // Pre-allocate all buffers
    let mut q_out = vec![0.0f32; hidden_dim];
    let mut k_out = vec![0.0f32; hidden_dim / 6];
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
    let ffn_input: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 / intermediate_dim as f32) * 2.0 - 1.0)
        .collect();

    // Warmup - run through all layers once
    println!("Warming up...");
    for layer in model.layers() {
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
        if layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_parallel_matvec_into(
                &layer.attn_output_weight.data,
                &attn_input,
                layer.attn_output_weight.in_dim,
                layer.attn_output_weight.out_dim,
                &mut attn_out_buf,
            )?;
        }
        if layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K {
            fused_q4k_parallel_matvec_into(
                &layer.ffn_up_weight.data,
                &hidden_input,
                layer.ffn_up_weight.in_dim,
                layer.ffn_up_weight.out_dim,
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
        if layer.ffn_down_weight.qtype == GGUF_TYPE_Q6_K {
            fused_q6k_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &ffn_input,
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
                &mut down_out,
            )?;
        }
    }

    // Measure: Run through all layers multiple times
    let runs = 20;
    let mut total_us = 0.0f64;

    println!(
        "\nMeasuring {} runs through all {} layers...",
        runs, num_layers
    );
    for _ in 0..runs {
        let start = Instant::now();

        for layer in model.layers() {
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
            if layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_parallel_matvec_into(
                    &layer.attn_output_weight.data,
                    &attn_input,
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                    &mut attn_out_buf,
                )?;
            }

            // FFN up, gate
            if layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K {
                fused_q4k_parallel_matvec_into(
                    &layer.ffn_up_weight.data,
                    &hidden_input,
                    layer.ffn_up_weight.in_dim,
                    layer.ffn_up_weight.out_dim,
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
            if layer.ffn_down_weight.qtype == GGUF_TYPE_Q6_K {
                fused_q6k_parallel_matvec_into(
                    &layer.ffn_down_weight.data,
                    &ffn_input,
                    layer.ffn_down_weight.in_dim,
                    layer.ffn_down_weight.out_dim,
                    &mut down_out,
                )?;
            }
        }

        total_us += start.elapsed().as_micros() as f64;
    }

    let per_forward_ms = total_us / runs as f64 / 1000.0;
    let per_layer_us = total_us / runs as f64 / num_layers as f64;

    println!("\n=== Results ===");
    println!("All-layers matmuls: {:.1} ms/forward", per_forward_ms);
    println!("Per-layer average:  {:.0} µs", per_layer_us);

    // Compare to isolated profiling
    let isolated_per_layer_us = 525.0;
    let isolated_full_ms = isolated_per_layer_us * num_layers as f64 / 1000.0;
    println!("\n=== Comparison ===");
    println!(
        "Isolated (warm, layer 0): {:.0} µs/layer ({:.1} ms/model)",
        isolated_per_layer_us, isolated_full_ms
    );
    println!(
        "All layers (realistic):   {:.0} µs/layer ({:.1} ms/model)",
        per_layer_us, per_forward_ms
    );
    println!(
        "Overhead factor:          {:.2}x",
        per_forward_ms / isolated_full_ms
    );

    // Full gap analysis
    println!("\n=== Full Gap Analysis ===");
    let attention_ms = 1.2; // From attention profiler at cache_len=50
    let other_ops_ms = 0.5; // RMSNorm, RoPE, etc (from instrumented profiler)
    let q8k_quant_ms = 0.2; // Q8K quantization overhead
    let rayon_overhead_ms = 2.0; // Rayon dispatch
    let total_estimated_ms =
        per_forward_ms + attention_ms + other_ops_ms + q8k_quant_ms + rayon_overhead_ms;

    println!("Matmuls (all layers): {:.1} ms", per_forward_ms);
    println!("Attention:            {:.1} ms", attention_ms);
    println!("Other ops:            {:.1} ms", other_ops_ms);
    println!("Q8K quantization:     {:.1} ms", q8k_quant_ms);
    println!("Rayon overhead:       {:.1} ms", rayon_overhead_ms);
    println!("Total estimated:      {:.1} ms", total_estimated_ms);

    let actual_ms = 33.5; // From forward pass profiler
    let unexplained_ms = actual_ms - total_estimated_ms;
    println!("\nActual:               {:.1} ms", actual_ms);
    println!(
        "Unexplained:          {:.1} ms ({:.0}%)",
        unexplained_ms,
        unexplained_ms / actual_ms * 100.0
    );

    // Ollama comparison
    let ollama_ms = 14.05;
    println!("\n=== vs Ollama ===");
    println!("Ollama:    {:.2} ms/tok (71.2 tok/s)", ollama_ms);
    println!(
        "realizar:  {:.2} ms/tok ({:.1} tok/s)",
        actual_ms,
        1000.0 / actual_ms
    );
    println!("Gap:       {:.2}x", actual_ms / ollama_ms);

    // What would it take to match?
    println!("\n=== Path to Parity ===");
    println!(
        "Our matmuls alone ({:.1} ms) exceed Ollama total ({:.1} ms)",
        per_forward_ms, ollama_ms
    );
    println!(
        "Root cause: Matmul kernels are {:.2}x slower than llama.cpp",
        per_forward_ms / ollama_ms
    );

    // ComputeBlocks/sec calculation
    let compute_blocks_per_token = num_layers; // One ComputeBlock per layer
    let compute_blocks_per_sec = compute_blocks_per_token as f64 * 1000.0 / actual_ms;
    println!("\n=== ComputeBlocks/sec ===");
    println!("ComputeBlocks per token: {}", compute_blocks_per_token);
    println!("Current: {:.0} CB/s", compute_blocks_per_sec);
    println!(
        "Target (2x Ollama): {:.0} CB/s",
        compute_blocks_per_token as f64 * 142.3
    );

    Ok(())
}
