//! PAR-126: Profile ALL matmul operations in a layer

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
    let num_heads = model.config().num_heads;
    let _num_kv_heads = model.config().num_kv_heads;
    let _head_dim = hidden_dim / num_heads;

    println!(
        "Model: {} layers, hidden={}, intermediate={}",
        model.config().num_layers,
        hidden_dim,
        intermediate_dim
    );

    let layer = &model.layers()[0];
    let iterations = 100;

    // Input vectors
    let hidden: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 / hidden_dim as f32) * 2.0 - 1.0)
        .collect();
    let ffn_activated: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 / intermediate_dim as f32) * 2.0 - 1.0)
        .collect();

    println!("\n=== Per-Operation Timing ({} iterations) ===", iterations);

    // Helper for profiling
    fn profile_matmul(
        name: &str,
        weight_data: &[u8],
        qtype: u32,
        input: &[f32],
        in_dim: usize,
        out_dim: usize,
        iterations: usize,
    ) -> Result<f64, RealizarError> {
        let mut output = vec![0.0f32; out_dim];

        // Warmup
        match qtype {
            GGUF_TYPE_Q4_K => {
                fused_q4k_parallel_matvec_into(weight_data, input, in_dim, out_dim, &mut output)?;
            },
            GGUF_TYPE_Q6_K => {
                fused_q6k_parallel_matvec_into(weight_data, input, in_dim, out_dim, &mut output)?;
            },
            _ => return Ok(0.0),
        }

        let start = Instant::now();
        for _ in 0..iterations {
            match qtype {
                GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec_into(
                    weight_data,
                    input,
                    in_dim,
                    out_dim,
                    &mut output,
                )?,
                GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec_into(
                    weight_data,
                    input,
                    in_dim,
                    out_dim,
                    &mut output,
                )?,
                _ => {},
            }
        }
        let us = start.elapsed().as_micros() as f64 / iterations as f64;

        let qtype_name = match qtype {
            GGUF_TYPE_Q4_K => "Q4_K",
            GGUF_TYPE_Q6_K => "Q6_K",
            _ => "?",
        };
        println!(
            "{:<12} {:>8.0} us  {}  {}x{}",
            name, us, qtype_name, out_dim, in_dim
        );
        Ok(us)
    }

    let mut total_us = 0.0;

    // QKV (separate Q, K, V)
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            total_us += profile_matmul(
                "Q", &q.data, q.qtype, &hidden, q.in_dim, q.out_dim, iterations,
            )?;
            total_us += profile_matmul(
                "K", &k.data, k.qtype, &hidden, k.in_dim, k.out_dim, iterations,
            )?;
            total_us += profile_matmul(
                "V", &v.data, v.qtype, &hidden, v.in_dim, v.out_dim, iterations,
            )?;
        },
        realizar::gguf::OwnedQKVWeights::Fused(ref w) => {
            total_us += profile_matmul(
                "QKV", &w.data, w.qtype, &hidden, w.in_dim, w.out_dim, iterations,
            )?;
        },
    }

    // Attn output
    let w = &layer.attn_output_weight;
    let attn_out: Vec<f32> = (0..w.in_dim)
        .map(|i| (i as f32 / w.in_dim as f32) * 2.0 - 1.0)
        .collect();
    total_us += profile_matmul(
        "Attn out", &w.data, w.qtype, &attn_out, w.in_dim, w.out_dim, iterations,
    )?;

    // FFN up
    let w = &layer.ffn_up_weight;
    total_us += profile_matmul(
        "FFN up", &w.data, w.qtype, &hidden, w.in_dim, w.out_dim, iterations,
    )?;

    // FFN gate
    if let Some(ref w) = layer.ffn_gate_weight {
        total_us += profile_matmul(
            "FFN gate", &w.data, w.qtype, &hidden, w.in_dim, w.out_dim, iterations,
        )?;
    }

    // FFN down
    let w = &layer.ffn_down_weight;
    total_us += profile_matmul(
        "FFN down",
        &w.data,
        w.qtype,
        &ffn_activated,
        w.in_dim,
        w.out_dim,
        iterations,
    )?;

    // Summary
    println!("\n=== Summary ===");
    println!(
        "Per-layer matmul: {:.0} us = {:.2} ms",
        total_us,
        total_us / 1000.0
    );

    let num_layers = model.config().num_layers as f64;
    let model_matmul_ms = total_us * num_layers / 1000.0;
    println!("Full model (matmuls only): {:.1} ms", model_matmul_ms);

    // Compare to actual
    let actual_ms = 55.0; // From measure_forward_time with 24 threads
    let other_ms = actual_ms - model_matmul_ms;
    println!("Actual forward pass: {:.1} ms", actual_ms);
    println!(
        "Other (attention + overhead): {:.1} ms ({:.0}%)",
        other_ms,
        other_ms / actual_ms * 100.0
    );

    // Ollama comparison
    let ollama_ms = 14.2;
    println!("\n=== vs Ollama ===");
    println!("Ollama: {:.1} ms/tok (70.59 tok/s)", ollama_ms);
    println!("realizar matmuls: {:.1} ms", model_matmul_ms);
    println!("Matmul ratio: {:.1}x", model_matmul_ms / ollama_ms);

    // If our matmuls alone take longer than Ollama's full forward pass,
    // our kernels are the bottleneck
    if model_matmul_ms > ollama_ms {
        println!(
            "\n⚠️  Our matmuls alone ({:.1} ms) exceed Ollama total ({:.1} ms)",
            model_matmul_ms, ollama_ms
        );
        println!("   Root cause: Matmul kernels need optimization");
    }

    Ok(())
}
