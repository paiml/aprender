//! Profile matmul sizes and timings for each layer operation

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let num_layers = model.layers().len();

    println!("Model config:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  intermediate_dim: {}", intermediate_dim);
    println!("  num_layers: {}", num_layers);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);

    let layer = &model.layers()[0];

    // Get QKV dimensions
    let (q_dim, k_dim, v_dim) = match &layer.qkv_weight {
        OwnedQKVWeights::Fused(w) => {
            let total = w.out_dim;
            let third = total / 3;
            (third, third, third)
        },
        OwnedQKVWeights::Separate { q, k, v } => (q.out_dim, k.out_dim, v.out_dim),
    };

    println!("\nLayer 0 weight dimensions:");
    println!(
        "  QKV: q={}, k={}, v={} (total {})",
        q_dim,
        k_dim,
        v_dim,
        q_dim + k_dim + v_dim
    );
    println!(
        "  attn_output: {}x{}",
        layer.attn_output_weight.out_dim, layer.attn_output_weight.in_dim
    );
    println!(
        "  ffn_up: {}x{}",
        layer.ffn_up_weight.out_dim, layer.ffn_up_weight.in_dim
    );
    if let Some(ref gate) = layer.ffn_gate_weight {
        println!("  ffn_gate: {}x{}", gate.out_dim, gate.in_dim);
    }
    println!(
        "  ffn_down: {}x{}",
        layer.ffn_down_weight.out_dim, layer.ffn_down_weight.in_dim
    );

    println!(
        "\nlm_head: {}x{}",
        model.lm_head_weight().out_dim,
        model.lm_head_weight().in_dim
    );

    // Benchmark each matmul type
    let iterations = 100;
    let activations: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.017).sin()).collect();
    let intermediate: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    println!("\n=== Matmul Timings ===\n");

    // QKV matmul
    let qkv_out_dim = q_dim + k_dim + v_dim;
    let mut qkv_out = vec![0.0f32; qkv_out_dim];
    match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => {
            let mut q_out = vec![0.0f32; q_dim];
            let mut k_out = vec![0.0f32; k_dim];
            let mut v_out = vec![0.0f32; v_dim];

            let start = Instant::now();
            for _ in 0..iterations {
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &q.data,
                    &activations,
                    q.in_dim,
                    q.out_dim,
                    &mut q_out,
                )?;
            }
            let q_us = start.elapsed().as_micros() as f64 / iterations as f64;

            let start = Instant::now();
            for _ in 0..iterations {
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &k.data,
                    &activations,
                    k.in_dim,
                    k.out_dim,
                    &mut k_out,
                )?;
            }
            let k_us = start.elapsed().as_micros() as f64 / iterations as f64;

            let start = Instant::now();
            for _ in 0..iterations {
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &v.data,
                    &activations,
                    v.in_dim,
                    v.out_dim,
                    &mut v_out,
                )?;
            }
            let v_us = start.elapsed().as_micros() as f64 / iterations as f64;

            println!(
                "Q matmul ({}x{}):     {:>8.1} us",
                q.out_dim, q.in_dim, q_us
            );
            println!(
                "K matmul ({}x{}):      {:>8.1} us",
                k.out_dim, k.in_dim, k_us
            );
            println!(
                "V matmul ({}x{}):      {:>8.1} us",
                v.out_dim, v.in_dim, v_us
            );
            println!("QKV total:             {:>8.1} us", q_us + k_us + v_us);
        },
        OwnedQKVWeights::Fused(w) => {
            let start = Instant::now();
            for _ in 0..iterations {
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &w.data,
                    &activations,
                    w.in_dim,
                    w.out_dim,
                    &mut qkv_out,
                )?;
            }
            let qkv_us = start.elapsed().as_micros() as f64 / iterations as f64;
            println!("QKV fused ({}x{}): {:>8.1} us", w.out_dim, w.in_dim, qkv_us);
        },
    }

    // Attn output
    let mut attn_out = vec![0.0f32; layer.attn_output_weight.out_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &layer.attn_output_weight.data,
            &activations,
            layer.attn_output_weight.in_dim,
            layer.attn_output_weight.out_dim,
            &mut attn_out,
        )?;
    }
    let attn_out_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Attn output ({}x{}): {:>8.1} us",
        layer.attn_output_weight.out_dim, layer.attn_output_weight.in_dim, attn_out_us
    );

    // FFN up
    let mut ffn_up = vec![0.0f32; layer.ffn_up_weight.out_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &layer.ffn_up_weight.data,
            &activations,
            layer.ffn_up_weight.in_dim,
            layer.ffn_up_weight.out_dim,
            &mut ffn_up,
        )?;
    }
    let ffn_up_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "FFN up ({}x{}):     {:>8.1} us",
        layer.ffn_up_weight.out_dim, layer.ffn_up_weight.in_dim, ffn_up_us
    );

    // FFN gate
    let mut ffn_gate_us = 0.0;
    if let Some(ref gate) = layer.ffn_gate_weight {
        let mut ffn_gate = vec![0.0f32; gate.out_dim];
        let start = Instant::now();
        for _ in 0..iterations {
            realizar::quantize::fused_q4k_parallel_matvec_into(
                &gate.data,
                &activations,
                gate.in_dim,
                gate.out_dim,
                &mut ffn_gate,
            )?;
        }
        ffn_gate_us = start.elapsed().as_micros() as f64 / iterations as f64;
        println!(
            "FFN gate ({}x{}):   {:>8.1} us",
            gate.out_dim, gate.in_dim, ffn_gate_us
        );
    }

    // FFN down
    let mut ffn_down = vec![0.0f32; layer.ffn_down_weight.out_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &layer.ffn_down_weight.data,
            &intermediate,
            layer.ffn_down_weight.in_dim,
            layer.ffn_down_weight.out_dim,
            &mut ffn_down,
        )?;
    }
    let ffn_down_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "FFN down ({}x{}):   {:>8.1} us",
        layer.ffn_down_weight.out_dim, layer.ffn_down_weight.in_dim, ffn_down_us
    );

    // LM head
    let mut logits = vec![0.0f32; model.lm_head_weight().out_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        realizar::quantize::fused_q4k_parallel_matvec_into(
            &model.lm_head_weight().data,
            &activations,
            model.lm_head_weight().in_dim,
            model.lm_head_weight().out_dim,
            &mut logits,
        )?;
    }
    let lm_head_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "LM head ({}x{}):  {:>8.1} us",
        model.lm_head_weight().out_dim,
        model.lm_head_weight().in_dim,
        lm_head_us
    );

    // Calculate total
    println!("\n{}", "=".repeat(60));
    println!("FORWARD PASS MATMUL TIME (28 layers + lm_head)");
    println!("{}", "=".repeat(60));

    // Get QKV time
    let qkv_per_layer = match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => {
            // Re-measure if needed (already have q_us, k_us, v_us from above)
            let mut q_out = vec![0.0f32; q_dim];
            let mut k_out = vec![0.0f32; k_dim];
            let mut v_out = vec![0.0f32; v_dim];
            let start = Instant::now();
            for _ in 0..iterations {
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &q.data,
                    &activations,
                    q.in_dim,
                    q.out_dim,
                    &mut q_out,
                )?;
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &k.data,
                    &activations,
                    k.in_dim,
                    k.out_dim,
                    &mut k_out,
                )?;
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &v.data,
                    &activations,
                    v.in_dim,
                    v.out_dim,
                    &mut v_out,
                )?;
            }
            start.elapsed().as_micros() as f64 / iterations as f64
        },
        OwnedQKVWeights::Fused(w) => {
            let mut qkv_out = vec![0.0f32; w.out_dim];
            let start = Instant::now();
            for _ in 0..iterations {
                realizar::quantize::fused_q4k_parallel_matvec_into(
                    &w.data,
                    &activations,
                    w.in_dim,
                    w.out_dim,
                    &mut qkv_out,
                )?;
            }
            start.elapsed().as_micros() as f64 / iterations as f64
        },
    };

    let per_layer_matmul = qkv_per_layer + attn_out_us + ffn_up_us + ffn_gate_us + ffn_down_us;
    let total_matmul_ms = (per_layer_matmul * num_layers as f64 + lm_head_us) / 1000.0;

    println!("QKV per layer:           {:>8.1} us", qkv_per_layer);
    println!("Attn output per layer:   {:>8.1} us", attn_out_us);
    println!("FFN up per layer:        {:>8.1} us", ffn_up_us);
    println!("FFN gate per layer:      {:>8.1} us", ffn_gate_us);
    println!("FFN down per layer:      {:>8.1} us", ffn_down_us);
    println!("--------------------------------");
    println!("Per layer total:         {:>8.1} us", per_layer_matmul);
    println!(
        "28 layers:               {:>8.1} ms",
        per_layer_matmul * 28.0 / 1000.0
    );
    println!("LM head:                 {:>8.1} us", lm_head_us);
    println!("================================");
    println!("Total matmul time:       {:>8.1} ms", total_matmul_ms);

    // Compare to actual
    println!("\n{}", "=".repeat(60));
    println!("COMPARISON");
    println!("{}", "=".repeat(60));
    println!("Measured matmul only:    {:>8.1} ms", total_matmul_ms);
    println!("Actual forward pass:     {:>8.1} ms", 102.0);
    println!(
        "Non-matmul overhead:     {:>8.1} ms ({:.0}%)",
        102.0 - total_matmul_ms,
        100.0 * (102.0 - total_matmul_ms) / 102.0
    );

    println!("\nllama.cpp target:        {:>8.1} ms", 12.8);
    println!("Our matmul gap vs llama: {:>8.2}x", total_matmul_ms / 12.8);

    Ok(())
}
