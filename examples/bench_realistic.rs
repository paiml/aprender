//! Realistic per-token benchmark with actual 1.5B model dimensions
//! Measures all matmuls that happen per token

use std::time::Instant;

fn main() {
    println!("Realistic 1.5B Model Per-Token Benchmark");
    println!("=========================================\n");

    // Limit threads for NUMA optimization - must build global pool before any rayon use
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .ok();

    println!("Using {} threads\n", rayon::current_num_threads());

    // Qwen2.5-Coder 1.5B dimensions (from GGUF)
    let hidden = 1536;
    let intermediate = 8960;
    let vocab = 151936;
    let layers = 28;

    // Create weight buffers for each matmul type
    let bytes_per_element = |in_dim: usize| -> usize {
        let super_blocks = in_dim.div_ceil(256);
        super_blocks * 144 // Q4_K format
    };

    // Pre-create all weight buffers
    let qkv_weight = vec![0u8; bytes_per_element(hidden) * (hidden * 3)];
    let attn_o_weight = vec![0u8; bytes_per_element(hidden) * hidden];
    let ffn_up_weight = vec![0u8; bytes_per_element(hidden) * intermediate];
    let ffn_down_weight = vec![0u8; bytes_per_element(intermediate) * hidden];
    let lm_head_weight = vec![0u8; bytes_per_element(hidden) * vocab];

    // Create activation buffers
    let act_hidden: Vec<f32> = (0..hidden).map(|i| i as f32 / hidden as f32).collect();
    let act_inter: Vec<f32> = (0..intermediate)
        .map(|i| i as f32 / intermediate as f32)
        .collect();

    // Pre-quantize activations
    let (hidden_q8_scales, hidden_q8_quants) = quantize_to_q8k(&act_hidden);
    let (inter_q8_scales, inter_q8_quants) = quantize_to_q8k(&act_inter);

    // Output buffers (reused)
    let mut out_qkv = vec![0.0f32; hidden * 3];
    let mut out_hidden = vec![0.0f32; hidden];
    let mut out_inter = vec![0.0f32; intermediate];
    let mut out_vocab = vec![0.0f32; vocab];

    // Warmup
    for _ in 0..3 {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &ffn_up_weight,
            &hidden_q8_scales,
            &hidden_q8_quants,
            hidden,
            intermediate,
            &mut out_inter,
        );
    }

    let iters = 50;

    // QKV projection
    let start = Instant::now();
    for _ in 0..iters {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &qkv_weight,
            &hidden_q8_scales,
            &hidden_q8_quants,
            hidden,
            hidden * 3,
            &mut out_qkv,
        );
    }
    let qkv_us = start.elapsed().as_micros() as f64 / iters as f64;

    // Attention output
    let start = Instant::now();
    for _ in 0..iters {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &attn_o_weight,
            &hidden_q8_scales,
            &hidden_q8_quants,
            hidden,
            hidden,
            &mut out_hidden,
        );
    }
    let attn_o_us = start.elapsed().as_micros() as f64 / iters as f64;

    // FFN up
    let start = Instant::now();
    for _ in 0..iters {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &ffn_up_weight,
            &hidden_q8_scales,
            &hidden_q8_quants,
            hidden,
            intermediate,
            &mut out_inter,
        );
    }
    let ffn_up_us = start.elapsed().as_micros() as f64 / iters as f64;

    // FFN gate (same as up)
    let ffn_gate_us = ffn_up_us;

    // FFN down
    let start = Instant::now();
    for _ in 0..iters {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &ffn_down_weight,
            &inter_q8_scales,
            &inter_q8_quants,
            intermediate,
            hidden,
            &mut out_hidden,
        );
    }
    let ffn_down_us = start.elapsed().as_micros() as f64 / iters as f64;

    // LM head
    let start = Instant::now();
    for _ in 0..iters {
        let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
            &lm_head_weight,
            &hidden_q8_scales,
            &hidden_q8_quants,
            hidden,
            vocab,
            &mut out_vocab,
        );
    }
    let lm_head_us = start.elapsed().as_micros() as f64 / iters as f64;

    // Per-layer total
    let per_layer_us = qkv_us + attn_o_us + ffn_up_us + ffn_gate_us + ffn_down_us;

    // Total per token
    let total_per_token_us = per_layer_us * layers as f64 + lm_head_us;
    let tok_s = 1e6 / total_per_token_us;

    println!("Per-layer matmul times (16 threads):");
    println!("  QKV ({}→{}): {:.1} µs", hidden, hidden * 3, qkv_us);
    println!("  Attn O ({}→{}): {:.1} µs", hidden, hidden, attn_o_us);
    println!(
        "  FFN up ({}→{}): {:.1} µs",
        hidden, intermediate, ffn_up_us
    );
    println!(
        "  FFN gate ({}→{}): {:.1} µs",
        hidden, intermediate, ffn_gate_us
    );
    println!(
        "  FFN down ({}→{}): {:.1} µs",
        intermediate, hidden, ffn_down_us
    );
    println!("  Per-layer total: {:.1} µs", per_layer_us);

    println!("\nLM head ({}→{}): {:.1} µs", hidden, vocab, lm_head_us);

    println!("\n=== Total per token ===");
    println!(
        "{} layers × {:.1} µs + LM head {:.1} µs = {:.1} µs",
        layers, per_layer_us, lm_head_us, total_per_token_us
    );
    println!("Throughput: {:.1} tok/s", tok_s);

    let ollama_tok_s = 265.0;
    println!("\nOllama CPU reference: {:.0} tok/s", ollama_tok_s);
    println!("Gap: {:.1}x slower", ollama_tok_s / tok_s);
}

fn quantize_to_q8k(values: &[f32]) -> (Vec<f32>, Vec<i8>) {
    const QK_K: usize = 256;
    let num_sb = values.len().div_ceil(QK_K);
    let padded_len = num_sb * QK_K;

    let mut scales = Vec::with_capacity(num_sb);
    let mut quants = vec![0i8; padded_len];

    for sb in 0..num_sb {
        let start = sb * QK_K;
        let end = (start + QK_K).min(values.len());
        let chunk = &values[start..end];

        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inv_scale = if scale > 0.0 { 127.0 / amax } else { 0.0 };

        scales.push(scale);

        for (i, v) in chunk.iter().enumerate() {
            quants[start + i] = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    (scales, quants)
}
