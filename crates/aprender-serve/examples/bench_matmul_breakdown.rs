//! PAR-126: Breakdown of matmul timing per matrix size

use realizar::quantize::{fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into};
use std::time::Instant;

fn bench_matmul(name: &str, in_dim: usize, out_dim: usize) -> f64 {
    let super_blocks = in_dim.div_ceil(256);
    let weight_data: Vec<u8> = vec![0u8; out_dim * super_blocks * 144];

    let activations: Vec<f32> = vec![0.5f32; in_dim];
    let padded = in_dim.next_multiple_of(256);
    let mut q8k_scales = vec![0.0f32; padded / 256];
    let mut q8k_quants = vec![0i8; padded];
    quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants).unwrap();

    let mut output = vec![0.0f32; out_dim];

    // Warmup
    for _ in 0..10 {
        fused_q4k_q8k_parallel_matvec_into(
            &weight_data,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        )
        .unwrap();
    }

    // Measure
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q4k_q8k_parallel_matvec_into(
            &weight_data,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        )
        .unwrap();
    }
    let elapsed_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("{}: {}×{} = {:.0} µs", name, out_dim, in_dim, elapsed_us);
    elapsed_us
}

fn main() {
    println!("=== PAR-126: Per-Matmul Breakdown ===\n");

    // Qwen2.5-1.5B dimensions
    let hidden = 1536;
    let intermediate = 8960;
    let kv_dim = 256;

    let q_time = bench_matmul("Q", hidden, hidden);
    let k_time = bench_matmul("K", hidden, kv_dim);
    let v_time = bench_matmul("V", hidden, kv_dim);
    let attn_out_time = bench_matmul("Attn Out", hidden, hidden);
    let ffn_up_time = bench_matmul("FFN Up", hidden, intermediate);
    let ffn_gate_time = bench_matmul("FFN Gate", hidden, intermediate);
    let ffn_down_time = bench_matmul("FFN Down", intermediate, hidden);

    let layer_time =
        q_time + k_time + v_time + attn_out_time + ffn_up_time + ffn_gate_time + ffn_down_time;
    let total_time = layer_time * 28.0;

    println!("\n=== Summary ===");
    println!(
        "Per-layer: {:.0} µs = {:.2} ms",
        layer_time,
        layer_time / 1000.0
    );
    println!(
        "28 layers: {:.0} µs = {:.1} ms",
        total_time,
        total_time / 1000.0
    );

    // Breakdown
    println!("\n=== Breakdown ===");
    println!(
        "Q+K+V+Attn:     {:.0} µs ({:.0}%)",
        q_time + k_time + v_time + attn_out_time,
        (q_time + k_time + v_time + attn_out_time) / layer_time * 100.0
    );
    println!(
        "FFN (up+gate):  {:.0} µs ({:.0}%)",
        ffn_up_time + ffn_gate_time,
        (ffn_up_time + ffn_gate_time) / layer_time * 100.0
    );
    println!(
        "FFN (down):     {:.0} µs ({:.0}%)",
        ffn_down_time,
        ffn_down_time / layer_time * 100.0
    );

    // vs Ollama
    let ollama_total_ms = 14.05;
    println!("\n=== vs Ollama ===");
    println!("realizar matmuls: {:.1} ms", total_time / 1000.0);
    println!("Ollama total:     {:.1} ms", ollama_total_ms);
    println!("Gap: {:.2}x", total_time / 1000.0 / ollama_total_ms);
}
