//! Micro-profiling: measure individual operations in forward pass

use realizar::quantize::{
    fused_q4k_parallel_matvec_into, fused_q4k_q8k_parallel_matvec_into,
    quantize_activations_q8k_into, QK_K,
};
use std::time::Instant;

fn main() {
    // Qwen2.5-1.5B dimensions
    let hidden_dim: usize = 1536;
    let intermediate_dim: usize = 8960;

    // Create test data
    const Q4K_SUPER_BLOCK_BYTES: usize = 144;
    let super_blocks = hidden_dim.div_ceil(QK_K);
    let _q4k_bytes = super_blocks * Q4K_SUPER_BLOCK_BYTES;

    // Q4K weight matrix for FFN down (intermediate_dim × hidden_dim)
    let _ffn_down_bytes =
        intermediate_dim.div_ceil(QK_K) * QK_K / QK_K * intermediate_dim.div_ceil(QK_K);
    let ffn_down_rows = hidden_dim;
    let ffn_down_cols = intermediate_dim;
    let ffn_down_super_blocks = ffn_down_cols.div_ceil(QK_K);
    let ffn_down_weight: Vec<u8> =
        vec![0x55; ffn_down_rows * ffn_down_super_blocks * Q4K_SUPER_BLOCK_BYTES];

    // Activations
    let activations_f32: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    // Q8K quantized activations
    let padded_len = intermediate_dim.next_multiple_of(256);
    let num_sb = padded_len / 256;
    let mut q8k_scales = vec![0.0f32; num_sb];
    let mut q8k_quants = vec![0i8; padded_len];

    // Output buffer
    let mut output = vec![0.0f32; hidden_dim];

    let iters = 100;

    println!(
        "=== Micro-profiling FFN Down ({}x{}) ===\n",
        hidden_dim, intermediate_dim
    );

    // 1. Measure Q8K quantization time
    let start = Instant::now();
    for _ in 0..iters {
        quantize_activations_q8k_into(&activations_f32, &mut q8k_scales, &mut q8k_quants).unwrap();
    }
    let q8k_time = start.elapsed();
    println!(
        "Q8K quantization: {:>7.1} µs/iter ({:.2}%)",
        q8k_time.as_micros() as f64 / iters as f64,
        0.0
    );

    // 2. Measure Q4K×Q8K matmul time
    let start = Instant::now();
    for _ in 0..iters {
        fused_q4k_q8k_parallel_matvec_into(
            &ffn_down_weight,
            &q8k_scales,
            &q8k_quants,
            intermediate_dim,
            hidden_dim,
            &mut output,
        )
        .unwrap();
    }
    let q4k_q8k_time = start.elapsed();
    let q4k_q8k_us = q4k_q8k_time.as_micros() as f64 / iters as f64;
    println!("Q4K×Q8K matmul:   {:>7.1} µs/iter", q4k_q8k_us);

    // 3. Measure Q4K×f32 matmul time (for comparison)
    let start = Instant::now();
    for _ in 0..iters {
        fused_q4k_parallel_matvec_into(
            &ffn_down_weight,
            &activations_f32,
            intermediate_dim,
            hidden_dim,
            &mut output,
        )
        .unwrap();
    }
    let q4k_f32_time = start.elapsed();
    let q4k_f32_us = q4k_f32_time.as_micros() as f64 / iters as f64;
    println!("Q4K×f32 matmul:   {:>7.1} µs/iter", q4k_f32_us);

    println!("\nQ8K speedup: {:.2}x", q4k_f32_us / q4k_q8k_us);

    // 4. Calculate theoretical throughput
    let flops = 2.0 * hidden_dim as f64 * intermediate_dim as f64;
    let gflops_q8k = flops / (q4k_q8k_us * 1000.0);
    let gflops_f32 = flops / (q4k_f32_us * 1000.0);

    println!("\n=== Throughput ===");
    println!("Q4K×Q8K: {:.1} GFLOP/s", gflops_q8k);
    println!("Q4K×f32: {:.1} GFLOP/s", gflops_f32);

    // 5. Memory bandwidth utilization
    let weight_bytes = ffn_down_weight.len() as f64;
    let activation_bytes = intermediate_dim as f64 * 4.0; // f32
    let q8k_activation_bytes = intermediate_dim as f64 * 1.0; // i8

    let total_bytes_f32 = weight_bytes + activation_bytes;
    let total_bytes_q8k = weight_bytes + q8k_activation_bytes;

    let bw_f32 = total_bytes_f32 / (q4k_f32_us * 1000.0);
    let bw_q8k = total_bytes_q8k / (q4k_q8k_us * 1000.0);

    println!("\n=== Memory Bandwidth ===");
    println!("Q4K×f32 effective: {:.1} GB/s", bw_f32);
    println!("Q4K×Q8K effective: {:.1} GB/s", bw_q8k);

    // 6. Arithmetic intensity
    let ai_f32 = flops / total_bytes_f32;
    let ai_q8k = flops / total_bytes_q8k;

    println!("\n=== Arithmetic Intensity ===");
    println!("Q4K×f32: {:.2} FLOP/byte", ai_f32);
    println!("Q4K×Q8K: {:.2} FLOP/byte", ai_q8k);

    // 7. Estimate full forward pass
    // Per layer: QKV (3×hidden²) + attn_out (hidden²) + gate (hidden×inter) + up (hidden×inter) + down (inter×hidden)
    let qkv_flops = 3.0 * (hidden_dim as f64).powi(2) * 2.0;
    let attn_out_flops = (hidden_dim as f64).powi(2) * 2.0;
    let ffn_gate_up_flops = 2.0 * hidden_dim as f64 * intermediate_dim as f64 * 2.0;
    let ffn_down_flops = intermediate_dim as f64 * hidden_dim as f64 * 2.0;
    let layer_flops = qkv_flops + attn_out_flops + ffn_gate_up_flops + ffn_down_flops;

    let num_layers = 28;
    let total_flops = layer_flops * num_layers as f64;

    // Estimate time based on FFN down throughput (dominant operation)
    let estimated_time_us = (total_flops / gflops_q8k) / 1000.0;
    let estimated_tok_s = 1_000_000.0 / estimated_time_us;

    println!("\n=== Estimated Full Forward Pass ===");
    println!("Total FLOPs per token: {:.2}B", total_flops / 1e9);
    println!(
        "If all matmuls at Q4K×Q8K speed: {:.1} µs",
        estimated_time_us
    );
    println!("Estimated throughput: {:.0} tok/s", estimated_tok_s);
    println!("\nActual measured: ~15-19 tok/s");
    println!(
        "Gap: {:.1}x slower than matmul-limited",
        estimated_tok_s / 17.0
    );
}
