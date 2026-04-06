//! Benchmark Q4K×f32 vs Q4K×Q8K matmul speedup
use realizar::quantize::{
    fused_q4k_parallel_matvec, fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
};
use std::time::Instant;

fn main() -> Result<(), realizar::RealizarError> {
    let in_dim: usize = 1536;
    let out_dim: usize = 8960;

    // Create test data
    let super_blocks = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks * 144;
    let weight_bytes = out_dim * bytes_per_row;
    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..in_dim)
        .map(|i| (i as f32 / in_dim as f32) * 2.0 - 1.0)
        .collect();

    // Warmup
    let _ = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim)?;

    let iterations = 100;

    // Q4K×f32 path
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim)?;
    }
    let f32_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Pre-quantize for Q8K path
    let padded_len = in_dim.next_multiple_of(256);
    let num_sb = padded_len / 256;
    let mut q8k_scales = vec![0.0f32; num_sb];
    let mut q8k_quants = vec![0i8; padded_len];

    // Quantize once
    quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants)?;

    // Q4K×Q8K path (pre-quantized)
    let mut output = vec![0.0f32; out_dim];
    let start = Instant::now();
    for _ in 0..iterations {
        fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        )?;
    }
    let q8k_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Quantization cost
    let start = Instant::now();
    for _ in 0..iterations {
        quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants)?;
    }
    let quant_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("=== Q4K Matmul Benchmark ({}x{}) ===", out_dim, in_dim);
    println!("Q4K×f32:          {:.0} us", f32_us);
    println!("Q4K×Q8K:          {:.0} us", q8k_us);
    println!("Q8K quantization: {:.0} us", quant_us);
    println!();
    println!("Speedup (kernel only):      {:.1}x", f32_us / q8k_us);
    println!(
        "Speedup (with 1 quant):     {:.1}x",
        f32_us / (q8k_us + quant_us)
    );
    println!(
        "Break-even point:           {} matmuls",
        (quant_us / (f32_us - q8k_us)).ceil() as i32
    );

    // Per-layer analysis (5 matmuls, 2 quantizations for Q8K)
    let f32_per_layer = 5.0 * f32_us;
    let q8k_per_layer = 5.0 * q8k_us + 2.0 * quant_us; // 2 quants: 1 for QKV, 1 for FFN up/gate
    println!();
    println!("=== Per-Layer Analysis ===");
    println!("Q4K×f32 (5 matmuls):        {:.0} us", f32_per_layer);
    println!("Q4K×Q8K (5 matmuls, 2 quant): {:.0} us", q8k_per_layer);
    println!(
        "Layer speedup:              {:.1}x",
        f32_per_layer / q8k_per_layer
    );

    // Full model (28 layers)
    let f32_full = 28.0 * f32_per_layer / 1000.0;
    let q8k_full = 28.0 * q8k_per_layer / 1000.0;
    println!();
    println!("=== Full Model (28 layers) ===");
    println!("Q4K×f32: {:.1} ms", f32_full);
    println!("Q4K×Q8K: {:.1} ms", q8k_full);
    println!("Savings: {:.1} ms", f32_full - q8k_full);

    Ok(())
}
