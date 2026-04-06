//! PAR-126: Measure Rayon parallelization overhead

use rayon::prelude::*;
use realizar::quantize::{fused_q4k_q8k_dot_simd, quantize_activations_q8k_into};
use std::time::Instant;

fn main() {
    let in_dim: usize = 1536;
    let out_dim: usize = 8960;

    // Setup
    let super_blocks = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks * 144;
    let weight_data: Vec<u8> = vec![0u8; out_dim * bytes_per_row];

    let activations: Vec<f32> = vec![0.5f32; in_dim];
    let padded = in_dim.next_multiple_of(256);
    let mut q8k_scales = vec![0.0f32; padded / 256];
    let mut q8k_quants = vec![0i8; padded];
    quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants).unwrap();

    let mut output = vec![0.0f32; out_dim];

    // Benchmark sequential
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        for o in 0..out_dim {
            let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
            output[o] = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
        }
    }
    let seq_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark parallel (current)
    let start = Instant::now();
    for _ in 0..iterations {
        output
            .par_iter_mut()
            .enumerate()
            .with_min_len(128)
            .for_each(|(o, out)| {
                let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
                *out = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
            });
    }
    let par_128_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark parallel with smaller chunks
    let start = Instant::now();
    for _ in 0..iterations {
        output
            .par_iter_mut()
            .enumerate()
            .with_min_len(32)
            .for_each(|(o, out)| {
                let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
                *out = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
            });
    }
    let par_32_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark parallel with larger chunks
    let start = Instant::now();
    for _ in 0..iterations {
        output
            .par_iter_mut()
            .enumerate()
            .with_min_len(256)
            .for_each(|(o, out)| {
                let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
                *out = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
            });
    }
    let par_256_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark parallel with chunks of 512
    let start = Instant::now();
    for _ in 0..iterations {
        output
            .par_iter_mut()
            .enumerate()
            .with_min_len(512)
            .for_each(|(o, out)| {
                let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
                *out = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
            });
    }
    let par_512_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("=== Rayon Overhead Analysis (8960×1536) ===\n");
    println!("Sequential:       {:.0} µs", seq_us);
    println!(
        "Parallel (32):    {:.0} µs (overhead: {:.0} µs)",
        par_32_us,
        par_32_us - seq_us / 24.0
    );
    println!(
        "Parallel (128):   {:.0} µs (speedup: {:.1}x)",
        par_128_us,
        seq_us / par_128_us
    );
    println!(
        "Parallel (256):   {:.0} µs (speedup: {:.1}x)",
        par_256_us,
        seq_us / par_256_us
    );
    println!(
        "Parallel (512):   {:.0} µs (speedup: {:.1}x)",
        par_512_us,
        seq_us / par_512_us
    );

    // Theoretical best with 24 cores
    let theoretical_best = seq_us / 24.0;
    println!("\nTheoretical (24 cores): {:.0} µs", theoretical_best);
    println!(
        "Parallel efficiency: {:.0}%",
        theoretical_best / par_128_us * 100.0
    );

    // Per-row timing
    let per_row_us = seq_us / out_dim as f64;
    println!("\nPer-row time: {:.3} µs", per_row_us);
}
