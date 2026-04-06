//! PAR-126: Chunked matmul to reduce Rayon overhead

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

    // Benchmark: process rows in explicit chunks
    let iterations = 100;
    let num_threads = 24;
    let rows_per_thread = out_dim / num_threads;

    // Manual chunking - each chunk processes multiple rows sequentially
    let start = Instant::now();
    for _ in 0..iterations {
        output
            .par_chunks_mut(rows_per_thread)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_row = chunk_idx * rows_per_thread;
                for (local_row, out) in chunk.iter_mut().enumerate() {
                    let row = start_row + local_row;
                    if row < out_dim {
                        let row_data = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                        *out = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants)
                            .unwrap_or(0.0);
                    }
                }
            });
    }
    let chunked_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark: standard par_iter approach
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

    // Benchmark: scoped threads (no Rayon)
    let start = Instant::now();
    for _ in 0..iterations {
        std::thread::scope(|s| {
            let chunks: Vec<_> = output.chunks_mut(rows_per_thread).collect();
            for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                let start_row = chunk_idx * rows_per_thread;
                let weight_data = &weight_data;
                let q8k_scales = &q8k_scales;
                let q8k_quants = &q8k_quants;
                s.spawn(move || {
                    for (local_row, out) in chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        if row < out_dim {
                            let row_data =
                                &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                            *out = fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants)
                                .unwrap_or(0.0);
                        }
                    }
                });
            }
        });
    }
    let scoped_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Sequential for reference
    let start = Instant::now();
    for _ in 0..iterations {
        for o in 0..out_dim {
            let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
            output[o] = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
        }
    }
    let seq_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("=== Chunked Matmul (8960×1536) ===\n");
    println!("Sequential:       {:.0} µs", seq_us);
    println!(
        "Rayon par_iter:   {:.0} µs (speedup: {:.1}x)",
        par_256_us,
        seq_us / par_256_us
    );
    println!(
        "Rayon chunked:    {:.0} µs (speedup: {:.1}x)",
        chunked_us,
        seq_us / chunked_us
    );
    println!(
        "Scoped threads:   {:.0} µs (speedup: {:.1}x)",
        scoped_us,
        seq_us / scoped_us
    );

    let theoretical = seq_us / 24.0;
    println!("\nTheoretical (24x): {:.0} µs", theoretical);
    println!(
        "Best efficiency:  {:.0}%",
        theoretical / chunked_us.min(scoped_us) * 100.0
    );
}
