//! Benchmark different chunk sizes for parallel matmul
//! PAR-126: Find optimal chunk size for NUMA cache behavior

use std::time::Instant;

fn main() {
    println!("Chunk Size Scaling Analysis");
    println!("============================\n");

    // Limit to 16 threads (NUMA optimal per bench_scaling)
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .ok();

    let hidden = 1536;
    let inter = 8960;
    let super_blocks = hidden / 256;
    let bytes_per_row = super_blocks * 144;
    let weight_bytes = inter * bytes_per_row;

    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();

    let (q8k_scales, q8k_quants) = quantize_to_q8k(&activations);
    let mut output = vec![0.0f32; inter];

    // Test different chunk sizes
    for chunk_size in [8, 16, 32, 64, 128, 256, 512, 1024] {
        // Warmup
        for _ in 0..3 {
            bench_parallel_matmul_chunked(
                &weights,
                &q8k_scales,
                &q8k_quants,
                hidden,
                inter,
                &mut output,
                chunk_size,
            );
        }

        let iters = 50;
        let start = Instant::now();
        for _ in 0..iters {
            bench_parallel_matmul_chunked(
                &weights,
                &q8k_scales,
                &q8k_quants,
                hidden,
                inter,
                &mut output,
                chunk_size,
            );
        }
        let us = start.elapsed().as_micros() as f64 / iters as f64;
        let gflops = 2.0 * hidden as f64 * inter as f64 / us / 1000.0;

        println!(
            "Chunk {:4}: {:6.1} µs ({:5.1} GFLOPS)",
            chunk_size, us, gflops
        );
    }
}

fn bench_parallel_matmul_chunked(
    weight_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
    chunk_size: usize,
) {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;

    output[..out_dim]
        .par_iter_mut()
        .enumerate()
        .with_min_len(chunk_size)
        .for_each(|(o, out)| {
            let row_start = o * bytes_per_row;
            let row_data = &weight_data[row_start..row_start + bytes_per_row];
            *out = realizar::quantize::fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants)
                .unwrap_or(0.0);
        });
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
