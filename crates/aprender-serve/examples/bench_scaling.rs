//! Benchmark thread scaling to understand parallelism bottleneck

use std::time::Instant;

fn main() {
    let hidden = 1536;
    let inter = 8960;
    let super_blocks = hidden / 256;
    let bytes_per_row = super_blocks * 144;
    let weight_bytes = inter * bytes_per_row;

    println!("Thread Scaling Analysis");
    println!("========================\n");

    let weights: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let activations: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 / hidden as f32) * 2.0 - 1.0)
        .collect();

    let (q8k_scales, q8k_quants) = quantize_to_q8k(&activations);

    // Single-threaded baseline
    let mut output = vec![0.0f32; inter];
    let iters = 50;

    // Test different thread counts
    for num_threads in [1, 2, 4, 8, 16, 24, 32, 48] {
        // Set thread count
        std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string());
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok(); // Ignore if already built

        // Warm up with current thread count
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| {
            // Warmup
            for _ in 0..3 {
                let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
                    &weights,
                    &q8k_scales,
                    &q8k_quants,
                    hidden,
                    inter,
                    &mut output,
                );
            }

            // Benchmark
            let start = Instant::now();
            for _ in 0..iters {
                let _ = realizar::quantize::fused_q4k_q8k_parallel_matvec_into(
                    &weights,
                    &q8k_scales,
                    &q8k_quants,
                    hidden,
                    inter,
                    &mut output,
                );
            }
            let elapsed = start.elapsed();
            let per_matmul_us = elapsed.as_micros() as f64 / iters as f64;

            let flops = 2 * hidden * inter;
            let gflops = flops as f64 / per_matmul_us / 1000.0;

            println!(
                "{:2} threads: {:6.1} µs  ({:5.1} GFLOPS, efficiency: {:5.1}%)",
                num_threads,
                per_matmul_us,
                gflops,
                gflops / num_threads as f64 / 25.0 * 100.0
            ); // 25 GFLOPS baseline per thread
        });
    }
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
