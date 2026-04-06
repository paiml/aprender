//! Check CPU utilization during parallel matmul

use rayon::prelude::*;
use realizar::quantize::{fused_q4k_q8k_dot_simd, quantize_activations_q8k_into};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn main() {
    let in_dim: usize = 1536;
    let out_dim: usize = 8960;

    let super_blocks = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks * 144;
    let weight_data: Vec<u8> = vec![0u8; out_dim * bytes_per_row];

    let activations: Vec<f32> = vec![0.5f32; in_dim];
    let padded = in_dim.next_multiple_of(256);
    let mut q8k_scales = vec![0.0f32; padded / 256];
    let mut q8k_quants = vec![0i8; padded];
    quantize_activations_q8k_into(&activations, &mut q8k_scales, &mut q8k_quants).unwrap();

    let mut output = vec![0.0f32; out_dim];

    // Count active threads
    static ACTIVE_THREADS: AtomicUsize = AtomicUsize::new(0);
    static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);

    // Run once with thread counting
    output
        .par_iter_mut()
        .enumerate()
        .with_min_len(256)
        .for_each(|(o, out)| {
            let current = ACTIVE_THREADS.fetch_add(1, Ordering::Relaxed) + 1;
            let _ = MAX_THREADS.fetch_max(current, Ordering::Relaxed);

            let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
            *out = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);

            ACTIVE_THREADS.fetch_sub(1, Ordering::Relaxed);
        });

    println!(
        "Max concurrent threads: {}",
        MAX_THREADS.load(Ordering::Relaxed)
    );
    println!("Rayon thread pool size: {}", rayon::current_num_threads());

    // Check if issue is memory bandwidth by doing sequential reads
    let iterations = 100;
    let start = Instant::now();
    let mut sum: f32 = 0.0;
    for _ in 0..iterations {
        for o in 0..out_dim {
            let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
            // Just read, don't compute
            sum += row_data[0] as f32;
        }
    }
    let read_only_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "\nSequential read-only: {:.0} µs (sum: {})",
        read_only_us, sum
    );

    // Parallel read-only
    let counter = AtomicUsize::new(0);
    let start = Instant::now();
    for _ in 0..iterations {
        output
            .par_iter_mut()
            .enumerate()
            .with_min_len(256)
            .for_each(|(o, out)| {
                let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
                counter.fetch_add(row_data[0] as usize, Ordering::Relaxed);
                *out = 0.0;
            });
    }
    let par_read_only_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!(
        "Parallel read-only:   {:.0} µs (speedup: {:.1}x)",
        par_read_only_us,
        read_only_us / par_read_only_us
    );

    // Check: is the kernel itself parallel-friendly?
    // Time a single kernel call many times
    let row_data = &weight_data[0..bytes_per_row];
    let iterations = 100000;
    let start = Instant::now();
    let mut total: f32 = 0.0;
    for _ in 0..iterations {
        total += fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
    }
    let kernel_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
    println!(
        "\nSingle kernel: {:.0} ns ({:.3} µs), result: {}",
        kernel_ns,
        kernel_ns / 1000.0,
        total
    );

    // Estimated parallel time if perfectly parallel
    let total_kernel_ns = kernel_ns * out_dim as f64;
    let perfect_parallel_ns = total_kernel_ns / 24.0;
    println!("Total kernel work: {:.0} µs", total_kernel_ns / 1000.0);
    println!(
        "Perfect parallel (24 cores): {:.0} µs",
        perfect_parallel_ns / 1000.0
    );
}
