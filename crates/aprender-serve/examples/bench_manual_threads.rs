//! PAR-126: Manual thread pool vs Rayon

use realizar::quantize::{fused_q4k_q8k_dot_simd, quantize_activations_q8k_into};
use std::thread;
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

    let iterations = 100;

    // Sequential baseline
    let start = Instant::now();
    for _ in 0..iterations {
        for o in 0..out_dim {
            let row_data = &weight_data[o * bytes_per_row..(o + 1) * bytes_per_row];
            output[o] = fused_q4k_q8k_dot_simd(row_data, &q8k_scales, &q8k_quants).unwrap_or(0.0);
        }
    }
    let seq_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Manual thread::scope with exactly N threads
    let num_threads = 12; // Try fewer threads to reduce contention
    let rows_per_thread = out_dim.div_ceil(num_threads);

    let start = Instant::now();
    for _ in 0..iterations {
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_threads);
            for t in 0..num_threads {
                let start_row = t * rows_per_thread;
                let end_row = ((t + 1) * rows_per_thread).min(out_dim);
                let weight_data = &weight_data;
                let q8k_scales = &q8k_scales;
                let q8k_quants = &q8k_quants;
                // Get mutable slice
                let output_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        output.as_mut_ptr().add(start_row),
                        end_row - start_row,
                    )
                };
                handles.push(s.spawn(move || {
                    for (i, out) in output_slice.iter_mut().enumerate() {
                        let row = start_row + i;
                        let row_data = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                        *out =
                            fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0);
                    }
                }));
            }
            for h in handles {
                let _ = h.join();
            }
        });
    }
    let manual_12_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Try with 24 threads
    let num_threads = 24;
    let rows_per_thread = out_dim.div_ceil(num_threads);

    let start = Instant::now();
    for _ in 0..iterations {
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_threads);
            for t in 0..num_threads {
                let start_row = t * rows_per_thread;
                let end_row = ((t + 1) * rows_per_thread).min(out_dim);
                let weight_data = &weight_data;
                let q8k_scales = &q8k_scales;
                let q8k_quants = &q8k_quants;
                let output_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        output.as_mut_ptr().add(start_row),
                        end_row - start_row,
                    )
                };
                handles.push(s.spawn(move || {
                    for (i, out) in output_slice.iter_mut().enumerate() {
                        let row = start_row + i;
                        let row_data = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                        *out =
                            fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0);
                    }
                }));
            }
        });
    }
    let manual_24_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Try with 6 threads
    let num_threads = 6;
    let rows_per_thread = out_dim.div_ceil(num_threads);

    let start = Instant::now();
    for _ in 0..iterations {
        thread::scope(|s| {
            for t in 0..num_threads {
                let start_row = t * rows_per_thread;
                let end_row = ((t + 1) * rows_per_thread).min(out_dim);
                let weight_data = &weight_data;
                let q8k_scales = &q8k_scales;
                let q8k_quants = &q8k_quants;
                let output_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        output.as_mut_ptr().add(start_row),
                        end_row - start_row,
                    )
                };
                s.spawn(move || {
                    for (i, out) in output_slice.iter_mut().enumerate() {
                        let row = start_row + i;
                        let row_data = &weight_data[row * bytes_per_row..(row + 1) * bytes_per_row];
                        *out =
                            fused_q4k_q8k_dot_simd(row_data, q8k_scales, q8k_quants).unwrap_or(0.0);
                    }
                });
            }
        });
    }
    let manual_6_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("=== Manual Thread Pool (8960×1536) ===\n");
    println!("Sequential:     {:.0} µs", seq_us);
    println!(
        "Manual 6 thr:   {:.0} µs (speedup: {:.1}x)",
        manual_6_us,
        seq_us / manual_6_us
    );
    println!(
        "Manual 12 thr:  {:.0} µs (speedup: {:.1}x)",
        manual_12_us,
        seq_us / manual_12_us
    );
    println!(
        "Manual 24 thr:  {:.0} µs (speedup: {:.1}x)",
        manual_24_us,
        seq_us / manual_24_us
    );

    println!("\nTheoretical (24x): {:.0} µs", seq_us / 24.0);
}
