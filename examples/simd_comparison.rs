#![allow(clippy::disallowed_methods)]
//! Benchmark: Scalar vs Trueno SIMD for load generation
use std::hint::black_box;
use std::time::Instant;
use trueno::Vector;

fn main() {
    let size = 1_048_576;
    let input_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    let input_b: Vec<f32> = (0..size).map(|i| ((i + 500) % 1000) as f32 / 1000.0).collect();
    let mut output = vec![0.0f32; size];

    println!("=== Load Generation Benchmark ===");
    println!("Problem size: {} elements ({:.1} MB)", size, size as f64 * 4.0 / 1e6);
    println!();

    // Warmup
    for _ in 0..5 {
        for i in 0..size {
            output[i] = input_a[i] * input_b[i] + 1.0;
        }
        black_box(&output);
    }

    // Scalar benchmark (FMA: a * b + c)
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            output[i] = input_a[i] * input_b[i] + 1.0;
        }
        black_box(&output);
    }
    let scalar_time = start.elapsed();
    let scalar_gflops = (size as f64 * 2.0 * iterations as f64) / scalar_time.as_secs_f64() / 1e9;
    println!("Scalar loop:    {:>8.2?} ({:>6.2} GFLOP/s)", scalar_time, scalar_gflops);

    // Trueno SIMD benchmark (mul only, then add)
    let vec_a = Vector::from_slice(&input_a);
    let vec_b = Vector::from_slice(&input_b);

    // Warmup SIMD
    for _ in 0..5 {
        let _ = black_box(vec_a.mul(&vec_b).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let result = vec_a.mul(&vec_b).unwrap();
        black_box(&result);
    }
    let simd_time = start.elapsed();
    let simd_gflops = (size as f64 * 1.0 * iterations as f64) / simd_time.as_secs_f64() / 1e9;
    println!("Trueno mul:     {:>8.2?} ({:>6.2} GFLOP/s)", simd_time, simd_gflops);

    // Trueno dot product (reduction)
    let start = Instant::now();
    for _ in 0..iterations {
        let result = vec_a.dot(&vec_b).unwrap();
        black_box(result);
    }
    let dot_time = start.elapsed();
    let dot_gflops = (size as f64 * 2.0 * iterations as f64) / dot_time.as_secs_f64() / 1e9;
    println!("Trueno dot:     {:>8.2?} ({:>6.2} GFLOP/s)", dot_time, dot_gflops);

    // Trueno add
    let start = Instant::now();
    for _ in 0..iterations {
        let result = vec_a.add(&vec_b).unwrap();
        black_box(&result);
    }
    let add_time = start.elapsed();
    let add_gflops = (size as f64 * 1.0 * iterations as f64) / add_time.as_secs_f64() / 1e9;
    println!("Trueno add:     {:>8.2?} ({:>6.2} GFLOP/s)", add_time, add_gflops);

    println!();
    println!("=== Analysis ===");
    println!(
        "Scalar vs SIMD mul speedup: {:.2}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );

    // Memory bandwidth estimate
    let bytes_per_iter = size as f64 * 4.0 * 3.0; // read a, read b, write output
    let bandwidth_gbs = (bytes_per_iter * iterations as f64) / scalar_time.as_secs_f64() / 1e9;
    println!("Estimated bandwidth: {:.1} GB/s", bandwidth_gbs);
}
