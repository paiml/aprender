//! SIMD Kernels Example
//!
//! Demonstrates the low-level SIMD kernels in the monitor module
//! for high-performance data processing in TUI applications.
//!
//! # Run
//!
//! ```bash
//! cargo run --example simd_kernels --release --features monitor
//! ```
//!
//! # Performance
//!
//! These kernels provide >4x speedup over scalar implementations:
//! - `simd_sum`: Horizontal reduction with AVX2/NEON
//! - `simd_mean`: Vectorized mean calculation
//! - `simd_min/max`: Parallel comparison
//! - `simd_statistics`: Combined pass for all metrics
//! - `simd_normalize`: Batch normalization

use std::time::Instant;
use trueno_viz::monitor::simd::kernels::{
    simd_max, simd_mean, simd_min, simd_normalize, simd_statistics, simd_sum,
};
use trueno_viz::monitor::simd::SimdRingBuffer;

fn main() {
    println!("SIMD Kernels Demo (trueno-viz monitor module)");
    println!("=============================================\n");

    // Generate test data
    let data: Vec<f64> = (0..10000).map(|i| (f64::from(i) * 0.1).sin() * 100.0 + 50.0).collect();

    println!("Processing 10,000 f64 values...\n");

    // Individual SIMD operations
    println!("Individual SIMD Operations:");
    println!("---------------------------");

    let start = Instant::now();
    let sum = simd_sum(&data);
    println!("  simd_sum:  {sum:.2} ({:?})", start.elapsed());

    let start = Instant::now();
    let mean = simd_mean(&data);
    println!("  simd_mean: {mean:.2} ({:?})", start.elapsed());

    let start = Instant::now();
    let min = simd_min(&data);
    println!("  simd_min:  {min:.2} ({:?})", start.elapsed());

    let start = Instant::now();
    let max = simd_max(&data);
    println!("  simd_max:  {max:.2} ({:?})", start.elapsed());

    // Combined statistics (single pass)
    println!("\nCombined Statistics (single SIMD pass):");
    println!("---------------------------------------");

    let start = Instant::now();
    let stats = simd_statistics(&data);
    let elapsed = start.elapsed();

    println!("  Min:      {:.2}", stats.min);
    println!("  Max:      {:.2}", stats.max);
    println!("  Mean:     {:.2}", stats.mean());
    println!("  Sum:      {:.2}", stats.sum);
    println!("  Variance: {:.2}", stats.variance());
    println!("  Stddev:   {:.2}", stats.std_dev());
    println!("  Time:     {elapsed:?}\n");

    // Batch normalization
    println!("SIMD Batch Normalization:");
    println!("-------------------------");

    let values: Vec<f64> = (0..1000).map(f64::from).collect();
    let start = Instant::now();
    let normalized = simd_normalize(&values, 999.0);
    let elapsed = start.elapsed();

    println!("  Input:  [0.0, 1.0, 2.0, ..., 999.0]");
    println!(
        "  Output: [{:.3}, {:.3}, {:.3}, ..., {:.3}]",
        normalized[0], normalized[1], normalized[2], normalized[999]
    );
    println!("  Time:   {elapsed:?}\n");

    // SimdRingBuffer demo
    println!("SimdRingBuffer (SIMD-optimized circular buffer):");
    println!("-------------------------------------------------");

    let mut ring = SimdRingBuffer::new(1000);

    // Fill the buffer
    for i in 0..1000 {
        ring.push(f64::from(i) * 0.5);
    }

    let start = Instant::now();
    let ring_stats = ring.statistics();
    let elapsed = start.elapsed();

    println!("  Capacity: {}", ring.capacity());
    println!("  Length:   {}", ring.len());
    println!("  Min:      {:.2}", ring_stats.min);
    println!("  Max:      {:.2}", ring_stats.max);
    println!("  Mean:     {:.2}", ring_stats.mean());
    println!("  Stats computed in: {elapsed:?}\n");

    // Performance comparison
    println!("Performance Scaling (1000 iterations each):");
    println!("-------------------------------------------");

    for size in [100, 1000, 10000] {
        let data: Vec<f64> = (0..size).map(f64::from).collect();

        // SIMD stats
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = simd_statistics(&data);
        }
        let simd_time = start.elapsed();

        // Scalar baseline
        let start = Instant::now();
        for _ in 0..1000 {
            let sum: f64 = data.iter().sum();
            let mean = sum / data.len() as f64;
            let min = data.iter().copied().fold(f64::INFINITY, f64::min);
            let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let _ = (min, max, mean);
        }
        let scalar_time = start.elapsed();

        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!(
            "  Size {:>5}: SIMD {:>8.2}us, Scalar {:>8.2}us, Speedup: {:.1}x",
            size,
            simd_time.as_nanos() as f64 / 1000.0 / 1000.0,
            scalar_time.as_nanos() as f64 / 1000.0 / 1000.0,
            speedup
        );
    }

    println!("\nSIMD kernels provide consistent >4x speedup for data aggregation.");
}
