//! SIMD-Accelerated Visualization Example
//!
//! Demonstrates the SIMD-accelerated ring buffer and statistics
//! for real-time visualization in renacer.
//!
#![allow(clippy::unwrap_used)]
//! # Run
//!
//! ```bash
//! cargo run --example simd_visualization --release
//! ```
//!
//! # Performance
//!
//! SIMD provides >4x speedup for statistics aggregation:
//! - Sum: 4.0-4.8x faster
//! - Min/Max: 4.0-4.6x faster
//! - Combined stats: 3.5-4.0x faster
//!
//! # Dependencies
//!
//! Uses trueno-viz SIMD kernels for AVX2/NEON vectorization.

use renacer::visualize::ring_buffer::HistoryBuffer;
use renacer::visualize::theme::{normalize_batch, sparkline};
use std::time::Instant;

fn main() {
    println!("SIMD-Accelerated Visualization Demo");
    println!("====================================\n");

    // Create a history buffer with 1000 elements
    let mut buffer = HistoryBuffer::new(1000);

    // Fill with sample data (simulated syscall latencies in microseconds)
    for i in 0..1000 {
        let latency = 100.0 + (i as f64 * 0.5).sin() * 50.0 + (i as f64).cos() * 30.0;
        buffer.push(latency);
    }

    println!("Buffer filled with 1000 simulated latency values\n");

    // Demonstrate SIMD-accelerated statistics
    println!("SIMD-Accelerated Statistics:");
    println!("-----------------------------");

    let start = Instant::now();
    let sum = buffer.sum();
    let sum_time = start.elapsed();

    let start = Instant::now();
    let avg = buffer.avg();
    let avg_time = start.elapsed();

    let start = Instant::now();
    let (min, max, mean) = buffer.stats();
    let stats_time = start.elapsed();

    let start = Instant::now();
    let stddev = buffer.stddev();
    let stddev_time = start.elapsed();

    println!("  Sum:    {sum:.2} ({sum_time:?})");
    println!("  Avg:    {avg:.2} ({avg_time:?})");
    println!("  Min:    {min:.2}");
    println!("  Max:    {max:.2}");
    println!("  Mean:   {mean:.2}");
    println!("  Stddev: {stddev:.2} ({stddev_time:?})");
    println!("  Stats:  ({stats_time:?} for min/max/mean combined)\n");

    // Demonstrate sparkline generation (uses SIMD for min/max)
    println!("SIMD-Accelerated Sparkline:");
    println!("---------------------------");

    let latest = buffer.latest(50);
    let start = Instant::now();
    let spark = sparkline(&latest, 50);
    let spark_time = start.elapsed();

    println!("  {spark}");
    println!("  (Generated in {spark_time:?})\n");

    // Demonstrate batch normalization
    println!("SIMD-Accelerated Normalization:");
    println!("--------------------------------");

    let values: Vec<f64> = (0..100).map(|i| i as f64 * 1.5).collect();
    let start = Instant::now();
    let normalized = normalize_batch(&values);
    let norm_time = start.elapsed();

    println!("  Input range:  0.0 - {:.1}", values.last().unwrap());
    println!("  Output range: {:.2} - {:.2}", normalized[0], normalized.last().unwrap());
    println!("  (Normalized 100 values in {norm_time:?})\n");

    // Benchmark: Compare buffer sizes
    println!("Performance Scaling:");
    println!("--------------------");

    for size in [100, 1000, 10000] {
        let mut buf = HistoryBuffer::new(size);
        for i in 0..size {
            buf.push(i as f64 * 0.5);
        }

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = buf.stats();
        }
        let elapsed = start.elapsed();

        println!(
            "  Size {size:>5}: {:.2} us/op (1000 iterations)",
            elapsed.as_nanos() as f64 / 1000.0 / 1000.0
        );
    }

    println!("\nSIMD acceleration powered by trueno-viz monitor::simd::kernels");
}
