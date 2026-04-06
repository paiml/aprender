//! Benchmark Example
//!
//! Demonstrates programmatic latency benchmarking.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example cli_bench
//! ```
//!
//! Or use the CLI:
//! ```bash
//! entrenar bench examples/yaml/latency.yaml --warmup 5 --iterations 100
//! ```

use std::time::Instant;

fn main() {
    println!("Latency Benchmark Example");
    println!("=========================\n");

    let batch_sizes = vec![1, 8, 32, 64];
    let warmup = 5;
    let iterations = 100;

    for batch_size in &batch_sizes {
        println!("Batch size: {batch_size}");

        // Warmup iterations
        for _ in 0..warmup {
            simulate_inference(*batch_size);
        }

        // Measure latency
        let mut latencies: Vec<f64> = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            simulate_inference(*batch_size);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms
            latencies.push(elapsed);
        }

        // Sort for percentile calculation
        latencies.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));

        let p50 = latencies[latencies.len() * 50 / 100];
        let p95 = latencies[latencies.len() * 95 / 100];
        let p99 = latencies[latencies.len() * 99 / 100];
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let throughput = 1000.0 / mean * *batch_size as f64;

        println!("  p50: {p50:.2}ms");
        println!("  p95: {p95:.2}ms");
        println!("  p99: {p99:.2}ms");
        println!("  mean: {mean:.2}ms");
        println!("  throughput: {throughput:.1} samples/sec\n");
    }
}

/// Simulate inference with batch-size-dependent latency
fn simulate_inference(batch_size: usize) {
    // Simulate work proportional to batch size
    std::thread::sleep(std::time::Duration::from_micros(50 + batch_size as u64 * 10));
}
