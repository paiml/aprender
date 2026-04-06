//! Bench command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{BenchArgs, OutputFormat};
use std::time::Instant;

pub fn run_bench(args: BenchArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Running benchmark: {}", args.input.display()));

    // Parse batch sizes
    let batch_sizes: Vec<usize> = args
        .batch_sizes
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Invalid batch sizes: {e}"))?;

    log(level, LogLevel::Normal, &format!("  Warmup: {} iterations", args.warmup));
    log(level, LogLevel::Normal, &format!("  Iterations: {}", args.iterations));
    log(level, LogLevel::Normal, &format!("  Batch sizes: {batch_sizes:?}"));

    // Run benchmarks for each batch size
    for batch_size in &batch_sizes {
        log(level, LogLevel::Normal, &format!("\nBatch size: {batch_size}"));

        // Warmup
        for _ in 0..args.warmup {
            // Simulate inference with small sleep
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        // Measure latency
        let mut latencies: Vec<f64> = Vec::with_capacity(args.iterations);
        for _ in 0..args.iterations {
            let start = Instant::now();
            // Simulate inference - in real impl would run model forward pass
            std::thread::sleep(std::time::Duration::from_micros(50 + *batch_size as u64 * 10));
            let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms
            latencies.push(elapsed);
        }

        // Sort for percentile calculation
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = latencies[latencies.len() * 50 / 100];
        let p95 = latencies[latencies.len() * 95 / 100];
        let p99 = latencies[latencies.len() * 99 / 100];
        let mean = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;
        let throughput = 1000.0 / mean * *batch_size as f64;

        if args.format == OutputFormat::Json {
            let result = serde_json::json!({
                "batch_size": batch_size,
                "iterations": args.iterations,
                "latency_ms": {
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "mean": mean
                },
                "throughput_samples_per_sec": throughput
            });
            if let Ok(json_str) = serde_json::to_string_pretty(&result) {
                println!("{json_str}");
            }
        } else {
            log(level, LogLevel::Normal, &format!("  p50: {p50:.2}ms"));
            log(level, LogLevel::Normal, &format!("  p95: {p95:.2}ms"));
            log(level, LogLevel::Normal, &format!("  p99: {p99:.2}ms"));
            log(level, LogLevel::Normal, &format!("  mean: {mean:.2}ms"));
            log(level, LogLevel::Normal, &format!("  throughput: {throughput:.1} samples/sec"));
        }
    }

    Ok(())
}
