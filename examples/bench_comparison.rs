//! Benchmark Comparison Visualization Example (PAR-040)
//!
//! Demonstrates the 2×3 grid benchmark visualization with:
//! - Rich terminal colors
//! - Multi-iteration scientific statistics
//! - Profiling hotspot log generation
//!
//! Run with:
//! ```bash
//! cargo run --example bench_comparison
//! cargo run --example bench_comparison -- --log       # Chat-pasteable log only
//! cargo run --example bench_comparison -- --compact   # One-liner output
//! cargo run --example bench_comparison -- --scientific # Criterion-style output
//! cargo run --example bench_comparison -- --no-color  # Disable colors
//! cargo run --example bench_comparison -- --iterations 20  # More iterations
//! ```

use aprender::bench_viz::{
    BenchConfig, BenchMeasurement, BenchmarkGrid, BenchmarkRunner, ProfilingHotspot,
};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let show_log = args.iter().any(|a| a == "--log");
    let show_compact = args.iter().any(|a| a == "--compact");
    let show_scientific = args.iter().any(|a| a == "--scientific");
    let no_color = args.iter().any(|a| a == "--no-color");

    // Parse iterations flag
    let iterations = args
        .iter()
        .position(|a| a == "--iterations")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Configure benchmark
    let config = BenchConfig {
        iterations,
        warmup_iterations: 3,
        outlier_threshold: 2.0,
        colors: !no_color,
        confidence_level: 0.95,
    };

    // Create benchmark grid with sample data
    // In real usage, these would come from actual benchmark runs
    let mut grid = BenchmarkGrid::new()
        .with_config(config.clone())
        .with_model("Qwen2.5-Coder-0.5B-Instruct", "0.5B params", "Q4_K_M")
        .with_gpu("NVIDIA RTX 4090", 24.0);

    // Row 1: GGUF format comparison with multi-iteration data
    // Simulate realistic variation in measurements
    grid.set_gguf_row(
        // APR serve GGUF - Phase 2 projected performance with realistic variance
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput_samples(generate_samples(500.0, 15.0, iterations))
            .with_ttft_samples(generate_samples(7.0, 1.5, iterations))
            .with_gpu(95.0, 2048.0),
        // Ollama baseline (measured) with realistic variance
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput_samples(generate_samples(318.0, 12.0, iterations))
            .with_ttft_samples(generate_samples(50.0, 8.0, iterations))
            .with_gpu(92.0, 1800.0),
        // llama.cpp baseline (estimated) with realistic variance
        BenchMeasurement::new("llama.cpp", "GGUF")
            .with_throughput_samples(generate_samples(200.0, 10.0, iterations))
            .with_ttft_samples(generate_samples(30.0, 5.0, iterations))
            .with_gpu(90.0, 1600.0),
    );

    // Row 2: APR server format comparison
    grid.set_apr_row(
        // APR serve .apr native format (best case)
        BenchMeasurement::new("APR", ".apr")
            .with_throughput_samples(generate_samples(600.0, 18.0, iterations))
            .with_ttft_samples(generate_samples(5.0, 1.0, iterations))
            .with_gpu(96.0, 1900.0),
        // APR serve GGUF (same as row 1)
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput_samples(generate_samples(500.0, 15.0, iterations))
            .with_ttft_samples(generate_samples(7.0, 1.5, iterations))
            .with_gpu(95.0, 2048.0),
        // Ollama baseline for comparison
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput_samples(generate_samples(318.0, 12.0, iterations))
            .with_ttft_samples(generate_samples(50.0, 8.0, iterations))
            .with_gpu(92.0, 1800.0),
    );

    // Add profiling hotspots
    grid.add_hotspot(ProfilingHotspot {
        component: "Q4K_GEMV".to_string(),
        time: Duration::from_millis(150),
        percentage: 42.5,
        call_count: 28 * 128, // layers × tokens
        avg_per_call: Duration::from_micros(42),
        explanation: "Matrix ops dominate (42.5%) - expected for transformer inference".to_string(),
        is_expected: true,
    });

    grid.add_hotspot(ProfilingHotspot {
        component: "Attention".to_string(),
        time: Duration::from_millis(80),
        percentage: 22.7,
        call_count: 28 * 128,
        avg_per_call: Duration::from_micros(22),
        explanation: "Attention at 22.7% - normal for autoregressive decoding".to_string(),
        is_expected: true,
    });

    grid.add_hotspot(ProfilingHotspot {
        component: "RMSNorm".to_string(),
        time: Duration::from_millis(30),
        percentage: 8.5,
        call_count: 28 * 2 * 128, // layers × 2 norms × tokens
        avg_per_call: Duration::from_micros(4),
        explanation: "Normalization within normal range".to_string(),
        is_expected: true,
    });

    grid.add_hotspot(ProfilingHotspot {
        component: "KernelLaunch".to_string(),
        time: Duration::from_millis(25),
        percentage: 7.1,
        call_count: 28 * 10 * 128, // layers × kernels × tokens
        avg_per_call: Duration::from_nanos(700),
        explanation: "Kernel launch overhead - consider CUDA graphs or megakernels".to_string(),
        is_expected: false,
    });

    // Output based on flags
    if show_compact {
        println!("{}", grid.render_compact());
    } else if show_log {
        println!("{}", grid.render_profiling_log());
    } else if show_scientific {
        println!("{}", grid.render_scientific());
    } else {
        // Default: show rich colored grid, scientific output, then profiling log
        println!("{}", grid.render());
        println!();
        println!("{}", grid.render_scientific());
        println!();
        println!("{}", grid.render_profiling_log());
    }
}

/// Generate realistic sample data with normal-like distribution
fn generate_samples(mean: f64, std_dev: f64, count: usize) -> Vec<f64> {
    // Simple pseudo-random generator for reproducible demo
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        // Generate deterministic "noise" using sin waves
        let noise = (i as f64 * 0.7).sin() * 0.5
            + (i as f64 * 1.3).sin() * 0.3
            + (i as f64 * 2.1).sin() * 0.2;
        let value = mean + noise * std_dev;
        samples.push(value.max(0.0)); // Ensure non-negative
    }
    samples
}

/// Example of using BenchmarkRunner with actual measurements
#[allow(dead_code)]
fn example_with_runner() {
    let config = BenchConfig {
        iterations: 10,
        warmup_iterations: 3,
        colors: true,
        ..Default::default()
    };

    let mut runner = BenchmarkRunner::with_config(config);
    runner.start();

    // Simulate component timings (in real usage, measure actual operations)
    runner.record_component("Q4K_GEMV", Duration::from_millis(150), 3584);
    runner.record_component("Attention", Duration::from_millis(80), 3584);
    runner.record_component("RMSNorm", Duration::from_millis(30), 7168);
    runner.record_component("Softmax", Duration::from_millis(15), 3584);
    runner.record_component("KernelLaunch", Duration::from_millis(25), 35840);
    runner.record_component("Embedding", Duration::from_millis(5), 128);
    runner.record_component("Sampling", Duration::from_millis(10), 128);

    // Compute hotspots
    runner.finalize();

    // Set up grid
    runner.grid = runner
        .grid
        .with_model("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
        .with_gpu("RTX 4090", 24.0);

    // Example of measuring with iterations
    let measurement = runner.measure_iterations("APR GGUF", || {
        // Simulate inference
        std::thread::sleep(Duration::from_micros(100));
        (128, Duration::from_millis(250), 7.0)
    });

    runner.grid.gguf_apr = Some(measurement);

    println!("{}", runner.grid.render());
    println!("{}", runner.grid.render_profiling_log());
}
