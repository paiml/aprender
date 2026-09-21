#![allow(clippy::disallowed_methods)]
//! Showcase Benchmark Example (PAR-040)
//!
//! Demonstrates the GPU inference showcase with rich visualization and PMAT verification.
//!
//! # Performance Targets
//!
//! - Point 41: ≥25% faster than llama.cpp
//! - Point 42: ≥60 tok/s minimum threshold
//! - Point 49: CV <5% consistency
//! - Target: >2x Ollama (against a measured Ollama baseline; none is recorded here)
//!
//! # Run
//!
//! ```bash
//! # Synthetic APR results, no competitor baseline (no GPU required)
//! cargo run --example showcase_benchmark
//!
//! # With custom iterations
//! cargo run --example showcase_benchmark -- --iterations 20
//!
//! # Compact output
//! cargo run --example showcase_benchmark -- --compact
//!
//! # Scientific output only
//! cargo run --example showcase_benchmark -- --scientific
//!
//! # PMAT verification report
//! cargo run --example showcase_benchmark -- --pmat
//! ```

use aprender::showcase::{
    BenchmarkResult, BenchmarkStats, PmatVerification, ProfilingCollector, ShowcaseConfig,
    ShowcaseRunner,
};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let compact = args.iter().any(|a| a == "--compact");
    let scientific = args.iter().any(|a| a == "--scientific");
    let pmat_only = args.iter().any(|a| a == "--pmat");
    let no_color = args.iter().any(|a| a == "--no-color");

    let iterations = args
        .iter()
        .position(|a| a == "--iterations")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Configure showcase
    let config = ShowcaseConfig {
        iterations,
        warmup_iterations: 3,
        gen_tokens: 128,
        colors: !no_color,
        ..Default::default()
    };

    // Create runner with model/GPU info
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("Qwen2.5-Coder-0.5B-Instruct", "0.5B params", "Q4_K_M")
        .with_gpu_info("NVIDIA RTX 4090", 24.0);

    // Synthetic APR results: this example demonstrates the renderer, it runs
    // no inference, so nothing it prints is a measurement.
    println!(
        "Synthesizing APR GGUF results ({} iterations, SIMULATED — not a measurement)...",
        iterations
    );
    let apr_gguf_results = simulate_apr_gguf_benchmark(iterations);
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_gguf_results));

    println!(
        "Synthesizing APR native results ({} iterations, SIMULATED — not a measurement)...",
        iterations
    );
    let apr_native_results = simulate_apr_native_benchmark(iterations);
    runner.record_apr_native(BenchmarkStats::from_results(apr_native_results));

    // #3773: no Ollama / llama.cpp baseline is recorded. The example used to
    // synthesize one around 318 / 200 tok/s; a competitor figure nobody measured
    // renders as UNMEASURED and judges no comparison.
    println!("Ollama / llama.cpp baselines: UNMEASURED (this example runs neither)");

    // Add profiling hotspots
    println!("Collecting profiling hotspots...");
    let mut profiler = ProfilingCollector::new();
    profiler.start();
    profiler.record("Q4K_GEMV", Duration::from_millis(150), 3584);
    profiler.record("Attention", Duration::from_millis(80), 3584);
    profiler.record("RMSNorm", Duration::from_millis(30), 7168);
    profiler.record("SwiGLU", Duration::from_millis(25), 3584);
    profiler.record("KernelLaunch", Duration::from_millis(20), 35840);

    for hotspot in profiler.into_hotspots() {
        runner.add_hotspot(hotspot);
    }

    println!();

    // Generate output based on flags
    if pmat_only {
        let verification = PmatVerification::verify(&runner);
        println!("{}", verification.to_report());
    } else if compact {
        let grid = runner.to_grid();
        println!("{}", grid.render_compact());
    } else if scientific {
        let grid = runner.to_grid();
        println!("{}", grid.render_scientific());
    } else {
        // Full report
        println!("{}", runner.render_report());
        println!();

        // PMAT verification
        let verification = PmatVerification::verify(&runner);
        println!("{}", verification.to_report());
    }
}

/// Simulate APR GGUF benchmark (Phase 2 optimizations)
fn simulate_apr_gguf_benchmark(iterations: usize) -> Vec<BenchmarkResult> {
    // Phase 2 target: 500+ tok/s with 3.28x improvement
    // Simulating realistic variance
    (0..iterations)
        .map(|i| {
            let noise = (i as f64 * 0.7).sin() * 0.03; // ~3% variance
            let base_tps = 500.0;
            let tps = base_tps * (1.0 + noise);
            let duration_ms = (128.0 / tps * 1000.0) as u64;

            BenchmarkResult::new(
                128,
                Duration::from_millis(duration_ms),
                Duration::from_millis(7),
            )
            .with_gpu_metrics(95.0, 2048.0)
        })
        .collect()
}

/// Simulate APR native (.apr) benchmark (best case with ZRAM)
fn simulate_apr_native_benchmark(iterations: usize) -> Vec<BenchmarkResult> {
    // Native format with ZRAM: ~600 tok/s target
    (0..iterations)
        .map(|i| {
            let noise = (i as f64 * 0.9).sin() * 0.03;
            let base_tps = 600.0;
            let tps = base_tps * (1.0 + noise);
            let duration_ms = (128.0 / tps * 1000.0) as u64;

            BenchmarkResult::new(
                128,
                Duration::from_millis(duration_ms),
                Duration::from_millis(5),
            )
            .with_gpu_metrics(96.0, 1900.0)
        })
        .collect()
}
