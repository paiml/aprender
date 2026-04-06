//! Benchmark Comparison Visualization Example (PAR-040)
//!
//! Demonstrates the 2×3 grid benchmark visualization and profiling log generation.
//!
//! Run with:
//! ```bash
//! cargo run --example bench_comparison
//! cargo run --example bench_comparison -- --log   # Generate chat-pasteable log
//! cargo run --example bench_comparison -- --compact  # One-liner output
//! ```

use realizar::bench_viz::{BenchMeasurement, BenchmarkGrid, BenchmarkRunner, ProfilingHotspot};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let show_log = args.iter().any(|a| a == "--log");
    let show_compact = args.iter().any(|a| a == "--compact");

    // Create benchmark grid with sample data
    // In real usage, these would come from actual benchmark runs
    let mut grid = BenchmarkGrid::new()
        .with_model("Qwen2.5-Coder-0.5B-Instruct", "0.5B params", "Q4_K_M")
        .with_gpu("NVIDIA RTX 4090", 24.0);

    // Row 1: GGUF format comparison
    grid.set_gguf_row(
        // APR serve GGUF - Phase 2 projected performance
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput(500.0) // Projected with PAR-036..039
            .with_ttft(7.0)
            .with_gpu(95.0, 2048.0),
        // Ollama baseline (measured)
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput(318.0)
            .with_ttft(50.0)
            .with_gpu(92.0, 1800.0),
        // llama.cpp baseline (estimated)
        BenchMeasurement::new("llama.cpp", "GGUF")
            .with_throughput(200.0)
            .with_ttft(30.0)
            .with_gpu(90.0, 1600.0),
    );

    // Row 2: APR server format comparison
    grid.set_apr_row(
        // APR serve .apr native format (best case)
        BenchMeasurement::new("APR", ".apr")
            .with_throughput(600.0) // Native format with optimizations
            .with_ttft(5.0)
            .with_gpu(96.0, 1900.0),
        // APR serve GGUF (same as row 1)
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput(500.0)
            .with_ttft(7.0)
            .with_gpu(95.0, 2048.0),
        // Ollama baseline for comparison
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput(318.0)
            .with_ttft(50.0)
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
    } else {
        // Default: show ASCII grid then log
        println!("{}", grid.render_ascii());
        println!();
        println!("{}", grid.render_profiling_log());
    }
}

/// Example of using BenchmarkRunner with actual measurements
#[allow(dead_code)]
fn example_with_runner() {
    let mut runner = BenchmarkRunner::new();
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

    runner.grid.gguf_apr = Some(
        BenchMeasurement::new("APR", "GGUF")
            .with_tokens(128, Duration::from_millis(315))
            .with_ttft(7.0),
    );

    println!("{}", runner.grid.render_profiling_log());
}
