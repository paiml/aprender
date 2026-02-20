//! Benchmark logic for showcase demo
//!
//! Extracted from monolithic showcase.rs (PMAT-201)

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use super::types::{Baseline, BenchMeasurement, BenchmarkComparison, ModelTier, ShowcaseConfig};

/// Export benchmark results to JSON or CSV (Point 85)
pub(super) fn export_benchmark_results(
    bench: &BenchmarkComparison,
    config: &ShowcaseConfig,
) -> Result<()> {
    match config.export_format {
        super::types::ExportFormat::None => Ok(()),
        super::types::ExportFormat::Json => {
            let path = config
                .export_path
                .clone()
                .unwrap_or_else(|| config.model_dir.join("benchmark-results.json"));

            let json = serde_json::to_string_pretty(bench).map_err(|e| {
                CliError::ValidationFailed(format!("JSON serialization failed: {e}"))
            })?;

            std::fs::write(&path, &json)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to write JSON: {e}")))?;

            println!(
                "{} Benchmark results exported to {} ({} bytes)",
                "✓".green(),
                path.display(),
                json.len()
            );
            Ok(())
        }
        super::types::ExportFormat::Csv => {
            let path = config
                .export_path
                .clone()
                .unwrap_or_else(|| config.model_dir.join("benchmark-results.csv"));

            let csv = format_benchmark_csv(bench);

            std::fs::write(&path, &csv)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to write CSV: {e}")))?;

            println!(
                "{} Benchmark results exported to {} ({} bytes)",
                "✓".green(),
                path.display(),
                csv.len()
            );
            Ok(())
        }
    }
}

/// Format benchmark results as CSV
pub(super) fn format_benchmark_csv(bench: &BenchmarkComparison) -> String {
    use std::fmt::Write;

    let mut csv = String::new();

    // Header
    csv.push_str("system,tokens_per_sec,ttft_ms,speedup_pct,stddev,runs\n");

    // APR row
    let _ = writeln!(
        csv,
        "APR,{:.2},{:.2},,{:.2},{}",
        bench.apr_tps, bench.apr_ttft_ms, bench.apr_tps_stddev, bench.runs
    );

    // Baseline rows (llama.cpp and Ollama follow the same CSV format)
    let baselines: &[(&str, Option<f64>, Option<f64>, Option<f64>)] = &[
        (
            "llama.cpp",
            bench.llama_cpp_tps,
            bench.llama_cpp_ttft_ms,
            bench.speedup_vs_llama,
        ),
        (
            "Ollama",
            bench.ollama_tps,
            bench.ollama_ttft_ms,
            bench.speedup_vs_ollama,
        ),
    ];
    for &(name, tps_opt, ttft_opt, speedup_opt) in baselines {
        if let Some(tps) = tps_opt {
            let ttft = ttft_opt.unwrap_or(0.0);
            let speedup = speedup_opt.map_or(String::new(), |s| format!("{s:.2}"));
            let _ = writeln!(csv, "{name},{tps:.2},{ttft:.2},{speedup},N/A,N/A");
        }
    }

    csv
}

/// Run baseline benchmarks (llama.cpp, Ollama) and build a `BenchmarkComparison`.
///
/// Shared by both the real-inference and simulated benchmark paths.
fn build_comparison(
    apr_tps: f64,
    apr_ttft_ms: f64,
    apr_tps_stddev: f64,
    runs: usize,
    config: &ShowcaseConfig,
) -> BenchmarkComparison {
    let llama_results = if config.baselines.contains(&Baseline::LlamaCpp) {
        println!();
        println!("{}", "Running llama.cpp benchmark...".yellow());
        run_llama_cpp_bench(config).ok()
    } else {
        None
    };

    let ollama_results = if config.baselines.contains(&Baseline::Ollama) {
        println!();
        println!("{}", "Running Ollama benchmark...".yellow());
        run_ollama_bench(config).ok()
    } else {
        None
    };

    let speedup_vs_llama = llama_results.map(|(tps, _)| ((apr_tps - tps) / tps) * 100.0);
    let speedup_vs_ollama = ollama_results.map(|(tps, _)| ((apr_tps - tps) / tps) * 100.0);

    BenchmarkComparison {
        apr_tps,
        apr_ttft_ms,
        apr_tps_stddev,
        runs,
        llama_cpp_tps: llama_results.map(|(tps, _)| tps),
        llama_cpp_ttft_ms: llama_results.map(|(_, ttft)| ttft),
        ollama_tps: ollama_results.map(|(tps, _)| tps),
        ollama_ttft_ms: ollama_results.map(|(_, ttft)| ttft),
        speedup_vs_llama,
        speedup_vs_ollama,
    }
}

/// Step E: Benchmark Comparison with real measurements
#[cfg(feature = "inference")]
pub(super) fn run_benchmark(config: &ShowcaseConfig) -> Result<BenchmarkComparison> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    println!();
    println!("{}", "═══ Step E: Performance Benchmark ═══".cyan().bold());
    println!();

    println!("Benchmark configuration:");
    println!(
        "  Runs: {} (per Hoefler & Belli 2015)",
        config.bench_runs.max(5)
    );
    println!("  Warmup: 5 iterations");
    println!("  Baselines: {:?}", config.baselines);
    println!(
        "  Backend: {}",
        if config.gpu { "GPU (CUDA)" } else { "CPU" }
    );
    println!();

    // Load model for APR benchmark - use tier-specific path
    let gguf_path = config.model_dir.join(config.tier.gguf_filename());
    if !gguf_path.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Model not found: {}. Run 'apr showcase --step import' first.",
            gguf_path.display()
        )));
    }
    println!(
        "Loading model for benchmark: {} ({})",
        config.tier.gguf_filename(),
        config.tier.params()
    );

    let mapped = MappedGGUFModel::from_path(&gguf_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    println!("{} Model loaded", "✓".green());

    // APR/GGUF benchmark - use GPU if requested
    println!();
    let apr_results = if config.gpu {
        println!("{}", "Running APR benchmark (GPU)...".yellow());
        // OwnedQuantizedModelCuda::new takes ownership, device 0 for first GPU
        match OwnedQuantizedModelCuda::new(model, 0) {
            Ok(mut cuda_model) => {
                println!("{} CUDA model created", "✓".green());
                run_real_benchmark_cuda(&mut cuda_model, &mapped, config)?
            }
            Err(e) => {
                println!(
                    "{} CUDA unavailable ({}), falling back to CPU",
                    "⚠".yellow(),
                    e
                );
                // Reload model since it was consumed by CUDA attempt
                let model = OwnedQuantizedModel::from_mapped(&mapped)
                    .map_err(|e| CliError::ValidationFailed(format!("Failed to reload: {e}")))?;
                run_real_benchmark(&model, &mapped, config)?
            }
        }
    } else {
        println!("{}", "Running APR benchmark (CPU)...".yellow());
        run_real_benchmark(&model, &mapped, config)?
    };

    let apr_tps = apr_results
        .iter()
        .map(BenchMeasurement::tokens_per_second)
        .sum::<f64>()
        / apr_results.len() as f64;
    let apr_ttft_ms = apr_results
        .iter()
        .map(|m| m.ttft.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / apr_results.len() as f64;
    let apr_tps_stddev = calculate_stddev(
        &apr_results
            .iter()
            .map(BenchMeasurement::tokens_per_second)
            .collect::<Vec<_>>(),
    );

    println!(
        "  APR: {:.1} ± {:.1} tok/s, TTFT: {:.1}ms ({} runs)",
        apr_tps,
        apr_tps_stddev,
        apr_ttft_ms,
        apr_results.len()
    );

    let comparison = build_comparison(
        apr_tps,
        apr_ttft_ms,
        apr_tps_stddev,
        apr_results.len(),
        config,
    );

    print_benchmark_results(&comparison);

    Ok(comparison)
}

/// Shared benchmark setup: tokenize prompt and create generation config.
///
/// Returns (prompt_tokens, gen_config) used by both CPU and GPU benchmark paths.
#[cfg(feature = "inference")]
fn bench_setup(
    mapped: &realizar::gguf::MappedGGUFModel,
) -> (Vec<u32>, realizar::gguf::QuantizedGenerateConfig) {
    let test_prompt = "Hello, I am a coding assistant. Write a function that calculates";
    let prompt_tokens: Vec<u32> = mapped.model.encode(test_prompt).unwrap_or_else(|| {
        // Fallback to Qwen2 pre-tokenized tokens if vocab not available
        vec![151643, 9707, 11, 358, 1079, 264, 11761, 18328, 13, 9842]
    });
    println!(
        "  Prompt: {} tokens (\"{}...\")",
        prompt_tokens.len(),
        &test_prompt[..test_prompt.len().min(30)]
    );

    // PERF-003: Use greedy sampling for fair benchmark (eliminates CPU top-k sort overhead)
    let gen_config = realizar::gguf::QuantizedGenerateConfig {
        max_tokens: 32,
        temperature: 0.0, // Greedy
        top_k: 1,         // Greedy
        ..Default::default()
    };

    (prompt_tokens, gen_config)
}

/// Record a single benchmark measurement from generation output.
#[cfg(feature = "inference")]
fn record_measurement(
    output_len: usize,
    prompt_len: usize,
    duration: Duration,
) -> BenchMeasurement {
    let tokens_generated = output_len.saturating_sub(prompt_len);
    let ttft = if tokens_generated > 0 {
        Duration::from_secs_f64(duration.as_secs_f64() / tokens_generated as f64)
    } else {
        duration
    };
    BenchMeasurement {
        tokens_generated,
        duration,
        ttft,
    }
}

#[cfg(feature = "inference")]
pub(super) fn run_real_benchmark(
    model: &realizar::gguf::OwnedQuantizedModel,
    mapped: &realizar::gguf::MappedGGUFModel,
    config: &ShowcaseConfig,
) -> Result<Vec<BenchMeasurement>> {
    let (prompt_tokens, gen_config) = bench_setup(mapped);

    // Warmup
    print!("  Warmup: ");
    for i in 0..5 {
        let _ = model.generate_with_cache(&prompt_tokens, &gen_config);
        print!("{} ", i + 1);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    println!("done");

    // Measurement runs
    let runs = config.bench_runs.clamp(5, 100);
    let mut measurements = Vec::with_capacity(runs);

    print!("  Measuring: ");
    for i in 0..runs {
        let start = Instant::now();
        let output = model
            .generate_with_cache(&prompt_tokens, &gen_config)
            .unwrap_or_default();
        let duration = start.elapsed();

        measurements.push(record_measurement(
            output.len(),
            prompt_tokens.len(),
            duration,
        ));

        if (i + 1) % 5 == 0 {
            print!("{} ", i + 1);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!("done");

    Ok(measurements)
}

/// GPU benchmark using CUDA-accelerated inference with KV cache
#[cfg(feature = "inference")]
pub(super) fn run_real_benchmark_cuda(
    model: &mut realizar::gguf::OwnedQuantizedModelCuda,
    mapped: &realizar::gguf::MappedGGUFModel,
    config: &ShowcaseConfig,
) -> Result<Vec<BenchMeasurement>> {
    let (prompt_tokens, gen_config) = bench_setup(mapped);

    // Warmup using GPU-resident inference
    print!("  Warmup: ");
    for i in 0..5 {
        if let Err(e) = model.generate_gpu_resident(&prompt_tokens, &gen_config) {
            eprintln!("\n  Warmup error: {e}");
            return Err(CliError::ValidationFailed(format!(
                "GPU warmup failed: {e}"
            )));
        }
        print!("{} ", i + 1);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    println!("done");

    // Measurement runs using GPU-resident path
    let runs = config.bench_runs.clamp(5, 100);
    let mut measurements = Vec::with_capacity(runs);

    print!("  Measuring: ");
    for i in 0..runs {
        let start = Instant::now();
        let output = match model.generate_gpu_resident(&prompt_tokens, &gen_config) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("\n  Generation error: {e}");
                // Return early with whatever measurements we have
                if measurements.is_empty() {
                    return Err(CliError::ValidationFailed(format!(
                        "GPU generation failed: {e}"
                    )));
                }
                break;
            }
        };
        let duration = start.elapsed();

        measurements.push(record_measurement(
            output.len(),
            prompt_tokens.len(),
            duration,
        ));

        if (i + 1) % 5 == 0 {
            print!("{} ", i + 1);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!("done");

    Ok(measurements)
}

#[cfg(not(feature = "inference"))]
pub(super) fn run_benchmark(config: &ShowcaseConfig) -> Result<BenchmarkComparison> {
    println!();
    println!("{}", "═══ Step E: Performance Benchmark ═══".cyan().bold());
    println!();
    println!(
        "{} Inference feature not enabled. Using simulated benchmarks.",
        "⚠".yellow()
    );

    // Simulated with real variance
    let apr_tps = 44.0 + generate_jitter() * 2.0;
    let apr_ttft_ms = 78.0 + generate_jitter() * 5.0;

    let comparison = build_comparison(apr_tps, apr_ttft_ms, 2.0, config.bench_runs, config);

    print_benchmark_results(&comparison);
    Ok(comparison)
}

include!("benchmark_part_02.rs");
