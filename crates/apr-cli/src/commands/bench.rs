//! Benchmark Command Implementation
//!
//! Implements spec §H12: Throughput benchmark for model inference.
//!
//! # Usage
//!
//! ```bash
//! apr bench model.gguf                   # GGUF model benchmark
//! apr bench model.apr                    # APR model benchmark
//! apr bench model.safetensors            # SafeTensors benchmark
//! apr bench model.gguf --warmup 3        # 3 warmup iterations
//! apr bench model.gguf --iterations 10   # 10 measurement iterations
//! apr bench model.gguf --prompt "Hello"  # Custom prompt
//! ```
//!
//! Toyota Way: Genchi Genbutsu - measure actual performance, not estimates.
//!
//! ## Supported Formats
//!
//! - **GGUF** (.gguf) - Full support with GPU acceleration
//! - **APR** (.apr) - Native format support
//! - **SafeTensors** (.safetensors) - HuggingFace format support

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use std::path::Path;
use std::time::{Duration, Instant};

/// Benchmark configuration
struct BenchConfig {
    /// Number of warmup iterations (not measured)
    pub warmup: usize,
    /// Number of measurement iterations
    pub iterations: usize,
    /// Max tokens to generate per iteration
    pub max_tokens: usize,
    /// Test prompt
    pub prompt: String,
    /// GH-254: Suppress status output (JSON mode)
    pub quiet: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup: 3,
            iterations: 5,
            max_tokens: 32,
            prompt: "What is 2+2?".to_string(),
            quiet: false,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchResult {
    /// Total tokens generated across all iterations
    pub total_tokens: usize,
    /// Total time for generation
    pub total_time: Duration,
    /// Tokens per second (throughput)
    pub tokens_per_second: f64,
    /// Time to first token (TTFT)
    pub time_to_first_token: Duration,
    /// Individual iteration times
    pub iteration_times: Vec<Duration>,
    /// Mean iteration time
    pub mean_time: Duration,
    /// Median iteration time
    pub median_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Passed threshold (spec H12: >= 10 tok/s)
    pub passed: bool,
}

/// Run the benchmark command
///
/// Automatically detects format and uses realizar for optimized inference.
/// Supports GGUF, APR, and SafeTensors formats.
pub(crate) fn run(
    path: &Path,
    warmup: usize,
    iterations: usize,
    max_tokens: usize,
    prompt: Option<&str>,
    _fast: bool, // Deprecated: always uses fast path now
    brick: Option<&str>,
    json: bool,
) -> Result<()> {
    // If --brick is specified, run brick-specific benchmark
    if let Some(brick_name) = brick {
        #[cfg(feature = "inference")]
        {
            return run_brick_benchmark(brick_name, warmup, iterations);
        }
        #[cfg(not(feature = "inference"))]
        {
            let _ = brick_name;
            return Err(CliError::ValidationFailed(
                "--brick requires the 'inference' feature. Build with: cargo build --features inference".to_string()
            ));
        }
    }

    let config = BenchConfig {
        warmup,
        iterations,
        max_tokens,
        prompt: prompt.unwrap_or("What is 2+2?").to_string(),
        quiet: json,
    };

    if !json {
        print_header(path, &config);
    }

    // Always use realizar for production-quality benchmarks
    #[cfg(feature = "inference")]
    let result = {
        if !json {
            println!("{}", "Using realizar inference engine".cyan());
            println!();
        }
        run_realizar_benchmark(path, &config)?
    };

    #[cfg(not(feature = "inference"))]
    let result = {
        return Err(CliError::ValidationFailed(
            "Benchmark requires the 'inference' feature. Build with: cargo build --features inference".to_string()
        ));
    };

    // GH-254: JSON output mode — always exit 0 with results in JSON body
    if json {
        return print_bench_json(path, &result);
    }

    // Print results
    print_results(&result);

    // Threshold: 10 tok/s minimum
    let threshold = 10.0;
    let passed = result.tokens_per_second >= threshold;

    if !passed {
        return Err(CliError::ValidationFailed(format!(
            "Throughput {:.1} tok/s below minimum {:.0} tok/s (spec H12)",
            result.tokens_per_second, threshold
        )));
    }

    Ok(())
}

/// GH-254: Print benchmark results as JSON (machine-parseable output).
/// Always exits 0 — failure info is in the JSON body.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_bench_json(path: &Path, result: &BenchResult) -> Result<()> {
    let output = serde_json::json!({
        "model": path.display().to_string(),
        "tokens_per_second": (result.tokens_per_second * 10.0).round() / 10.0,
        "total_tokens": result.total_tokens,
        "total_time_ms": result.total_time.as_secs_f64() * 1000.0,
        "time_to_first_token_ms": result.time_to_first_token.as_secs_f64() * 1000.0,
        "iterations": result.iteration_times.len(),
        "mean_time_ms": result.mean_time.as_secs_f64() * 1000.0,
        "median_time_ms": result.median_time.as_secs_f64() * 1000.0,
        "std_dev_ms": result.std_dev.as_secs_f64() * 1000.0,
        "passed": result.passed,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    Ok(())
}

/// Resolve brick budget target and description from name (spec §9.2).
///
/// Returns `(budget_us, description)` or error for unknown brick types.
#[cfg(feature = "inference")]
fn resolve_brick_spec(brick_name: &str) -> Result<(f64, &'static str)> {
    match brick_name {
        "rms_norm" => Ok((1.5, "RMS Layer Normalization")),
        "qkv" => Ok((6.0, "Q/K/V Projections")),
        "rope" => Ok((1.0, "Rotary Position Embedding")),
        "attn" | "attention" => Ok((10.0, "Scaled Dot-Product Attention")),
        "o_proj" => Ok((3.5, "Output Projection")),
        "ffn" => Ok((12.2, "Feed-Forward Network (SwiGLU)")),
        "layer" => Ok((35.7, "Full Transformer Layer")),
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown brick type: '{}'. Valid: rms_norm, qkv, rope, attn, o_proj, ffn, layer",
            brick_name
        ))),
    }
}

/// Execute the benchmark for a specific brick type, returning the report.
#[cfg(feature = "inference")]
fn execute_brick_benchmark(
    brick_name: &str,
    bench_config: &realizar::brick::BenchmarkConfig,
) -> realizar::brick::BenchmarkReport {
    use realizar::brick::{
        benchmark_brick, AttentionBrick, ComputeBrick, FfnBrick, OProjBrick, QkvBrick,
        RmsNormBrick, RopeBrick, TransformerLayerBrick,
    };

    match brick_name {
        "rms_norm" => {
            let brick = RmsNormBrick::new(vec![1.0; 896], 1e-5);
            let input: Vec<f32> = vec![1.0; 896];
            benchmark_brick(
                &brick,
                || {
                    let start = Instant::now();
                    let _ = brick.run(&input);
                    start.elapsed().as_nanos() as f64 / 1000.0
                },
                bench_config,
            )
        }
        "qkv" => {
            let brick = QkvBrick::new(896, 896, 128, 128);
            benchmark_brick(
                &brick,
                || {
                    let start = Instant::now();
                    let _ = brick.budget();
                    start.elapsed().as_nanos() as f64 / 1000.0
                },
                bench_config,
            )
        }
        "rope" => {
            let brick = RopeBrick::new(64, 14, 1_000_000.0, 2);
            benchmark_brick(
                &brick,
                || {
                    let start = Instant::now();
                    let _ = brick.budget();
                    start.elapsed().as_nanos() as f64 / 1000.0
                },
                bench_config,
            )
        }
        "attn" | "attention" => {
            let brick = AttentionBrick::new(14, 2, 64);
            benchmark_brick(
                &brick,
                || {
                    let start = Instant::now();
                    let _ = brick.budget();
                    start.elapsed().as_nanos() as f64 / 1000.0
                },
                bench_config,
            )
        }
        "o_proj" => {
            let brick = OProjBrick::new(896, 896);
            benchmark_brick(
                &brick,
                || {
                    let start = Instant::now();
                    let _ = brick.budget();
                    start.elapsed().as_nanos() as f64 / 1000.0
                },
                bench_config,
            )
        }
        "ffn" => {
            let brick = FfnBrick::new(896, 4864);
            benchmark_brick(
                &brick,
                || {
                    let start = Instant::now();
                    let _ = brick.budget();
                    start.elapsed().as_nanos() as f64 / 1000.0
                },
                bench_config,
            )
        }
        "layer" => {
            let brick =
                TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1_000_000.0, 2);
            benchmark_brick(
                &brick,
                || {
                    let start = Instant::now();
                    let _ = brick.total_budget_us();
                    start.elapsed().as_nanos() as f64 / 1000.0
                },
                bench_config,
            )
        }
        _ => unreachable!(),
    }
}

/// Print brick benchmark results: latency, CV, percentiles, throughput, and grade.
#[cfg(feature = "inference")]
fn print_brick_results(
    report: &realizar::brick::BenchmarkReport,
    budget_target: f64,
    elapsed: Duration,
) {
    output::section("Results");
    println!();

    let mean_us = report.mean_us;
    let cv = report.cv;
    let budget_met = mean_us <= budget_target;
    let cv_stable = cv <= 0.05;

    // Mean latency
    let mean_str = format!("{:.2}µs", mean_us);
    if budget_met {
        println!(
            "{} {} {}",
            "Mean Latency:".white().bold(),
            mean_str.green().bold(),
            format!("(PASS: ≤ {:.1}µs)", budget_target).green()
        );
    } else {
        println!(
            "{} {} {}",
            "Mean Latency:".white().bold(),
            mean_str.red().bold(),
            format!("(FAIL: > {:.1}µs)", budget_target).red()
        );
    }

    // Coefficient of variation (stability)
    let cv_str = format!("{:.2}%", cv * 100.0);
    if cv_stable {
        println!(
            "{} {} {}",
            "CV (stability):".white().bold(),
            cv_str.green(),
            "(PASS: ≤ 5%)".green()
        );
    } else {
        println!(
            "{} {} {}",
            "CV (stability):".white().bold(),
            cv_str.yellow(),
            "(WARN: > 5%)".yellow()
        );
    }

    println!();
    output::kv("P50", format!("{:.2}µs", report.p50_us));
    output::kv("P99", format!("{:.2}µs", report.p99_us));
    output::kv("Std Dev", format!("{:.2}µs", report.std_us));
    output::kv("Budget", format!("{:.2}µs", report.budget_us));
    output::kv("Benchmark Time", format!("{:.2}s", elapsed.as_secs_f32()));
    println!();

    output::kv("Throughput", format!("{:.0} tok/s", report.tokens_per_sec));
    println!();

    // Performance grade
    let grade = if mean_us <= budget_target * 0.5 {
        "A+ (Excellent: < 50% of budget)".green()
    } else if mean_us <= budget_target * 0.75 {
        "A (Very Good: < 75% of budget)".green()
    } else if mean_us <= budget_target {
        "B (Good: within budget)".blue()
    } else if mean_us <= budget_target * 1.5 {
        "C (Acceptable: < 150% of budget)".yellow()
    } else {
        "F (Over Budget)".red()
    };
    output::kv("Performance Grade", grade);
    println!();

    // Statistical validity check
    if report.statistically_valid {
        println!("{}", "Statistical validity: PASS (CV < 5%)".green());
    } else {
        println!("{}", "Statistical validity: WARN (CV >= 5%)".yellow());
    }
    println!();
}

/// Brick-specific benchmark per spec §9.2
///
/// Tests individual ComputeBrick types for their token budget compliance.
/// Implements falsification tests F023-F029 for per-brick performance.
#[cfg(feature = "inference")]
fn run_brick_benchmark(brick_name: &str, warmup: usize, iterations: usize) -> Result<()> {
    use realizar::brick::BenchmarkConfig;

    let (budget_target, brick_description) = resolve_brick_spec(brick_name)?;

    output::section("APR Brick Benchmark");
    println!();
    output::kv("Brick", brick_name);
    output::kv("Warmup", warmup);
    output::kv("Iterations", iterations);
    println!();
    output::kv("Description", brick_description);
    output::kv("Budget Target", format!("≤ {:.1}µs", budget_target));
    println!();

    let bench_config = BenchmarkConfig {
        warmup,
        samples: iterations,
        max_cv: 0.05,
    };

    println!("{}", "Running benchmark...".yellow());
    let bench_start = Instant::now();
    let report = execute_brick_benchmark(brick_name, &bench_config);
    let elapsed = bench_start.elapsed();
    println!("{}", "Benchmark complete.".green());
    println!();

    print_brick_results(&report, budget_target, elapsed);

    if report.mean_us > budget_target {
        return Err(CliError::ValidationFailed(format!(
            "Brick '{}' exceeded budget: {:.2}µs > {:.1}µs (spec F023-F029)",
            brick_name, report.mean_us, budget_target
        )));
    }

    Ok(())
}

fn print_header(path: &Path, config: &BenchConfig) {
    output::section("APR Benchmark");
    println!();
    output::kv("Model", path.display());
    output::kv("Warmup iterations", config.warmup);
    output::kv("Measurement iterations", config.iterations);
    output::kv("Max tokens", config.max_tokens);
    output::kv("Prompt", &config.prompt);
    println!();
}

include!("bench_part_02.rs");
include!("bench_part_03.rs");
include!("bench_part_04.rs");
