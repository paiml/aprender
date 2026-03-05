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

#[cfg(not(feature = "visualization"))]
use brick_tracer_shim::BrickTracer as TracerImpl;
#[cfg(feature = "visualization")]
use renacer::brick_tracer::BrickTracer as TracerImpl;

/// No-op BrickTracer shim when the `visualization` (renacer) feature is disabled.
/// Provides the same API surface so callers compile without cfg gates on every call site.
#[cfg(not(feature = "visualization"))]
mod brick_tracer_shim {
    /// Stub syscall breakdown — all zeros.
    pub struct SyscallBreakdown {
        pub compute_us: u64,
        pub mmap_us: u64,
        pub futex_us: u64,
        pub ioctl_us: u64,
    }
    impl SyscallBreakdown {
        pub fn syscall_overhead_percent(&self) -> f64 {
            0.0
        }
        pub fn dominant_syscall(&self) -> &'static str {
            "none"
        }
    }

    /// Stub trace metadata.
    pub struct TraceMetadata {
        pub budget_us: u64,
        pub actual_us: u64,
        pub efficiency: f64,
    }

    /// Result of a traced operation — contains the closure result + timing.
    pub struct TracedResult<T> {
        pub result: T,
        pub duration_us: u64,
        pub syscall_breakdown: SyscallBreakdown,
        pub metadata: Option<TraceMetadata>,
    }

    /// No-op tracer that just times the closure with `Instant`.
    pub struct BrickTracer;
    impl BrickTracer {
        pub fn new_local() -> Self {
            Self
        }
        pub fn trace<T>(
            &self,
            _name: &str,
            _budget_us: u64,
            f: impl FnOnce() -> T,
        ) -> TracedResult<T> {
            let start = std::time::Instant::now();
            let result = f();
            let duration_us = start.elapsed().as_micros() as u64;
            TracedResult {
                result,
                duration_us,
                syscall_breakdown: SyscallBreakdown {
                    compute_us: duration_us,
                    mmap_us: 0,
                    futex_us: 0,
                    ioctl_us: 0,
                },
                metadata: None,
            }
        }
    }
}

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
            return run_brick_benchmark(brick_name, warmup, iterations, path);
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
        "tokenize" | "bpe" => Ok((80.0, "BPE Tokenizer Encode (GH-378)")),
        // Training bricks
        "lora_forward" | "lora" => Ok((5.0, "LoRA Forward Pass (rank-16)")),
        "optimizer" | "adamw" => Ok((50.0, "SIMD AdamW Optimizer Step")),
        "loss" | "cross_entropy" => Ok((20.0, "Cross-Entropy Loss Computation")),
        "train_step" | "training" => Ok((5000.0, "Full Training Step (fwd+bwd+optim)")),
        // Serving bricks
        "ttft" | "time_to_first_token" => Ok((500.0, "Time to First Token")),
        "throughput" | "decode" => Ok((20000.0, "Decode Throughput (50 tok/s target)")),
        "batch" | "batch_generate" => Ok((1000.0, "Batch Generation (4 concurrent)")),
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown brick type: '{}'. Valid: rms_norm, qkv, rope, attn, o_proj, ffn, layer, \
             tokenize, lora_forward, optimizer, loss, train_step, ttft, throughput, batch",
            brick_name
        ))),
    }
}

/// GH-90: Return analytical budget for bricks without run() implementations.
/// These bricks are architectural contracts — they define performance budgets
/// but don't execute real computation. The budget is a theoretical estimate
/// based on FLOP counts and memory bandwidth, not measured wall-clock time.
#[cfg(feature = "inference")]
fn analytical_budget_report(
    brick: &impl realizar::brick::ComputeBrick,
) -> realizar::brick::BenchmarkReport {
    let budget = brick.budget();
    eprintln!(
        "[GH-90] Brick '{}' has no run() implementation — reporting analytical budget ({:.1}µs), not measured timing",
        brick.name(),
        budget.us_per_token
    );
    realizar::brick::BenchmarkReport {
        brick_name: brick.name().to_string(),
        mean_us: budget.us_per_token,
        std_us: 0.0,
        cv: 0.0,
        p50_us: budget.us_per_token,
        p99_us: budget.us_per_token,
        tokens_per_sec: 1_000_000.0 / budget.us_per_token,
        budget_us: budget.us_per_token,
        budget_met: true,
        statistically_valid: true,
    }
}

/// Read model architecture config from an APR file (metadata only).
///
/// Loads AprTransformer to extract config dimensions (hidden_dim, num_layers, etc.)
/// used by training and serving bricks for real-model benchmarking.
// TODO: Re-enable when realizar publishes training/serving bricks
#[cfg(feature = "inference")]
#[allow(dead_code)]
fn read_apr_model_config(
    model_path: &Path,
) -> Result<realizar::apr_transformer::AprTransformerConfig> {
    use realizar::apr_transformer::AprTransformer;

    let transformer = AprTransformer::from_apr_file(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR config: {e}")))?;
    Ok(transformer.config)
}

/// Execute the benchmark for a specific brick type, returning the report.
#[cfg(feature = "inference")]
fn execute_brick_benchmark(
    brick_name: &str,
    bench_config: &realizar::brick::BenchmarkConfig,
    _model_path: &Path,
) -> Result<realizar::brick::BenchmarkReport> {
    use realizar::brick::{
        benchmark_brick, AttentionBrick, FfnBrick, OProjBrick, QkvBrick, RmsNormBrick, RopeBrick,
        TransformerLayerBrick,
    };

    // GH-90: Bricks without run() return analytical budget directly.
    // Only rms_norm and tokenize have real run() implementations — all others are
    // architectural contracts with budget() only. Report the analytical
    // budget rather than timing a no-op budget() call.
    let report = match brick_name {
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
        // TODO: TokenizeBrick not yet published in realizar 0.8.0
        // Restore when realizar publishes TokenizeBrick
        "tokenize" | "bpe" => {
            return Err(CliError::ValidationFailed(
                "tokenize brick not yet available: TokenizeBrick is not published in realizar 0.8.0".to_string()
            ));
        }
        "qkv" => {
            let brick = QkvBrick::new(896, 896, 128, 128);
            analytical_budget_report(&brick)
        }
        "rope" => {
            let brick = RopeBrick::new(64, 14, 1_000_000.0, 2);
            analytical_budget_report(&brick)
        }
        "attn" | "attention" => {
            let brick = AttentionBrick::new(14, 2, 64);
            analytical_budget_report(&brick)
        }
        "o_proj" => {
            let brick = OProjBrick::new(896, 896);
            analytical_budget_report(&brick)
        }
        "ffn" => {
            let brick = FfnBrick::new(896, 4864);
            analytical_budget_report(&brick)
        }
        "layer" => {
            let brick =
                TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1_000_000.0, 2);
            let budget_us = brick.total_budget_us();
            realizar::brick::BenchmarkReport {
                brick_name: "layer".to_string(),
                mean_us: budget_us,
                std_us: 0.0,
                cv: 0.0,
                p50_us: budget_us,
                p99_us: budget_us,
                tokens_per_sec: 1_000_000.0 / budget_us,
                budget_us,
                budget_met: true,
                statistically_valid: true,
            }
        }

        // TODO: Training and serving bricks not yet published in realizar 0.8.0.
        // Restore when realizar publishes: LoraForwardBrick, OptimizerStepBrick,
        // LossComputeBrick, TrainingStepBrick, ServeTtftBrick, ServeThroughputBrick,
        // ServeBatchBrick.
        "lora_forward"
        | "lora"
        | "optimizer"
        | "adamw"
        | "loss"
        | "cross_entropy"
        | "train_step"
        | "training"
        | "ttft"
        | "time_to_first_token"
        | "throughput"
        | "decode"
        | "batch"
        | "batch_generate" => {
            return Err(CliError::ValidationFailed(format!(
                "brick '{}' not yet available: its brick type is not published in realizar 0.8.0",
                brick_name
            )));
        }

        _ => unreachable!(),
    };
    Ok(report)
}

/// Load a BPE tokenizer for the tokenize brick benchmark.
///
/// Searches for tokenizer.json in multiple locations relative to the model path:
/// 1. Sibling `{model_stem}.tokenizer.json`
/// 2. Sibling `tokenizer.json` in same directory
/// 3. Embedded tokenizer in GGUF/APR model (extracts to temp file)
// TODO: Re-enable when realizar publishes TokenizeBrick
#[cfg(feature = "inference")]
#[allow(dead_code)]
fn load_tokenizer_for_brick(model_path: &Path) -> Result<aprender::text::bpe::BpeTokenizer> {
    use aprender::text::bpe::BpeTokenizer;

    // 1. Sibling {stem}.tokenizer.json
    let stem = model_path.file_stem().unwrap_or_default().to_string_lossy();
    let sibling = model_path.with_file_name(format!("{stem}.tokenizer.json"));
    if sibling.exists() {
        return BpeTokenizer::from_huggingface(&sibling).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to load tokenizer from {}: {e}",
                sibling.display()
            ))
        });
    }

    // 2. tokenizer.json in same directory
    if let Some(parent) = model_path.parent() {
        let tokenizer_json = parent.join("tokenizer.json");
        if tokenizer_json.exists() {
            return BpeTokenizer::from_huggingface(&tokenizer_json).map_err(|e| {
                CliError::ValidationFailed(format!(
                    "Failed to load tokenizer from {}: {e}",
                    tokenizer_json.display()
                ))
            });
        }
    }

    Err(CliError::ValidationFailed(format!(
        "No tokenizer.json found for '{}'. Place tokenizer.json next to the model or use \
         '{}.tokenizer.json'",
        model_path.display(),
        stem
    )))
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

    // GH-90: Indicate when results are analytical (not measured)
    let is_analytical =
        report.std_us == 0.0 && report.p50_us == report.p99_us && report.p50_us == report.mean_us;
    if is_analytical {
        println!(
            "{}",
            "NOTE: This is an ANALYTICAL budget estimate (no run() implementation).".yellow()
        );
        println!(
            "{}",
            "Use `apr bench <model> --fast` for real measured throughput.".yellow()
        );
        println!();
    }

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
fn run_brick_benchmark(
    brick_name: &str,
    warmup: usize,
    iterations: usize,
    model_path: &Path,
) -> Result<()> {
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
    let report = execute_brick_benchmark(brick_name, &bench_config, model_path)?;
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

include!("benchmark.rs");
include!("bench_safetensors.rs");
include!("bench_04.rs");
