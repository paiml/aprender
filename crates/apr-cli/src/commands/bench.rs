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
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
/// Parse one `--percentiles` point, enforcing the documented (0, 100] range.
///
/// Out-of-range points used to sail through to `print_bench_json`, where the
/// non-`Ok` `PercentileOutcome` was serialised as `null` under a plausible key
/// (`latency_p101_ms`) — a consumer of the report saw a metric, not an error.
pub(crate) fn parse_percentile(s: &str) -> std::result::Result<f64, String> {
    let value: f64 = s
        .trim()
        .parse()
        .map_err(|e| format!("invalid percentile '{s}': {e}"))?;
    if !value.is_finite() || value <= 0.0 || value > 100.0 {
        return Err(format!(
            "percentile '{s}' out of range: values must be in (0, 100]"
        ));
    }
    Ok(value)
}

pub(crate) fn run(
    path: &Path,
    warmup: usize,
    iterations: usize,
    max_tokens: usize,
    prompt: Option<&str>,
    fast: bool,
    brick: Option<&str>,
    json: bool,
    percentiles: &[f64],
) -> Result<()> {
    // Defence in depth: the clap value_parser rejects out-of-range points, but
    // `run` is also reachable from non-clap callers.
    for &p in percentiles {
        if !p.is_finite() || p <= 0.0 || p > 100.0 {
            return Err(CliError::ValidationFailed(format!(
                "percentile {p} out of range: values must be in (0, 100]"
            )));
        }
    }

    // GH-512: Warn on deprecated --fast flag instead of silently ignoring
    if fast && !json {
        eprintln!("Warning: --fast is deprecated (always uses fast path now). Flag has no effect.");
    }

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
        return print_bench_json(path, &result, percentiles);
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

/// Print benchmark results as JSON (machine-parseable output).
/// GH-254→GH-601: Exit code matches `passed` field — non-zero when failed.
/// PARITY-001 / PERF-006 — the dispatch path this binary can actually take.
///
/// DELEGATES. The body used to live here and read apr-cli's own `cfg!`s, which
/// made it the third of three disagreeing answers: this receipt said
/// `cpu/cuda/metal/wgpu`, `GET /health` said `cpu/gpu` from a different
/// derivation, and the serve banner said nothing at all. APR-PERF-GATE-001
/// v2.2 §4 lists that row as the Andon countermeasure — *one* `compute_class()`
/// feeding all three — and it was marked **pending**. This is that one function
/// (`realizar::andon::compute_class`); the banner and `/health` call the same
/// symbol.
///
/// One behaviour changed with the move, deliberately: the old body returned
/// `"wgpu"` for `cfg!(feature = "wgpu")`, but apr-cli declares
/// `wgpu = ["inference"]` — it enables no wgpu dispatch anywhere, it only
/// widens `serve::ensure_accelerator_available`. So that arm named a path the
/// binary could not take, in the one field whose job is to prove which path it
/// took. It now reports `cpu`, which is what such a build runs on.
///
/// A build with no `inference` feature has no inference engine linked at all,
/// so there is no dispatch path to name. That is `"unknown"` — a vocabulary
/// member, not a second implementation of the class.
#[cfg(feature = "inference")]
fn compute_class() -> &'static str {
    realizar::andon::compute_class()
}

#[cfg(not(feature = "inference"))]
fn compute_class() -> &'static str {
    "unknown"
}

/// PERF-006 — how many generations this process runs AT ONCE.
///
/// §14 calls this "the cheapest confirmation of defect #2 in this document",
/// defect #2 being that apr does not batch. `apr bench` drives one stream, and
/// nothing in it records a scheduler, so this reads 1 — which is the point.
/// The field is emitted on the serialized path rather than only when batching
/// is active; a number that appears only in the flattering case reports
/// success and is silent on the failure it exists to expose (#2696's shape).
#[cfg(feature = "inference")]
fn max_in_flight() -> usize {
    realizar::andon::max_in_flight()
}

#[cfg(not(feature = "inference"))]
fn max_in_flight() -> usize {
    1
}

/// PARITY-001 — sha256 of a file's contents, for model identity.
///
/// A benchmark receipt that names a model by PATH is not reproducible: the
/// path is a claim about the filesystem, the digest is a claim about the
/// bytes that were actually fed to the model loader.
fn file_sha256(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut f, &mut hasher).ok()?;
    Some(format!("{:x}", hasher.finalize()))
}

/// PARITY-001 — the provenance block: which binary produced this receipt.
///
/// `resolution` is `path` when the binary was found through `$PATH` and
/// `pinned` when it came from a pin script. On a release-deciding gate,
/// `path` is RED — five separate incidents in this repo were a gate
/// believing the wrong binary (PMAT_BIN, pv_bin.sh, apr_bin.sh, #2384,
/// unpinned llama.cpp). `reported_version` is recorded SEPARATELY from
/// `build_commit` because #2384 is exactly the case where they disagree:
/// two binaries both printed 3.32.0 and only the commit differed.
fn provenance_json() -> serde_json::Value {
    let exe = std::env::current_exe().ok();
    let binary_sha256 = exe.as_deref().and_then(file_sha256);
    let mut features: Vec<&'static str> = Vec::new();
    if cfg!(feature = "inference") {
        features.push("inference");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "wgpu") {
        features.push("wgpu");
    }
    if cfg!(feature = "training") {
        features.push("training");
    }
    serde_json::json!({
        "binary_path": exe.as_ref().map(|p| p.display().to_string()),
        "binary_sha256": binary_sha256,
        "reported_version": env!("CARGO_PKG_VERSION"),
        "build_commit": option_env!("APR_GIT_SHA").unwrap_or("unknown"),
        "resolution": "path",
        "compute_class": compute_class(),
        "max_in_flight": max_in_flight(),
        "feature_set": features,
    })
}

/// CRUX-E-07: emits `latency_p<N>_ms` key per requested percentile point.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_bench_json(path: &Path, result: &BenchResult, percentiles: &[f64]) -> Result<()> {
    let mut output = serde_json::json!({
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
    if let Some(obj) = output.as_object_mut() {
        let samples_ms: Vec<f64> = result
            .iteration_times
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        // PARITY-001: emit the RAW samples, not only the summary.
        //
        // These were already computed here and thrown away after feeding the
        // percentiles. That is not a convenience loss: summary statistics
        // CANNOT BE RESAMPLED, so a receipt carrying only mean/stddev
        // permanently forecloses bootstrap — which is the only threshold
        // method that survived review (PARITY-008 falsified the RFC's
        // `3 x pooled relative stddev` against this repo's own data: derived
        // 19.836% vs actual 19.306%, i.e. GREEN on the one regression we
        // have on record, and weakening as more data accumulates).
        obj.insert("samples_ms".to_string(), serde_json::json!(samples_ms));
        obj.insert("n".to_string(), serde_json::json!(samples_ms.len()));
        // runs_discarded is deliberately NOT constrained to zero. The one
        // working throughput harness in this tree discards by construction on
        // token-count inconstancy, so a zero-invariant would contradict the
        // only correct implementation we have.
        obj.insert("runs_discarded".to_string(), serde_json::json!(0));
        obj.insert("provenance".to_string(), provenance_json());
        obj.insert(
            "model_sha256".to_string(),
            serde_json::json!(file_sha256(path)),
        );
        for &p in percentiles {
            let key = format!("latency_p{}_ms", p.round() as u64);
            let v = match aprender::metrics::percentile::compute_percentile(&samples_ms, p) {
                aprender::metrics::percentile::PercentileOutcome::Ok(v) => {
                    serde_json::json!(v)
                }
                _ => serde_json::Value::Null,
            };
            obj.insert(key, v);
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    // GH-601: Exit code must match JSON "passed" field.
    if result.passed {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "Throughput {:.1} tok/s below minimum (spec H12)",
            result.tokens_per_second
        )))
    }
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
// Disabled until realizar publishes training/serving bricks
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
        // TokenizeBrick not yet published in realizar 0.8.0
        // Will be restored when realizar publishes TokenizeBrick
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

        // Training and serving bricks not yet published in realizar 0.8.0.
        // Pending realizar publication of: LoraForwardBrick, OptimizerStepBrick,
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
// Disabled until realizar publishes TokenizeBrick
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
    maybe_print_analytical_notice(report);
    print_mean_latency_line(report.mean_us, budget_target);
    print_cv_line(report.cv);
    print_stats_block(report, elapsed);
    print_throughput_line(report.tokens_per_sec);
    print_performance_grade(report.mean_us, budget_target);
    print_statistical_validity(report.statistically_valid);
}

#[cfg(feature = "inference")]
fn maybe_print_analytical_notice(report: &realizar::brick::BenchmarkReport) {
    // GH-90: Indicate when results are analytical (not measured)
    let is_analytical =
        report.std_us == 0.0 && report.p50_us == report.p99_us && report.p50_us == report.mean_us;
    if !is_analytical {
        return;
    }
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

#[cfg(feature = "inference")]
fn print_mean_latency_line(mean_us: f64, budget_target: f64) {
    let mean_str = format!("{:.2}µs", mean_us);
    if mean_us <= budget_target {
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
}

#[cfg(feature = "inference")]
fn print_cv_line(cv: f64) {
    let cv_str = format!("{:.2}%", cv * 100.0);
    if cv <= 0.05 {
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
}

#[cfg(feature = "inference")]
fn print_stats_block(report: &realizar::brick::BenchmarkReport, elapsed: Duration) {
    println!();
    output::kv("P50", format!("{:.2}µs", report.p50_us));
    output::kv("P99", format!("{:.2}µs", report.p99_us));
    output::kv("Std Dev", format!("{:.2}µs", report.std_us));
    output::kv("Budget", format!("{:.2}µs", report.budget_us));
    output::kv("Benchmark Time", format!("{:.2}s", elapsed.as_secs_f32()));
    println!();
}

#[cfg(feature = "inference")]
fn print_throughput_line(tokens_per_sec: f64) {
    output::kv("Throughput", format!("{:.0} tok/s", tokens_per_sec));
    println!();
}

#[cfg(feature = "inference")]
fn print_performance_grade(mean_us: f64, budget_target: f64) {
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
}

#[cfg(feature = "inference")]
fn print_statistical_validity(valid: bool) {
    if valid {
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
include!("bench_moe.rs");
include!("bench_04.rs");

// ── PARITY-001: the bench receipt's provenance fields ───────────────────────
//
// These tests assert the two properties that make a receipt usable as
// evidence rather than as decoration: the raw samples survive, and the
// compute class describes the path this build can actually take.
#[cfg(test)]
mod parity_001_receipt_tests {
    use super::*;

    /// `compute_class` must be decided by what was COMPILED IN, not by what
    /// hardware happens to be attached. A default-features build on a machine
    /// with four GPUs is still `cpu`, because the dispatch code is not there.
    ///
    /// This is the assertion that would have refused the fabricated 14x
    /// regression: a CPU-only apr side measured against a CUDA comparator.
    #[test]
    fn compute_class_is_cpu_without_a_gpu_feature() {
        let class = compute_class();
        if !cfg!(feature = "inference") {
            // PERF-006: no inference engine is linked, so there is no dispatch
            // path to name. `unknown` is the vocabulary member for that; `cpu`
            // would be a claim about a code path this binary does not contain.
            assert_eq!(class, "unknown");
        } else if cfg!(feature = "cuda") {
            // Built with a GPU feature: the class may legitimately be a GPU
            // path, or `cpu` if the runtime turned out to be absent.
            assert!(
                ["cuda", "cpu"].contains(&class),
                "unexpected compute_class {class} for a GPU-feature build"
            );
        } else {
            assert_eq!(
                class, "cpu",
                "a build without cuda/wgpu features cannot take a GPU path, \
                 whatever hardware is present — this is the field whose absence \
                 lets a cross-class ratio validate cleanly"
            );
        }
    }

    /// PERF-006 — the receipt and the server must not be able to disagree.
    ///
    /// This asserts the receipt's class IS the shared function's answer, not
    /// merely that it is spelled the same way. Reimplementing the derivation
    /// here — even identically — would make the test survive the very drift it
    /// exists to catch.
    #[test]
    #[cfg(feature = "inference")]
    fn the_receipt_reports_the_shared_compute_class_verbatim() {
        let p = provenance_json();
        assert_eq!(
            p.get("compute_class").and_then(serde_json::Value::as_str),
            Some(realizar::andon::compute_class()),
            "the receipt must render `realizar::andon::compute_class()`, the same \
             symbol the serve banner prints and `GET /health` returns"
        );
    }

    /// PERF-006 — `max_in_flight` is present on the SERIALIZED path.
    ///
    /// `apr bench` wires no scheduler, so this is 1. A receipt that omitted
    /// the field here would report a number only when batching made it
    /// flattering, which is #2696's shape.
    #[test]
    fn the_receipt_reports_max_in_flight_even_when_it_is_one() {
        let p = provenance_json();
        let n = p
            .get("max_in_flight")
            .and_then(serde_json::Value::as_u64)
            .expect("provenance must always carry max_in_flight");
        assert!(n >= 1, "max_in_flight is a count of concurrent generations");
        assert_eq!(
            n as usize,
            max_in_flight(),
            "the receipt must render the shared accessor, not its own count"
        );
    }

    /// The class must be one of the values the receipt schema admits. An
    /// unrecognised value is worse than a missing one: it reads as measured.
    #[test]
    fn compute_class_is_in_the_schema_vocabulary() {
        const ALLOWED: [&str; 5] = ["cpu", "cuda", "metal", "wgpu", "unknown"];
        assert!(
            ALLOWED.contains(&compute_class()),
            "compute_class must be one of {ALLOWED:?}"
        );
    }

    /// The provenance block must carry every field the schema requires, and
    /// must never assert a digest it did not compute.
    #[test]
    fn provenance_carries_the_required_fields() {
        let p = provenance_json();
        for key in [
            "binary_path",
            "binary_sha256",
            "resolution",
            "compute_class",
        ] {
            assert!(
                p.get(key).is_some(),
                "provenance is missing the required key {key}"
            );
        }
        // `reported_version` is recorded separately from `build_commit`
        // because #2384 is precisely the case where the two disagree.
        assert!(p.get("reported_version").is_some());
        assert!(p.get("build_commit").is_some());
    }

    /// A digest of real bytes, and `None` rather than a placeholder when the
    /// file is absent. A fabricated digest is the F12 class.
    #[test]
    fn file_sha256_digests_bytes_and_refuses_to_invent() {
        let dir = std::env::temp_dir().join("apr_parity001_sha");
        let _ = std::fs::create_dir_all(&dir);
        let f = dir.join("m.bin");
        std::fs::write(&f, b"abc").expect("write fixture");
        // sha256("abc") is a fixed, externally checkable constant.
        assert_eq!(
            file_sha256(&f).as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(
            file_sha256(&dir.join("does-not-exist.bin")),
            None,
            "a missing model must yield no digest — never a placeholder"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
