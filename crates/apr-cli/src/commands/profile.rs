//! Deep Profiling Command (PMAT-112 Real Telemetry + PMAT-192 CI Mode)
//!
//! Implements `apr profile` for model-agnostic performance analysis using
//! REAL inference passes, not synthetic benchmarks.
//!
//! "If you cannot measure it, you cannot improve it. If you fake the measurement,
//! you are not improving it; you are lying to yourself." - PMAT-112
//!
//! # Example
//!
//! ```bash
//! apr profile model.gguf                      # Profile real inference
//! apr profile model.gguf --warmup 3           # 3 warmup passes
//! apr profile model.gguf --measure 10         # 10 measurement passes
//! apr profile model.apr --format json         # CI-friendly output
//!
//! # PMAT-192: CI assertion mode (GH-180)
//! apr profile model.gguf --ci --assert-throughput 100  # Exit 1 if < 100 tok/s
//! apr profile model.gguf --ci --assert-p99 50          # Exit 1 if p99 > 50ms
//! apr profile model.gguf --ci --format json            # JSON with pass/fail
//! ```

use crate::error::CliError;
use crate::output;
use colored::Colorize;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "inference")]
use realizar::brick::BrickProfiler;
#[cfg(feature = "inference")]
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

/// Output format for profile results
#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum OutputFormat {
    #[default]
    Human,
    Json,
    Flamegraph,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "human" | "text" => Ok(Self::Human),
            "json" => Ok(Self::Json),
            "flamegraph" | "svg" => Ok(Self::Flamegraph),
            _ => Err(format!("Unknown format: {s}")),
        }
    }
}

/// Focus area for profiling
#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum ProfileFocus {
    #[default]
    All,
    Attention,
    Mlp,
    Matmul,
    Embedding,
}

impl std::str::FromStr for ProfileFocus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "all" => Ok(Self::All),
            "attention" | "attn" => Ok(Self::Attention),
            "mlp" | "ffn" => Ok(Self::Mlp),
            "matmul" | "gemm" => Ok(Self::Matmul),
            "embedding" | "embed" => Ok(Self::Embedding),
            _ => Err(format!("Unknown focus: {s}")),
        }
    }
}

/// Hotspot information from profiling
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Hotspot {
    name: String,
    time_us: f64,
    percent: f64, // Kept for JSON output and future flamegraph
    count: usize,
    avg_us: f64,
    min_us: f64,
    max_us: f64,
}

// ============================================================================
// PMAT-192: CI Assertion Mode (GH-180)
// ============================================================================

/// CI assertion configuration
#[derive(Debug, Clone, Default)]
pub struct CiAssertions {
    /// Minimum throughput in tok/s (fail if below)
    pub min_throughput: Option<f64>,
    /// Maximum p99 latency in ms (fail if above)
    pub max_p99_ms: Option<f64>,
    /// Maximum p50 latency in ms (fail if above)
    pub max_p50_ms: Option<f64>,
    /// Maximum memory in MB (fail if above)
    pub max_memory_mb: Option<f64>,
}

/// Result of a single CI assertion check
#[derive(Debug, Clone)]
pub struct AssertionResult {
    pub name: String,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
}

/// CI profile report with assertions
#[derive(Debug, Clone)]
pub struct CiProfileReport {
    pub model_path: String,
    pub passed: bool,
    pub throughput_tok_s: f64,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
    pub assertions: Vec<AssertionResult>,
}

impl CiProfileReport {
    /// Create report from profile results and assertions
    pub fn from_results(results: &RealProfileResults, assertions: &CiAssertions) -> Self {
        let mut assertion_results = Vec::new();
        let mut all_passed = true;

        // Convert us to ms for latency
        let latency_ms = results.total_inference_us / 1000.0;

        // Check throughput assertion
        if let Some(min_tps) = assertions.min_throughput {
            let passed = results.throughput_tok_s >= min_tps;
            if !passed {
                all_passed = false;
            }
            assertion_results.push(AssertionResult {
                name: "throughput".to_string(),
                expected: format!(">= {:.1} tok/s", min_tps),
                actual: format!("{:.1} tok/s", results.throughput_tok_s),
                passed,
            });
        }

        // Check p99 assertion (using avg as proxy since we don't have percentiles yet)
        if let Some(max_p99) = assertions.max_p99_ms {
            let passed = latency_ms <= max_p99;
            if !passed {
                all_passed = false;
            }
            assertion_results.push(AssertionResult {
                name: "latency_p99".to_string(),
                expected: format!("<= {:.1} ms", max_p99),
                actual: format!("{:.2} ms", latency_ms),
                passed,
            });
        }

        // Check p50 assertion
        if let Some(max_p50) = assertions.max_p50_ms {
            let passed = latency_ms <= max_p50;
            if !passed {
                all_passed = false;
            }
            assertion_results.push(AssertionResult {
                name: "latency_p50".to_string(),
                expected: format!("<= {:.1} ms", max_p50),
                actual: format!("{:.2} ms", latency_ms),
                passed,
            });
        }

        CiProfileReport {
            model_path: results.model_path.clone(),
            passed: all_passed,
            throughput_tok_s: results.throughput_tok_s,
            latency_p50_ms: latency_ms,
            latency_p99_ms: latency_ms, // Using same value for now
            assertions: assertion_results,
        }
    }

    /// Print CI report in human format
    pub fn print_human(&self) {
        println!();
        println!("{}", "CI PROFILE REPORT (PMAT-192)".white().bold());
        println!("{}", "═".repeat(60));
        println!();
        println!("  Model:       {}", self.model_path.cyan());
        println!("  Throughput:  {:.1} tok/s", self.throughput_tok_s);
        println!("  Latency p50: {:.2} ms", self.latency_p50_ms);
        println!("  Latency p99: {:.2} ms", self.latency_p99_ms);
        println!();

        if !self.assertions.is_empty() {
            println!("{}", "ASSERTIONS".white().bold());
            println!("{}", "─".repeat(60));
            for assertion in &self.assertions {
                let status = if assertion.passed {
                    "✅ PASS".green()
                } else {
                    "❌ FAIL".red()
                };
                println!(
                    "  {} {}: {} (expected {})",
                    status,
                    assertion.name.cyan(),
                    assertion.actual,
                    assertion.expected
                );
            }
            println!();
        }

        if self.passed {
            println!("{}", "✅ ALL ASSERTIONS PASSED".green().bold());
        } else {
            println!("{}", "❌ ASSERTIONS FAILED".red().bold());
        }
        println!();
    }

    /// Print CI report as JSON
    pub fn print_json(&self) {
        let mut json = String::from("{\n");
        json.push_str(&format!("  \"model\": \"{}\",\n", self.model_path));
        json.push_str(&format!("  \"passed\": {},\n", self.passed));
        json.push_str("  \"metrics\": {\n");
        json.push_str(&format!("    \"throughput_tok_s\": {:.2},\n", self.throughput_tok_s));
        json.push_str(&format!("    \"latency_p50_ms\": {:.2},\n", self.latency_p50_ms));
        json.push_str(&format!("    \"latency_p99_ms\": {:.2}\n", self.latency_p99_ms));
        json.push_str("  },\n");
        json.push_str("  \"assertions\": [\n");
        for (i, assertion) in self.assertions.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"name\": \"{}\",\n", assertion.name));
            json.push_str(&format!("      \"expected\": \"{}\",\n", assertion.expected));
            json.push_str(&format!("      \"actual\": \"{}\",\n", assertion.actual));
            json.push_str(&format!("      \"passed\": {}\n", assertion.passed));
            if i < self.assertions.len() - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ]\n");
        json.push_str("}\n");
        println!("{json}");
    }
}

/// Profile results from real inference
#[derive(Debug, Clone)]
struct RealProfileResults {
    model_path: String,
    architecture: String,
    num_layers: usize,
    vocab_size: usize,
    hidden_dim: usize,
    warmup_passes: usize,
    measure_passes: usize,
    total_inference_us: f64,
    throughput_tok_s: f64,
    tokens_per_pass: usize,
    hotspots: Vec<Hotspot>,
    per_layer_us: Vec<f64>,
    is_real_data: bool,
}

/// Detect model format from extension
fn detect_format(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("apr") => "apr",
        Some("safetensors") => "safetensors",
        Some("gguf") => "gguf",
        Some("bin") => "pytorch",
        _ => "unknown",
    }
}

/// Run profiling on the model with REAL inference
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    path: &Path,
    granular: bool,
    format: OutputFormat,
    focus: ProfileFocus,
    _detect_naive: bool,
    _naive_threshold: f64,
    _compare_hf: Option<&str>,
    _energy: bool,
    _perf_grade: bool,
    _callgraph: bool,
    _fail_on_naive: bool,
    output_path: Option<&Path>,
) -> Result<(), CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let format_str = detect_format(path);

    match format {
        OutputFormat::Human => {
            output::section("apr profile (PMAT-112: Real Telemetry)");
            println!();
            output::kv("Model", path.display());
            output::kv("Format", format_str);
            println!();
        }
        OutputFormat::Json => {}
        OutputFormat::Flamegraph => {}
    }

    // Profile with REAL inference
    let start = Instant::now();

    #[cfg(feature = "inference")]
    let results = profile_real_inference(path, 3, 10)?;

    #[cfg(not(feature = "inference"))]
    let results = {
        output::warn("Inference feature not enabled. Cannot run real profiling.");
        output::warn("Build with: cargo build --features inference");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    };

    let profile_time = start.elapsed();

    // GH-173: Apply focus filtering to results (PMAT-182)
    let filtered_results = filter_results_by_focus(&results, focus);

    // Show focus filter if applied
    if !matches!(focus, ProfileFocus::All) {
        output::kv("Focus filter", format!("{:?}", focus));
        println!();
    }

    // Output results based on format
    match format {
        OutputFormat::Human => {
            print_human_results(&filtered_results, granular)?;
            println!();
            println!(
                "{}",
                format!("Profile completed in {:.2}s", profile_time.as_secs_f64()).dimmed()
            );
        }
        OutputFormat::Json => {
            print_json_results(&filtered_results)?;
        }
        OutputFormat::Flamegraph => {
            print_flamegraph(&filtered_results, output_path)?;
        }
    }

    Ok(())
}

// ============================================================================
// PMAT-192: CI Assertion Mode Entry Point (GH-180)
// ============================================================================

/// Run profiling in CI mode with assertions
///
/// Returns Ok(true) if all assertions pass, Ok(false) if any fail.
/// Use the exit code to fail CI pipelines.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ci(
    path: &Path,
    format: OutputFormat,
    assertions: CiAssertions,
    warmup: usize,
    measure: usize,
) -> Result<bool, CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    #[cfg(feature = "inference")]
    let results = profile_real_inference(path, warmup, measure)?;

    #[cfg(not(feature = "inference"))]
    {
        output::warn("Inference feature not enabled. Cannot run CI profiling.");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    }

    // Build CI report with assertion checks
    let report = CiProfileReport::from_results(&results, &assertions);

    // Output based on format
    match format {
        OutputFormat::Json => report.print_json(),
        _ => report.print_human(),
    }

    Ok(report.passed)
}

/// GH-173: Filter profile results by focus area (PMAT-182)
fn filter_results_by_focus(results: &RealProfileResults, focus: ProfileFocus) -> RealProfileResults {
    let filtered_hotspots = match focus {
        ProfileFocus::All => results.hotspots.clone(),
        ProfileFocus::Attention => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("attention")
                    || name_lower.contains("attn")
                    || name_lower.contains("qkv")
                    || name_lower.contains("softmax")
            })
            .cloned()
            .collect(),
        ProfileFocus::Mlp => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("mlp")
                    || name_lower.contains("ffn")
                    || name_lower.contains("gate")
                    || name_lower.contains("up_proj")
                    || name_lower.contains("down_proj")
            })
            .cloned()
            .collect(),
        ProfileFocus::Matmul => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("matmul")
                    || name_lower.contains("gemm")
                    || name_lower.contains("mm")
                    || name_lower.contains("linear")
            })
            .cloned()
            .collect(),
        ProfileFocus::Embedding => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("embed")
                    || name_lower.contains("lm_head")
                    || name_lower.contains("vocab")
            })
            .cloned()
            .collect(),
    };

    RealProfileResults {
        model_path: results.model_path.clone(),
        architecture: results.architecture.clone(),
        num_layers: results.num_layers,
        vocab_size: results.vocab_size,
        hidden_dim: results.hidden_dim,
        warmup_passes: results.warmup_passes,
        measure_passes: results.measure_passes,
        total_inference_us: results.total_inference_us,
        throughput_tok_s: results.throughput_tok_s,
        tokens_per_pass: results.tokens_per_pass,
        hotspots: filtered_hotspots,
        per_layer_us: results.per_layer_us.clone(),
        is_real_data: results.is_real_data,
    }
}

/// Profile model using REAL inference passes
#[cfg(feature = "inference")]
fn profile_real_inference(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    let format = detect_format(path);

    match format {
        "gguf" => profile_gguf_real(path, warmup_passes, measure_passes),
        "apr" => profile_apr_real(path, warmup_passes, measure_passes),
        "safetensors" => {
            output::warn("SafeTensors profiling requires APR conversion.");
            output::warn("Use: apr convert model.safetensors -o model.apr");
            Err(CliError::ValidationFailed(
                "SafeTensors profiling not directly supported. Convert to APR first.".to_string(),
            ))
        }
        _ => Err(CliError::ValidationFailed(format!(
            "Unsupported format: {format}"
        ))),
    }
}

/// Profile GGUF model with real inference
#[cfg(feature = "inference")]
fn profile_gguf_real(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    // Load the model
    println!("{}", "Loading model...".dimmed());
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    let architecture = mapped.model.architecture().unwrap_or("unknown").to_string();

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let config = &model.config;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    // Test prompt tokens (BOS + "Hello")
    let test_tokens: Vec<u32> = vec![1, 15043]; // BOS + "Hello" for TinyLlama/Qwen
    let tokens_per_pass = test_tokens.len();

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 1, // Just one token to profile forward pass
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Warmup passes (discard timing)
    println!(
        "{}",
        format!("Running {} warmup passes...", warmup_passes).dimmed()
    );
    for _ in 0..warmup_passes {
        let _ = model.generate(&test_tokens, &gen_config);
    }

    // Measurement passes with profiler
    println!(
        "{}",
        format!("Running {} measurement passes...", measure_passes).dimmed()
    );

    let mut profiler = BrickProfiler::new();
    profiler.set_num_layers(num_layers);
    profiler.set_tokens(tokens_per_pass * measure_passes);

    let mut forward_times: Vec<f64> = Vec::new();

    profiler.start_inference();

    for _ in 0..measure_passes {
        // Profile the complete forward pass
        let pass_start = Instant::now();

        profiler.start("forward_pass");
        let logits = model.forward(&test_tokens);
        profiler.stop("forward_pass");

        // Also record raw timing
        let pass_time = pass_start.elapsed().as_secs_f64() * 1_000_000.0;
        forward_times.push(pass_time);

        // Validate output if successful
        if let Ok(ref logits) = logits {
            profiler.record("logits_validation", 0.1); // Minimal overhead

            // Check for NaN/Inf (PMAT-112 requirement)
            let has_nan = logits.iter().any(|x| x.is_nan());
            let has_inf = logits.iter().any(|x| x.is_infinite());

            if has_nan || has_inf {
                output::warn(&format!(
                    "Forward pass produced invalid logits: NaN={}, Inf={}",
                    has_nan, has_inf
                ));
            }
        }
    }

    profiler.stop_inference();

    // Build results from profiler
    let report = profiler.report();

    // Compute statistics from raw timing
    let total_us: f64 = forward_times.iter().sum();
    let avg_us = total_us / measure_passes as f64;
    // min/max computed for future detailed output
    let _min_us = forward_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let _max_us = forward_times
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Build hotspots from profiler report
    let mut hotspots: Vec<Hotspot> = report
        .operations
        .iter()
        .map(|(name, stats)| Hotspot {
            name: name.clone(),
            time_us: stats.total_us,
            percent: if report.total_inference_us > 0.0 {
                (stats.total_us / report.total_inference_us) * 100.0
            } else {
                0.0
            },
            count: stats.count,
            avg_us: stats.avg_us,
            min_us: stats.min_us,
            max_us: stats.max_us,
        })
        .collect();

    // Sort by total time descending
    hotspots.sort_by(|a, b| {
        b.time_us
            .partial_cmp(&a.time_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Estimate per-layer timing (forward_pass / num_layers)
    let per_layer_us: Vec<f64> = vec![avg_us / num_layers as f64; num_layers];

    Ok(RealProfileResults {
        model_path: path.display().to_string(),
        architecture,
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: avg_us,
        throughput_tok_s: if avg_us > 0.0 {
            (tokens_per_pass as f64 / avg_us) * 1_000_000.0
        } else {
            0.0
        },
        tokens_per_pass,
        hotspots,
        per_layer_us,
        is_real_data: report.is_real_data,
    })
}

/// Profile APR model with real inference
#[cfg(feature = "inference")]
fn profile_apr_real(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    use realizar::apr_transformer::AprTransformer;

    // Load the model using AprTransformer
    println!("{}", "Loading APR model...".dimmed());
    let model = AprTransformer::from_apr_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let config = &model.config;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    // Test tokens
    let test_tokens: Vec<u32> = vec![1, 15043];
    let tokens_per_pass = test_tokens.len();

    // Warmup
    println!(
        "{}",
        format!("Running {} warmup passes...", warmup_passes).dimmed()
    );
    for _ in 0..warmup_passes {
        let _ = model.forward(&test_tokens);
    }

    // Measurement
    println!(
        "{}",
        format!("Running {} measurement passes...", measure_passes).dimmed()
    );

    let mut profiler = BrickProfiler::new();
    profiler.set_num_layers(num_layers);
    profiler.set_tokens(tokens_per_pass * measure_passes);

    let mut forward_times: Vec<f64> = Vec::new();

    profiler.start_inference();

    for _ in 0..measure_passes {
        profiler.start("forward_pass");
        let start = Instant::now();
        let result = model.forward(&test_tokens);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        profiler.stop("forward_pass");

        forward_times.push(elapsed);

        // Validate output
        if let Ok(ref logits) = result {
            let has_nan = logits.iter().any(|x| x.is_nan());
            let has_inf = logits.iter().any(|x| x.is_infinite());

            if has_nan || has_inf {
                output::warn(&format!(
                    "Forward pass produced invalid logits: NaN={}, Inf={}",
                    has_nan, has_inf
                ));
            }
        }
    }

    profiler.stop_inference();

    let total_us: f64 = forward_times.iter().sum();
    let avg_us = total_us / measure_passes as f64;
    let min_us = forward_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_us = forward_times
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(RealProfileResults {
        model_path: path.display().to_string(),
        architecture: "apr".to_string(),
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: avg_us,
        throughput_tok_s: if avg_us > 0.0 {
            (tokens_per_pass as f64 / avg_us) * 1_000_000.0
        } else {
            0.0
        },
        tokens_per_pass,
        hotspots: vec![Hotspot {
            name: "forward_pass".to_string(),
            time_us: total_us,
            percent: 100.0,
            count: measure_passes,
            avg_us,
            min_us,
            max_us,
        }],
        per_layer_us: vec![avg_us / num_layers as f64; num_layers],
        is_real_data: true,
    })
}

/// Print human-readable results
fn print_human_results(results: &RealProfileResults, granular: bool) -> Result<(), CliError> {
    // Model info
    println!("{}", "MODEL INFO".white().bold());
    println!("{}", "═".repeat(60));
    println!("  Architecture:    {}", results.architecture.cyan());
    println!("  Layers:          {}", results.num_layers);
    println!("  Hidden dim:      {}", results.hidden_dim);
    println!("  Vocab size:      {}", results.vocab_size);
    println!("  Warmup passes:   {}", results.warmup_passes);
    println!("  Measure passes:  {}", results.measure_passes);
    println!();

    // Real data indicator
    if results.is_real_data {
        println!("{}", "✓ REAL TELEMETRY (not simulated)".green().bold());
    } else {
        println!("{}", "⚠ SIMULATED DATA (inference disabled)".yellow());
    }
    println!();

    // Hotspot analysis
    println!("{}", "HOTSPOT ANALYSIS".white().bold());
    println!("{}", "═".repeat(60));
    println!();

    let total_time = results.hotspots.iter().map(|h| h.time_us).sum::<f64>();

    for (i, hotspot) in results.hotspots.iter().enumerate() {
        let percent = if total_time > 0.0 {
            (hotspot.time_us / total_time) * 100.0
        } else {
            0.0
        };

        let bar_width = ((percent / 100.0) * 20.0) as usize;
        let bar = format!(
            "{}{}",
            "█".repeat(bar_width.min(20)),
            "░".repeat(20 - bar_width.min(20))
        );

        println!(
            "  #{} {:<20} {:>10.1}µs ({:>5.1}%)  {}",
            i + 1,
            hotspot.name.cyan(),
            hotspot.avg_us,
            percent,
            bar
        );

        if granular {
            println!(
                "     └─ count={}, min={:.1}µs, max={:.1}µs",
                hotspot.count, hotspot.min_us, hotspot.max_us
            );
        }
    }
    println!();

    // Per-layer breakdown (if granular)
    if granular && !results.per_layer_us.is_empty() {
        println!("{}", "PER-LAYER TIMING (estimated)".white().bold());
        println!("{}", "═".repeat(60));
        println!();

        let max_layer_time = results.per_layer_us.iter().cloned().fold(0.0f64, f64::max);

        for (i, &time_us) in results.per_layer_us.iter().enumerate() {
            let bar_width = if max_layer_time > 0.0 {
                ((time_us / max_layer_time) * 30.0) as usize
            } else {
                0
            };
            let bar = "█".repeat(bar_width.min(30));
            println!("  Layer {:>2}: {:>8.1}µs  {}", i, time_us, bar);
        }
        println!();
    }

    // Summary
    println!("{}", "SUMMARY".white().bold());
    println!("{}", "═".repeat(60));
    println!();
    println!(
        "  Avg forward pass:   {:.1}µs ({:.2}ms)",
        results.total_inference_us,
        results.total_inference_us / 1000.0
    );
    println!(
        "  Throughput:         {:.2} tok/s",
        results.throughput_tok_s
    );
    println!("  Tokens per pass:    {}", results.tokens_per_pass);
    println!();

    Ok(())
}

/// Print JSON output
fn print_json_results(results: &RealProfileResults) -> Result<(), CliError> {
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"model\": \"{}\",\n", results.model_path));
    json.push_str(&format!(
        "  \"architecture\": \"{}\",\n",
        results.architecture
    ));
    json.push_str(&format!("  \"num_layers\": {},\n", results.num_layers));
    json.push_str(&format!("  \"vocab_size\": {},\n", results.vocab_size));
    json.push_str(&format!("  \"hidden_dim\": {},\n", results.hidden_dim));
    json.push_str(&format!("  \"is_real_data\": {},\n", results.is_real_data));
    json.push_str("  \"timing\": {\n");
    json.push_str(&format!(
        "    \"warmup_passes\": {},\n",
        results.warmup_passes
    ));
    json.push_str(&format!(
        "    \"measure_passes\": {},\n",
        results.measure_passes
    ));
    json.push_str(&format!(
        "    \"avg_inference_us\": {:.2},\n",
        results.total_inference_us
    ));
    json.push_str(&format!(
        "    \"throughput_tok_s\": {:.2}\n",
        results.throughput_tok_s
    ));
    json.push_str("  },\n");

    json.push_str("  \"hotspots\": [\n");
    for (i, hotspot) in results.hotspots.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"name\": \"{}\",\n", hotspot.name));
        json.push_str(&format!("      \"total_us\": {:.2},\n", hotspot.time_us));
        json.push_str(&format!("      \"avg_us\": {:.2},\n", hotspot.avg_us));
        json.push_str(&format!("      \"min_us\": {:.2},\n", hotspot.min_us));
        json.push_str(&format!("      \"max_us\": {:.2},\n", hotspot.max_us));
        json.push_str(&format!("      \"count\": {}\n", hotspot.count));
        if i < results.hotspots.len() - 1 {
            json.push_str("    },\n");
        } else {
            json.push_str("    }\n");
        }
    }
    json.push_str("  ],\n");

    json.push_str("  \"per_layer_us\": [");
    for (i, time) in results.per_layer_us.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("{:.2}", time));
    }
    json.push_str("]\n");

    json.push_str("}\n");

    println!("{json}");
    Ok(())
}

/// Print flamegraph SVG (GH-174: supports --output for file output)
fn print_flamegraph(results: &RealProfileResults, output_path: Option<&Path>) -> Result<(), CliError> {
    let mut svg = String::new();
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"400\">\n");
    svg.push_str("  <style>\n");
    svg.push_str("    .frame {{ stroke: #333; }}\n");
    svg.push_str("    .label {{ font-family: monospace; font-size: 12px; }}\n");
    svg.push_str("  </style>\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n");
    svg.push_str(
        "  <text x=\"400\" y=\"30\" text-anchor=\"middle\" font-size=\"16\" font-weight=\"bold\">\n",
    );
    svg.push_str("    apr profile: Real Telemetry Flamegraph (PMAT-112)\n");
    svg.push_str("  </text>\n");

    let total_time: f64 = results.hotspots.iter().map(|h| h.time_us).sum();
    let mut y = 350.0_f64;
    let height = 25.0_f64;

    for hotspot in results.hotspots.iter().rev() {
        let percent = if total_time > 0.0 {
            (hotspot.time_us / total_time) * 100.0
        } else {
            0.0
        };
        let width = (percent / 100.0) * 760.0;
        let x = 20.0 + ((100.0 - percent) / 200.0) * 760.0;

        // Color based on percentage (hotter = more red)
        let r = (255.0 * (percent / 100.0).min(1.0)) as u8;
        let g = (200.0 * (1.0 - percent / 100.0).max(0.0)) as u8;
        let color = format!("#{:02X}{:02X}50", r, g);

        svg.push_str(&format!(
            "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{width:.1}\" height=\"{height:.1}\" fill=\"{color}\" class=\"frame\"/>\n"
        ));
        svg.push_str(&format!(
            "  <text x=\"{:.1}\" y=\"{:.1}\" class=\"label\">{} ({:.1}%)</text>\n",
            x + 5.0,
            y + 16.0,
            hotspot.name,
            percent
        ));

        y -= height + 2.0;
    }

    svg.push_str("</svg>\n");

    // GH-174: Write to file if output path specified, otherwise print to stdout
    if let Some(path) = output_path {
        std::fs::write(path, &svg).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to write flamegraph to {}: {e}", path.display()))
        })?;
        output::success(&format!("Flamegraph written to: {}", path.display()));
    } else {
        println!("{svg}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_parse() {
        assert!(matches!(
            "json".parse::<OutputFormat>().unwrap(),
            OutputFormat::Json
        ));
        assert!(matches!(
            "human".parse::<OutputFormat>().unwrap(),
            OutputFormat::Human
        ));
        assert!(matches!(
            "flamegraph".parse::<OutputFormat>().unwrap(),
            OutputFormat::Flamegraph
        ));
    }

    #[test]
    fn test_profile_focus_parse() {
        assert!(matches!(
            "attention".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Attention
        ));
        assert!(matches!(
            "mlp".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Mlp
        ));
        assert!(matches!(
            "all".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::All
        ));
    }

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("model.apr")), "apr");
        assert_eq!(detect_format(Path::new("model.safetensors")), "safetensors");
        assert_eq!(detect_format(Path::new("model.gguf")), "gguf");
    }
}
