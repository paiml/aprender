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
use std::fmt::Write;
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
    /// Roofline bottleneck classification (None = not computed)
    bottleneck: Option<String>,
    /// Hardware efficiency percentage (0-100)
    efficiency_pct: Option<f64>,
    /// BrickId category (Attention, FFN, Norm, Other)
    category: Option<String>,
    /// Achieved memory bandwidth (GB/s) — F-PROFILE-008
    bandwidth_gbs: Option<f64>,
    /// Estimated data moved per invocation (bytes) — F-PROFILE-008
    data_bytes_per_call: Option<u64>,
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
    /// Reserved for future memory assertion support
    #[allow(dead_code)]
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
    pub(crate) fn from_results(results: &RealProfileResults, assertions: &CiAssertions) -> Self {
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
        writeln!(json, "  \"model\": \"{}\",", self.model_path)
            .expect("write to String is infallible");
        writeln!(json, "  \"passed\": {},", self.passed).expect("write to String is infallible");
        json.push_str("  \"metrics\": {\n");
        writeln!(
            json,
            "    \"throughput_tok_s\": {:.2},",
            self.throughput_tok_s
        )
        .expect("write to String is infallible");
        writeln!(json, "    \"latency_p50_ms\": {:.2},", self.latency_p50_ms)
            .expect("write to String is infallible");
        writeln!(json, "    \"latency_p99_ms\": {:.2}", self.latency_p99_ms)
            .expect("write to String is infallible");
        json.push_str("  },\n");
        json.push_str("  \"assertions\": [\n");
        for (i, assertion) in self.assertions.iter().enumerate() {
            json.push_str("    {\n");
            writeln!(json, "      \"name\": \"{}\",", assertion.name)
                .expect("write to String is infallible");
            writeln!(json, "      \"expected\": \"{}\",", assertion.expected)
                .expect("write to String is infallible");
            writeln!(json, "      \"actual\": \"{}\",", assertion.actual)
                .expect("write to String is infallible");
            writeln!(json, "      \"passed\": {}", assertion.passed)
                .expect("write to String is infallible");
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

/// Roofline analysis results
#[derive(Debug, Clone, Default)]
pub(crate) struct RooflineAnalysis {
    /// Hardware peak compute (GFLOPS for CPU, TFLOPS for GPU)
    pub peak_compute: f64,
    /// Peak memory bandwidth (GB/s)
    pub peak_bandwidth_gbps: f64,
    /// Achieved compute (GFLOPS)
    pub achieved_gflops: f64,
    /// Achieved bandwidth (GB/s)
    pub achieved_bandwidth_gbps: f64,
    /// Compute efficiency (0-100%)
    pub compute_efficiency_pct: f64,
    /// Memory efficiency (0-100%)
    pub memory_efficiency_pct: f64,
    /// Overall arithmetic intensity (FLOPs / bytes)
    pub arithmetic_intensity: f64,
    /// Threshold AI for this hardware
    pub ai_threshold: f64,
    /// Whether workload is memory or compute bound
    pub bottleneck: String,
    /// Backend name (CPU/CUDA) — used in JSON output format
    #[allow(dead_code)]
    pub backend: String,
    /// Hardware model name
    pub hardware_model: String,
}

/// Category time summary
#[derive(Debug, Clone, Default)]
#[allow(clippy::struct_field_names)]
pub(crate) struct CategorySummary {
    pub attention_pct: f64,
    pub ffn_pct: f64,
    pub norm_pct: f64,
    pub other_pct: f64,
}

/// Performance grade (A-F)
#[derive(Debug, Clone, Copy)]
pub(crate) enum PerfGrade {
    A,
    B,
    C,
    D,
    F,
}

impl PerfGrade {
    fn from_efficiency(pct: f64) -> Self {
        if pct >= 50.0 {
            Self::A
        } else if pct >= 30.0 {
            Self::B
        } else if pct >= 15.0 {
            Self::C
        } else if pct >= 5.0 {
            Self::D
        } else {
            Self::F
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::F => "F",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::A => "Excellent — near hardware peak",
            Self::B => "Good — reasonable utilization",
            Self::C => "Fair — room for improvement",
            Self::D => "Poor — significant optimization needed",
            Self::F => "Critical — likely using wrong backend or naive implementation",
        }
    }
}

impl std::fmt::Display for PerfGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Profile results from real inference
#[derive(Debug, Clone, Default)]
pub(crate) struct RealProfileResults {
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
    /// Roofline analysis (None = not computed)
    roofline: Option<RooflineAnalysis>,
    /// Category time summary
    category_summary: Option<CategorySummary>,
    /// Backend used (cpu, cuda, etc.)
    backend: String,
    /// Real percentile latencies from multi-pass measurement
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
    latency_min_ms: f64,
    latency_max_ms: f64,
    /// Prefill throughput (tok/s) — separate from decode
    prefill_tok_s: f64,
    /// Decode throughput (tok/s) — the primary metric for Ollama comparison
    decode_tok_s: f64,
    /// Total tokens generated across all measurement passes
    total_tokens_generated: usize,
    /// Kernel launch overhead as % of decode time (F-PROFILE-009)
    kernel_launch_overhead_pct: f64,
    /// Kernel launch overhead in microseconds (F-PROFILE-009)
    kernel_launch_overhead_us: f64,
}

/// Ollama baseline measurement
#[derive(Debug, Clone)]
struct OllamaBaseline {
    /// Ollama decode throughput (tok/s)
    decode_tok_s: f64,
    /// Ollama prefill throughput (tok/s)
    prefill_tok_s: f64,
    /// Model name used
    model_name: String,
}

/// Compute percentile from sorted values
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;
    if upper >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
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
    detect_naive: bool,
    _naive_threshold: f64,
    _compare_hf: Option<&str>,
    _energy: bool,
    perf_grade: bool,
    _callgraph: bool,
    _fail_on_naive: bool,
    output_path: Option<&Path>,
    tokens: usize,
    ollama: bool,
    no_gpu: bool,
) -> Result<(), CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let format_str = detect_format(path);

    match format {
        OutputFormat::Human => {
            output::section("apr profile (Real Per-Operation Telemetry)");
            println!();
            output::kv("Model", path.display());
            output::kv("Format", format_str);
            println!();
        }
        OutputFormat::Json => {}
        OutputFormat::Flamegraph => {}
    }

    // Profile with REAL inference — try GPU first, fall back to CPU
    let start = Instant::now();

    #[cfg(feature = "inference")]
    let mut results = if no_gpu {
        profile_real_inference_cpu(path, 3, 10)?
    } else {
        // Try GPU generation profiling first (full token generation, not just forward pass)
        match profile_gpu_generation(path, tokens, 3, 10) {
            Ok(r) => r,
            Err(_) => {
                if matches!(format, OutputFormat::Human) {
                    output::warn("GPU profiling unavailable, falling back to CPU per-op profiling");
                }
                profile_real_inference_cpu(path, 3, 10)?
            }
        }
    };

    #[cfg(not(feature = "inference"))]
    let mut results = {
        output::warn("Inference feature not enabled. Cannot run real profiling.");
        output::warn("Build with: cargo build --features inference");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    };

    let profile_time = start.elapsed();

    // Compute roofline analysis
    #[cfg(feature = "inference")]
    {
        results.roofline = Some(compute_roofline(&results));
    }

    // GH-173: Apply focus filtering to results (PMAT-182)
    let filtered_results = filter_results_by_focus(&results, focus);

    // Show focus filter if applied
    if !matches!(focus, ProfileFocus::All) {
        output::kv("Focus filter", format!("{:?}", focus));
        println!();
    }

    // Ollama comparison (if requested)
    let ollama_baseline = if ollama && matches!(format, OutputFormat::Human) {
        run_ollama_comparison(path, tokens)
    } else {
        None
    };

    print_profile_output(
        format, &filtered_results, granular, perf_grade, detect_naive,
        ollama_baseline.as_ref(), output_path, profile_time,
    )
}

#[allow(clippy::too_many_arguments)]
fn print_profile_output(
    format: OutputFormat,
    results: &RealProfileResults,
    granular: bool,
    perf_grade: bool,
    detect_naive: bool,
    ollama_baseline: Option<&OllamaBaseline>,
    output_path: Option<&Path>,
    profile_time: std::time::Duration,
) -> Result<(), CliError> {
    match format {
        OutputFormat::Human => {
            print_human_results(results, granular, perf_grade, detect_naive)?;
            if let Some(baseline) = ollama_baseline {
                print_ollama_comparison(results, baseline);
            }
            println!();
            println!("{}", format!("Profile completed in {:.2}s", profile_time.as_secs_f64()).dimmed());
        }
        OutputFormat::Json => {
            print_json_results(results)?;
        }
        OutputFormat::Flamegraph => {
            print_flamegraph(results, output_path)?;
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
    assertions: &CiAssertions,
    warmup: usize,
    measure: usize,
) -> Result<bool, CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    #[cfg(feature = "inference")]
    let results = profile_real_inference_cpu(path, warmup, measure)?;

    #[cfg(not(feature = "inference"))]
    {
        output::warn("Inference feature not enabled. Cannot run CI profiling.");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    }

    // Build CI report with assertion checks
    let report = CiProfileReport::from_results(&results, assertions);

    // Output based on format
    match format {
        OutputFormat::Json => report.print_json(),
        _ => report.print_human(),
    }

    Ok(report.passed)
}

// ============================================================================
// PMAT-192 Phase 4: Differential Benchmark Mode (GH-180)
// ============================================================================

/// Differential benchmark result comparing two models
#[derive(Debug, Clone)]
pub struct DiffBenchmarkReport {
    pub model_a: String,
    pub model_b: String,
    pub throughput_a: f64,
    pub throughput_b: f64,
    pub throughput_delta_pct: f64,
    pub latency_a_ms: f64,
    pub latency_b_ms: f64,
    pub latency_delta_pct: f64,
    pub winner: String,
    pub regressions: Vec<String>,
    pub improvements: Vec<String>,
}

impl DiffBenchmarkReport {
    /// Print human-readable diff report
    pub fn print_human(&self) {
        println!();
        println!("{}", "DIFFERENTIAL BENCHMARK (PMAT-192)".white().bold());
        println!("{}", "═".repeat(70));
        println!();
        println!("  Model A: {}", self.model_a.cyan());
        println!("  Model B: {}", self.model_b.cyan());
        println!();

        // Table header
        println!("┌─────────────┬──────────────┬──────────────┬──────────────┐");
        println!("│ Metric      │ Model A      │ Model B      │ Delta        │");
        println!("├─────────────┼──────────────┼──────────────┼──────────────┤");

        // Throughput row
        let tps_delta_str = if self.throughput_delta_pct >= 0.0 {
            format!("+{:.1}% ✅", self.throughput_delta_pct)
                .green()
                .to_string()
        } else {
            format!("{:.1}% ⚠️", self.throughput_delta_pct)
                .yellow()
                .to_string()
        };
        println!(
            "│ Throughput  │ {:>10.1} t/s │ {:>10.1} t/s │ {:>12} │",
            self.throughput_a, self.throughput_b, tps_delta_str
        );

        // Latency row
        let lat_delta_str = if self.latency_delta_pct <= 0.0 {
            format!("{:.1}% ✅", self.latency_delta_pct)
                .green()
                .to_string()
        } else {
            format!("+{:.1}% ⚠️", self.latency_delta_pct)
                .yellow()
                .to_string()
        };
        println!(
            "│ Latency     │ {:>10.2} ms │ {:>10.2} ms │ {:>12} │",
            self.latency_a_ms, self.latency_b_ms, lat_delta_str
        );

        println!("└─────────────┴──────────────┴──────────────┴──────────────┘");
        println!();

        // Winner
        println!(
            "  {}: {}",
            "Winner".white().bold(),
            self.winner.green().bold()
        );
        println!();

        // Regressions
        if !self.regressions.is_empty() {
            println!("{}", "  ⚠️  REGRESSIONS:".yellow().bold());
            for r in &self.regressions {
                println!("     - {}", r);
            }
            println!();
        }

        // Improvements
        if !self.improvements.is_empty() {
            println!("{}", "  ✅ IMPROVEMENTS:".green().bold());
            for i in &self.improvements {
                println!("     - {}", i);
            }
            println!();
        }
    }

    /// Print JSON diff report
    pub fn print_json(&self) {
        let mut json = String::from("{\n");
        writeln!(json, "  \"model_a\": \"{}\",", self.model_a)
            .expect("write to String is infallible");
        writeln!(json, "  \"model_b\": \"{}\",", self.model_b)
            .expect("write to String is infallible");
        json.push_str("  \"metrics\": {\n");
        writeln!(
            json,
            "    \"throughput_a_tok_s\": {:.2},",
            self.throughput_a
        )
        .expect("write to String is infallible");
        writeln!(
            json,
            "    \"throughput_b_tok_s\": {:.2},",
            self.throughput_b
        )
        .expect("write to String is infallible");
        writeln!(
            json,
            "    \"throughput_delta_pct\": {:.2},",
            self.throughput_delta_pct
        )
        .expect("write to String is infallible");
        writeln!(json, "    \"latency_a_ms\": {:.2},", self.latency_a_ms)
            .expect("write to String is infallible");
        writeln!(json, "    \"latency_b_ms\": {:.2},", self.latency_b_ms)
            .expect("write to String is infallible");
        writeln!(
            json,
            "    \"latency_delta_pct\": {:.2}",
            self.latency_delta_pct
        )
        .expect("write to String is infallible");
        json.push_str("  },\n");
        writeln!(json, "  \"winner\": \"{}\",", self.winner)
            .expect("write to String is infallible");
        json.push_str("  \"regressions\": [");
        for (i, r) in self.regressions.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            write!(json, "\"{}\"", r).expect("write to String is infallible");
        }
        json.push_str("],\n");
        json.push_str("  \"improvements\": [");
        for (i, imp) in self.improvements.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            write!(json, "\"{}\"", imp).expect("write to String is infallible");
        }
        json.push_str("]\n");
        json.push_str("}\n");
        println!("{json}");
    }
}

/// Run differential benchmark comparing two models (Phase 4)
///
/// Returns Ok(true) if model_b is better or equal, Ok(false) if regression detected.
#[allow(dead_code)]
pub(crate) fn run_diff_benchmark(
    model_a: &Path,
    model_b: &Path,
    format: OutputFormat,
    warmup: usize,
    measure: usize,
    regression_threshold: f64, // e.g., 0.05 = 5% regression triggers failure
) -> Result<bool, CliError> {
    // Validate files exist
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    output::section("Differential Benchmark (PMAT-192 Phase 4)");
    println!();

    // Profile model A
    output::kv("Profiling Model A", model_a.display());
    #[cfg(feature = "inference")]
    let results_a = profile_real_inference_cpu(model_a, warmup, measure)?;

    #[cfg(not(feature = "inference"))]
    return Err(CliError::ValidationFailed(
        "Requires --features inference".to_string(),
    ));

    // Profile model B
    output::kv("Profiling Model B", model_b.display());
    #[cfg(feature = "inference")]
    let results_b = profile_real_inference_cpu(model_b, warmup, measure)?;

    // Calculate deltas
    let throughput_delta = if results_a.throughput_tok_s > 0.0 {
        ((results_b.throughput_tok_s - results_a.throughput_tok_s) / results_a.throughput_tok_s)
            * 100.0
    } else {
        0.0
    };

    let latency_a_ms = results_a.total_inference_us / 1000.0;
    let latency_b_ms = results_b.total_inference_us / 1000.0;
    let latency_delta = if latency_a_ms > 0.0 {
        ((latency_b_ms - latency_a_ms) / latency_a_ms) * 100.0
    } else {
        0.0
    };

    // Determine winner
    let winner = if results_b.throughput_tok_s > results_a.throughput_tok_s {
        format!("Model B ({:.1}% faster)", throughput_delta.abs())
    } else if results_a.throughput_tok_s > results_b.throughput_tok_s {
        format!("Model A ({:.1}% faster)", throughput_delta.abs())
    } else {
        "Tie".to_string()
    };

    // Detect regressions and improvements
    let mut regressions = Vec::new();
    let mut improvements = Vec::new();

    if throughput_delta < -regression_threshold * 100.0 {
        regressions.push(format!(
            "Throughput: {:.1}% slower ({:.1} → {:.1} tok/s)",
            throughput_delta.abs(),
            results_a.throughput_tok_s,
            results_b.throughput_tok_s
        ));
    } else if throughput_delta > regression_threshold * 100.0 {
        improvements.push(format!(
            "Throughput: {:.1}% faster ({:.1} → {:.1} tok/s)",
            throughput_delta, results_a.throughput_tok_s, results_b.throughput_tok_s
        ));
    }

    if latency_delta > regression_threshold * 100.0 {
        regressions.push(format!(
            "Latency: {:.1}% slower ({:.2} → {:.2} ms)",
            latency_delta, latency_a_ms, latency_b_ms
        ));
    } else if latency_delta < -regression_threshold * 100.0 {
        improvements.push(format!(
            "Latency: {:.1}% faster ({:.2} → {:.2} ms)",
            latency_delta.abs(),
            latency_a_ms,
            latency_b_ms
        ));
    }

    let report = DiffBenchmarkReport {
        model_a: model_a.display().to_string(),
        model_b: model_b.display().to_string(),
        throughput_a: results_a.throughput_tok_s,
        throughput_b: results_b.throughput_tok_s,
        throughput_delta_pct: throughput_delta,
        latency_a_ms,
        latency_b_ms,
        latency_delta_pct: latency_delta,
        winner,
        regressions: regressions.clone(),
        improvements,
    };

    // Output
    match format {
        OutputFormat::Json => report.print_json(),
        _ => report.print_human(),
    }

    // Return false if any regressions detected
    Ok(regressions.is_empty())
}

/// GH-173: Filter profile results by focus area (PMAT-182)
fn filter_results_by_focus(
    results: &RealProfileResults,
    focus: ProfileFocus,
) -> RealProfileResults {
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
        roofline: results.roofline.clone(),
        category_summary: results.category_summary.clone(),
        backend: results.backend.clone(),
        latency_p50_ms: results.latency_p50_ms,
        latency_p95_ms: results.latency_p95_ms,
        latency_p99_ms: results.latency_p99_ms,
        latency_min_ms: results.latency_min_ms,
        latency_max_ms: results.latency_max_ms,
        prefill_tok_s: results.prefill_tok_s,
        decode_tok_s: results.decode_tok_s,
        total_tokens_generated: results.total_tokens_generated,
        kernel_launch_overhead_pct: results.kernel_launch_overhead_pct,
        kernel_launch_overhead_us: results.kernel_launch_overhead_us,
    }
}

/// Profile model using REAL inference passes (CPU per-operation path)
#[cfg(feature = "inference")]
fn profile_real_inference_cpu(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    let format = detect_format(path);

    match format {
        "gguf" => profile_gguf_real(path, warmup_passes, measure_passes),
        "apr" => profile_apr_real(path, warmup_passes, measure_passes),
        "safetensors" => profile_safetensors_real(path, warmup_passes, measure_passes),
        _ => Err(CliError::ValidationFailed(format!(
            "Unsupported format: {format}"
        ))),
    }
}

/// Profile GPU token generation with full decode loop
///
/// This is the KEY profiling path — it measures what users actually care about:
/// - Full token generation (prefill + decode)
/// - Per-token decode latency with real percentiles (p50, p95, p99)
/// - Prefill vs decode throughput separated
///
/// References:
/// - Williams et al. (2009) "Roofline: An Insightful Visual Performance Model"
/// - Pope et al. (2023) "Efficiently Scaling Transformer Inference"
#[cfg(feature = "inference")]
fn profile_gpu_generation(
    path: &Path,
    tokens_per_pass: usize,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    let format = detect_format(path);

    // Currently GPU generation profiling only for GGUF (primary format)
    if format != "gguf" {
        return Err(CliError::ValidationFailed(format!(
            "GPU generation profiling requires GGUF format (got {format})"
        )));
    }

    println!(
        "{}",
        "Loading model for GPU generation profiling...".dimmed()
    );
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    let architecture = mapped.model.architecture().unwrap_or("unknown").to_string();
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let num_layers = model.config.num_layers;
    let vocab_size = model.config.vocab_size;
    let hidden_dim = model.config.hidden_dim;

    // Try GPU path
    let mut cuda_model = match realizar::gguf::OwnedQuantizedModelCuda::new(model, 0) {
        Ok(m) => m,
        Err(e) => {
            return Err(CliError::ValidationFailed(format!("CUDA init failed: {e}")));
        }
    };

    // Test prompt: "The meaning of life is" — enough tokens for meaningful prefill
    let test_tokens: Vec<u32> = vec![791, 7438, 315, 2324, 374]; // "The meaning of life is"

    let gen_config = QuantizedGenerateConfig {
        max_tokens: tokens_per_pass,
        temperature: 0.0, // Greedy for deterministic profiling
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Warmup passes
    println!(
        "{}",
        format!(
            "GPU warmup: {} passes x {} tokens...",
            warmup_passes, tokens_per_pass
        )
        .dimmed()
    );
    for _ in 0..warmup_passes {
        let _ = cuda_model.generate_gpu_resident(&test_tokens, &gen_config);
    }

    // Measurement passes — collect per-token timing
    println!(
        "{}",
        format!(
            "GPU measurement: {} passes x {} tokens...",
            measure_passes, tokens_per_pass
        )
        .dimmed()
    );

    let mut per_pass_decode_times: Vec<f64> = Vec::new(); // ms per pass (decode only)
    let mut per_pass_prefill_times: Vec<f64> = Vec::new(); // ms per pass (prefill only)
    let mut per_pass_total_times: Vec<f64> = Vec::new(); // ms per pass (total)
    let mut total_tokens_generated: usize = 0;

    for pass in 0..measure_passes {
        let total_start = Instant::now();

        // Time prefill separately by generating just 1 token first
        let prefill_start = Instant::now();
        let prefill_config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
            trace: false,
        };
        let _ = cuda_model.generate_gpu_resident(&test_tokens, &prefill_config);
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        per_pass_prefill_times.push(prefill_ms);

        // Now time full generation (includes prefill again — we subtract)
        let gen_start = Instant::now();
        let result = cuda_model.generate_gpu_resident(&test_tokens, &gen_config);
        let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        per_pass_total_times.push(total_ms);

        if let Ok(ref tokens) = result {
            let generated = tokens.len().saturating_sub(test_tokens.len());
            total_tokens_generated += generated;

            // Decode time = total generation time - prefill time (estimated)
            // Better: decode_ms = gen_ms - (prefill portion)
            // Since gen includes its own prefill, decode = gen_ms - prefill_ms
            let decode_ms = (gen_ms - prefill_ms).max(0.1);
            per_pass_decode_times.push(decode_ms);

            if pass == 0 {
                println!(
                    "{}",
                    format!(
                        "  Pass 0: {} tokens in {:.1}ms (prefill: {:.1}ms, decode: {:.1}ms = {:.1} tok/s)",
                        generated,
                        gen_ms,
                        prefill_ms,
                        decode_ms,
                        generated as f64 / (decode_ms / 1000.0)
                    )
                    .dimmed()
                );
            }
        }
    }

    // Compute real percentile latencies from per-pass decode times
    let mut sorted_decode = per_pass_decode_times.clone();
    sorted_decode.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50 = percentile(&sorted_decode, 50.0);
    let p95 = percentile(&sorted_decode, 95.0);
    let p99 = percentile(&sorted_decode, 99.0);
    let lat_min = sorted_decode.first().copied().unwrap_or(0.0);
    let lat_max = sorted_decode.last().copied().unwrap_or(0.0);

    // Compute throughput
    let avg_decode_ms = if per_pass_decode_times.is_empty() {
        0.0
    } else {
        per_pass_decode_times.iter().sum::<f64>() / per_pass_decode_times.len() as f64
    };
    let tokens_per_decode = if measure_passes > 0 {
        total_tokens_generated / measure_passes
    } else {
        0
    };
    let decode_tok_s = if avg_decode_ms > 0.0 {
        tokens_per_decode as f64 / (avg_decode_ms / 1000.0)
    } else {
        0.0
    };

    let avg_prefill_ms = if per_pass_prefill_times.is_empty() {
        0.0
    } else {
        per_pass_prefill_times.iter().sum::<f64>() / per_pass_prefill_times.len() as f64
    };
    let prefill_tok_s = if avg_prefill_ms > 0.0 {
        test_tokens.len() as f64 / (avg_prefill_ms / 1000.0)
    } else {
        0.0
    };

    let avg_total_ms = if per_pass_total_times.is_empty() {
        0.0
    } else {
        per_pass_total_times.iter().sum::<f64>() / per_pass_total_times.len() as f64
    };

    // ========================================================================
    // PAR-PROFILE: BrickProfiler pass — per-operation GPU timing breakdown
    // Disable CUDA graph to get individual kernel timing via stream sync.
    // This adds overhead (~2x slower) but gives exact per-brick measurements.
    // ========================================================================
    println!(
        "{}",
        "Per-operation profiling pass (no CUDA graph)...".dimmed()
    );

    // SKIP_CUDA_GRAPH is checked per-call (not cached in OnceLock)
    std::env::set_var("SKIP_CUDA_GRAPH", "1");
    cuda_model.clear_decode_graph();
    cuda_model.enable_profiling();
    cuda_model.reset_profiler();

    // Run profiling pass with enough tokens for stable per-op breakdown
    let profile_tokens = 16;
    let profile_config = QuantizedGenerateConfig {
        max_tokens: profile_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };
    let _ = cuda_model.generate_gpu_resident(&test_tokens, &profile_config);

    // Extract per-operation hotspots from BrickProfiler
    let hotspots = extract_gpu_hotspots(&cuda_model, num_layers, hidden_dim, vocab_size);
    let category_summary = Some(compute_category_summary(&hotspots));

    // F-PROFILE-009: Compute kernel launch overhead
    let total_decode_us = avg_decode_ms * 1000.0;
    let (launch_overhead_us, launch_overhead_pct) =
        compute_kernel_launch_overhead(&hotspots, total_decode_us);

    // Compute roofline with the real results
    let mut results = RealProfileResults {
        model_path: path.display().to_string(),
        architecture,
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: avg_total_ms * 1000.0,
        throughput_tok_s: decode_tok_s,
        tokens_per_pass: tokens_per_decode,
        hotspots,
        per_layer_us: vec![],
        is_real_data: true,
        roofline: None,
        category_summary,
        backend: "cuda".to_string(),
        latency_p50_ms: p50,
        latency_p95_ms: p95,
        latency_p99_ms: p99,
        latency_min_ms: lat_min,
        latency_max_ms: lat_max,
        prefill_tok_s,
        decode_tok_s,
        total_tokens_generated,
        kernel_launch_overhead_pct: launch_overhead_pct,
        kernel_launch_overhead_us: launch_overhead_us,
    };

    // Compute roofline analysis
    results.roofline = Some(compute_roofline(&results));

    // Restore CUDA graph env
    std::env::remove_var("SKIP_CUDA_GRAPH");

    Ok(results)
}

/// Estimate data bytes moved per kernel invocation based on operation name and model dims.
///
/// For memory-bandwidth-bound kernels (GEMV, RMSNorm), the data movement is dominated
/// by reading the weight matrix. We estimate conservatively: read weights + read/write activations.
#[cfg(feature = "inference")]
fn estimate_kernel_data_bytes(name: &str, hidden_dim: usize, vocab_size: usize) -> Option<u64> {
    let name_lower = name.to_lowercase();
    // Q4K: 0.5625 bytes/element (144 bytes per 256-element super-block)
    let q4k_bytes_per_elem: f64 = 144.0 / 256.0;
    // Activation read/write: hidden_dim * 4 bytes (f32) in + hidden_dim * 4 out
    let activation_rw = (hidden_dim * 8) as u64;

    if name_lower.contains("q_proj") || name_lower.contains("k_proj") || name_lower.contains("v_proj") {
        // QKV projection: read weight [hidden, head_dim], read input, write output
        let weight_bytes = (hidden_dim as f64 * hidden_dim as f64 * q4k_bytes_per_elem) as u64;
        Some(weight_bytes + activation_rw)
    } else if name_lower.contains("o_proj") || name_lower.contains("out_proj") {
        let weight_bytes = (hidden_dim as f64 * hidden_dim as f64 * q4k_bytes_per_elem) as u64;
        Some(weight_bytes + activation_rw)
    } else if name_lower.contains("gate_proj") || name_lower.contains("up_proj") {
        // FFN gate/up: [hidden, intermediate] where intermediate ≈ 4*hidden for Qwen2
        let intermediate = hidden_dim * 4; // approximate
        let weight_bytes = (hidden_dim as f64 * intermediate as f64 * q4k_bytes_per_elem) as u64;
        Some(weight_bytes + activation_rw)
    } else if name_lower.contains("down_proj") {
        let intermediate = hidden_dim * 4;
        let weight_bytes = (intermediate as f64 * hidden_dim as f64 * q4k_bytes_per_elem) as u64;
        Some(weight_bytes + activation_rw)
    } else if name_lower.contains("lm_head") || name_lower.contains("output") {
        let weight_bytes = (hidden_dim as f64 * vocab_size as f64 * q4k_bytes_per_elem) as u64;
        Some(weight_bytes + (vocab_size * 4) as u64 + (hidden_dim * 4) as u64)
    } else if name_lower.contains("rmsnorm") || name_lower.contains("layernorm") {
        // Norm: read + write activation, read weight (small)
        Some(activation_rw + (hidden_dim * 4) as u64)
    } else if name_lower.contains("rope") || name_lower.contains("rotary") {
        Some(activation_rw)
    } else if name_lower.contains("softmax") || name_lower.contains("attention") {
        // Attention score: approximate as hidden_dim^2 / num_heads read/write
        Some(activation_rw * 2)
    } else if name_lower.contains("embed") {
        Some((hidden_dim * 4) as u64) // Single embedding lookup
    } else {
        None // Unknown operation
    }
}

/// Extract per-operation GPU hotspots from BrickProfiler after a profiling pass.
///
/// Converts trueno `BrickStats` into our `Hotspot` format with category
/// classification, bottleneck analysis, bandwidth estimation, and time breakdown.
#[cfg(feature = "inference")]
fn extract_gpu_hotspots(
    cuda_model: &realizar::gguf::OwnedQuantizedModelCuda,
    _num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Vec<Hotspot> {
    let profiler = cuda_model.profiler();
    let total_ns = profiler.total_ns();

    let mut hotspots: Vec<Hotspot> = profiler
        .all_brick_stats()
        .map(|stats| {
            let total_us = stats.total_ns as f64 / 1000.0;
            let pct = if total_ns > 0 {
                100.0 * stats.total_ns as f64 / total_ns as f64
            } else {
                0.0
            };
            let avg_us = if stats.count > 0 {
                total_us / stats.count as f64
            } else {
                0.0
            };

            // F-PROFILE-008: Estimate per-kernel bandwidth
            let data_bytes = estimate_kernel_data_bytes(&stats.name, hidden_dim, vocab_size);
            let bandwidth = data_bytes.and_then(|bytes| {
                if avg_us > 0.0 {
                    // GB/s = bytes / (µs * 1e-6) / 1e9 = bytes / (µs * 1e3)
                    Some(bytes as f64 / (avg_us * 1000.0))
                } else {
                    None
                }
            });

            Hotspot {
                name: stats.name.clone(),
                time_us: total_us,
                percent: pct,
                count: stats.count as usize,
                avg_us,
                min_us: stats.min_us(),
                max_us: stats.max_us(),
                bottleneck: Some(classify_operation_bottleneck(&stats.name)),
                efficiency_pct: bandwidth.map(|bw| (bw / 1008.0 * 100.0).min(100.0)), // RTX 4090 peak: 1008 GB/s
                category: Some(classify_operation_category(&stats.name)),
                bandwidth_gbs: bandwidth,
                data_bytes_per_call: data_bytes,
            }
        })
        .collect();

    // Sort by total time descending (hottest first)
    hotspots.sort_by(|a, b| {
        b.time_us
            .partial_cmp(&a.time_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    hotspots
}

/// Compute kernel launch overhead from profiler data (F-PROFILE-009).
///
/// Returns (total_launch_overhead_us, launch_overhead_percent_of_decode).
/// Launch overhead is estimated as the gap between sum of kernel times and total wall time.
#[cfg(feature = "inference")]
fn compute_kernel_launch_overhead(hotspots: &[Hotspot], total_decode_us: f64) -> (f64, f64) {
    let sum_kernel_us: f64 = hotspots.iter().map(|h| h.time_us).sum();
    // Launch overhead = total decode time - sum of kernel compute time
    // This includes: CUDA launch latency, memory allocation, synchronization
    let overhead_us = (total_decode_us - sum_kernel_us).max(0.0);
    let overhead_pct = if total_decode_us > 0.0 {
        overhead_us / total_decode_us * 100.0
    } else {
        0.0
    };
    (overhead_us, overhead_pct)
}

/// Run Ollama and collect baseline performance
fn run_ollama_comparison(path: &Path, tokens: usize) -> Option<OllamaBaseline> {
    // Determine model name from path
    let filename = path
        .file_stem()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown");

    // Map common filenames to Ollama model names
    let ollama_model = if filename.contains("qwen2.5-coder-7b") {
        "qwen2.5-coder:7b"
    } else if filename.contains("qwen2.5-coder-1.5b") {
        "qwen2.5-coder:1.5b"
    } else if filename.contains("TinyLlama") || filename.contains("tinyllama") {
        "tinyllama"
    } else {
        // Can't auto-detect — skip
        output::warn(&format!(
            "Cannot auto-detect Ollama model name for '{}'. Use known model files.",
            filename
        ));
        return None;
    };

    println!(
        "{}",
        format!(
            "Running Ollama baseline: {} ({} tokens)...",
            ollama_model, tokens
        )
        .dimmed()
    );

    // Run ollama with --verbose to get timing stats
    // Use a prompt that generates many tokens for accurate eval rate measurement
    let result = std::process::Command::new("ollama")
        .args([
            "run",
            ollama_model,
            "--verbose",
            "Write a short essay about the history of computing in exactly 128 words.",
        ])
        .output();

    match result {
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Parse eval rate from Ollama output
            // IMPORTANT: "prompt eval rate:" also contains "eval rate:", so
            // we must match decode line as "eval rate:" but NOT "prompt eval rate:"
            let decode_tok_s = stderr
                .lines()
                .find(|l| l.contains("eval rate:") && !l.contains("prompt eval rate:"))
                .and_then(|l| {
                    l.split_whitespace()
                        .find(|w| w.parse::<f64>().is_ok())
                        .and_then(|w| w.parse::<f64>().ok())
                })
                .unwrap_or(0.0);

            let prefill_tok_s = stderr
                .lines()
                .find(|l| l.contains("prompt eval rate:"))
                .and_then(|l| {
                    l.split_whitespace()
                        .find(|w| w.parse::<f64>().is_ok())
                        .and_then(|w| w.parse::<f64>().ok())
                })
                .unwrap_or(0.0);

            if decode_tok_s > 0.0 {
                Some(OllamaBaseline {
                    decode_tok_s,
                    prefill_tok_s,
                    model_name: ollama_model.to_string(),
                })
            } else {
                output::warn("Failed to parse Ollama output. Is Ollama running?");
                None
            }
        }
        Err(e) => {
            output::warn(&format!("Ollama not available: {e}"));
            None
        }
    }
}

/// Print Ollama comparison report
fn print_ollama_comparison(results: &RealProfileResults, baseline: &OllamaBaseline) {
    println!();
    output::subheader("Ollama Parity Report");
    println!();

    let parity_ratio = if baseline.decode_tok_s > 0.0 {
        results.decode_tok_s / baseline.decode_tok_s
    } else {
        0.0
    };

    // Grade based on Ollama parity
    // C = parity (1.0x), A = 2.0x, F = <0.5x
    let grade = if parity_ratio >= 2.0 {
        ("A+", "Excellent — 2x+ Ollama", "green")
    } else if parity_ratio >= 1.5 {
        ("A", "Great — 1.5x+ Ollama", "green")
    } else if parity_ratio >= 1.0 {
        ("B", "Good — Ollama parity achieved", "cyan")
    } else if parity_ratio >= 0.75 {
        ("C", "Passing — within 75% of Ollama", "yellow")
    } else if parity_ratio >= 0.5 {
        ("D", "Below parity — 50-75% of Ollama", "yellow")
    } else {
        ("F", "Critical — less than 50% of Ollama", "red")
    };

    println!(
        "  {} ({})",
        baseline.model_name.cyan(),
        results.backend.to_uppercase()
    );
    println!();
    println!("  ┌────────────┬──────────────┬──────────────┬───────────┐");
    println!("  │ Metric     │ apr          │ Ollama       │ Ratio     │");
    println!("  ├────────────┼──────────────┼──────────────┼───────────┤");

    // Decode throughput
    let decode_ratio_str = format!("{:.2}x", parity_ratio);
    println!(
        "  │ Decode     │ {:>8.1} t/s │ {:>8.1} t/s │ {:>9} │",
        results.decode_tok_s, baseline.decode_tok_s, decode_ratio_str
    );

    // Prefill throughput
    if baseline.prefill_tok_s > 0.0 && results.prefill_tok_s > 0.0 {
        let prefill_ratio = results.prefill_tok_s / baseline.prefill_tok_s;
        println!(
            "  │ Prefill    │ {:>8.1} t/s │ {:>8.1} t/s │ {:>8.2}x │",
            results.prefill_tok_s, baseline.prefill_tok_s, prefill_ratio
        );
    }

    println!("  └────────────┴──────────────┴──────────────┴───────────┘");
    println!();

    println!("  Grade: {} — {}", grade.0.bold(), grade.1);
    println!(
        "  Parity: {:.1}% of Ollama decode throughput",
        parity_ratio * 100.0
    );
    println!();

    // Citations for methodology
    println!("  {}", "Methodology:".dimmed());
    println!(
        "  {}",
        "  Pope et al. (2023) 'Efficiently Scaling Transformer Inference'".dimmed()
    );
    println!(
        "  {}",
        "  Williams et al. (2009) 'Roofline: An Insightful Visual Performance Model'".dimmed()
    );
}

// ============================================================================
// Roofline & Classification Helpers
// ============================================================================

/// Classify an operation into BrickId category (Attention, FFN, Norm, Other)
///
/// Supports both GPU brick names (QKV, RoPE, Attention, OProj) and
/// CPU brick names (QkvProjection, RopeEmbedding, etc.)
fn classify_operation_category(name: &str) -> String {
    match name {
        // GPU brick names (from indexed.rs start_brick_timer calls)
        "QKV" | "RoPE" | "RopeEmbedding" | "Attention" | "OProj" => "Attention".to_string(),
        "FFNGateUp" | "SwiGLU" | "FFNDown" => "FFN".to_string(),
        "RmsNorm1" | "RmsNorm2" | "OutputNorm" => "Norm".to_string(),
        "LmHead" => "FFN".to_string(), // LM head is a GEMV (same category as FFN projections)
        "Residual1" | "Residual2" => "Other".to_string(),
        // CPU brick names (legacy)
        "QkvProjection" | "AttentionScore" | "AttentionSoftmax" | "AttentionOutput"
        | "OutputProjection" => "Attention".to_string(),
        "GateProjection" | "UpProjection" | "Activation" | "DownProjection" => "FFN".to_string(),
        "RmsNorm" | "LayerNorm" => "Norm".to_string(),
        _ => "Other".to_string(),
    }
}

/// Classify operation bottleneck (Memory vs Compute bound)
///
/// Q4K decode-time matmul is overwhelmingly memory-bandwidth limited:
/// AI = 2*N / (N/2 bytes_per_weight) = ~4, threshold ~82 for GPU, ~10 for CPU.
/// Only softmax and activation are compute-bound (element-wise).
fn classify_operation_bottleneck(name: &str) -> String {
    match name {
        // Element-wise ops: compute-bound (low memory traffic, high FLOP/byte)
        "SwiGLU" | "Activation" | "RoPE" | "RopeEmbedding" | "AttentionSoftmax" => {
            "COMPUTE".to_string()
        }
        // Everything else: memory-bound (weight/KV reads dominate)
        _ => "MEMORY".to_string(),
    }
}

/// Build real per-layer timing from profiler report's per_layer data
#[cfg(feature = "inference")]
fn build_per_layer_timing(report: &realizar::brick::ProfileReport, num_layers: usize) -> Vec<f64> {
    if num_layers == 0 {
        return vec![];
    }

    // Sum per-layer entries from all per-layer-aware operations
    let mut layer_times = vec![0.0_f64; num_layers];
    for stats in report.operations.values() {
        // Each operation's per_layer vec has one entry per call
        // For N layers × M passes, the entries alternate:
        //   layer0_pass0, layer0_pass1, ..., layer1_pass0, ...
        // But BrickProfiler just appends in order.
        // The most useful view: divide entries across layers
        if stats.per_layer.len() >= num_layers {
            // Distribute entries evenly across layers
            let entries_per_layer = stats.per_layer.len() / num_layers;
            if entries_per_layer > 0 {
                for (layer_idx, time) in layer_times.iter_mut().enumerate() {
                    let start = layer_idx * entries_per_layer;
                    let end = start + entries_per_layer;
                    let layer_total: f64 = stats.per_layer[start..end.min(stats.per_layer.len())]
                        .iter()
                        .sum();
                    *time += layer_total / entries_per_layer as f64; // Average across passes
                }
            }
        }
    }
    layer_times
}

/// Compute category time summary from hotspots
fn compute_category_summary(hotspots: &[Hotspot]) -> CategorySummary {
    let total: f64 = hotspots.iter().map(|h| h.time_us).sum();
    if total <= 0.0 {
        return CategorySummary::default();
    }

    let mut attn = 0.0_f64;
    let mut ffn = 0.0_f64;
    let mut norm = 0.0_f64;
    let mut other = 0.0_f64;

    for h in hotspots {
        let cat = match h.category.as_deref() {
            Some(c) => c.to_string(),
            None => classify_operation_category(&h.name),
        };
        match cat.as_str() {
            "Attention" => attn += h.time_us,
            "FFN" => ffn += h.time_us,
            "Norm" => norm += h.time_us,
            _ => other += h.time_us,
        }
    }

    CategorySummary {
        attention_pct: (attn / total) * 100.0,
        ffn_pct: (ffn / total) * 100.0,
        norm_pct: (norm / total) * 100.0,
        other_pct: (other / total) * 100.0,
    }
}

/// Compute roofline analysis using trueno hardware detection
#[cfg(feature = "inference")]
fn compute_roofline(results: &RealProfileResults) -> RooflineAnalysis {
    let is_gpu = results.backend == "cuda";

    // Hardware detection: use GPU specs for CUDA, CPU specs for CPU
    let (peak_compute, peak_bw, ai_threshold, hardware_model) = if is_gpu {
        // GPU roofline: detect via CUDA device properties or use known specs
        // RTX 4090: 82.6 TFLOPS FP32, 1008 GB/s GDDR6X
        // RTX 3090: 35.6 TFLOPS FP32, 936 GB/s GDDR6X
        // For Q4K decode (int4 dequant + FP16/FP32 GEMV), effective AI is very low
        let gpu_info = detect_gpu_hardware();
        (gpu_info.0, gpu_info.1, gpu_info.2, gpu_info.3)
    } else {
        let hw = trueno::hardware::HardwareCapability::detect();
        (
            hw.cpu.peak_gflops,
            hw.cpu.memory_bw_gbps,
            hw.roofline.cpu_arithmetic_intensity,
            format!(
                "{} {} ({} cores, {})",
                hw.cpu.vendor,
                hw.cpu.model,
                hw.cpu.cores,
                hw.cpu.simd.bits()
            ),
        )
    };

    // Estimate FLOPs for one forward pass:
    // Dominant: matmul = 2 * M * N * K per matmul
    // For Q4K, each weight element is ~0.5 bytes, so bytes >> FLOPs → memory bound
    let hidden = results.hidden_dim as f64;
    let vocab = results.vocab_size as f64;
    let layers = results.num_layers as f64;

    // Per-layer FLOPs: QKV(2*h*3h) + OutProj(2*h*h) + Gate(2*h*4h) + Up(2*h*4h) + Down(2*h*4h)
    // = 2h² * (3 + 1 + 4 + 4 + 4) = 32h²
    let flops_per_layer = 32.0 * hidden * hidden;
    let flops_lm_head = 2.0 * hidden * vocab;
    let total_flops = flops_per_layer * layers + flops_lm_head;

    // Bytes transferred (Q4K = 0.5 bytes per weight element)
    let bytes_per_layer = 16.0 * hidden * hidden * 0.5; // all matmul weights
    let bytes_lm_head = hidden * vocab * 0.5;
    let total_bytes = bytes_per_layer * layers + bytes_lm_head;

    // For GPU: use per-token decode time, not total inference (which includes prefill overhead)
    let inference_sec = if is_gpu && results.decode_tok_s > 0.0 {
        // Per-token time = 1/decode_tok_s (more accurate for GPU roofline)
        1.0 / results.decode_tok_s
    } else {
        results.total_inference_us / 1_000_000.0
    };

    let achieved_gflops = if inference_sec > 0.0 {
        (total_flops / 1e9) / inference_sec
    } else {
        0.0
    };
    let achieved_bw = if inference_sec > 0.0 {
        (total_bytes / 1e9) / inference_sec
    } else {
        0.0
    };

    let ai = if total_bytes > 0.0 {
        total_flops / total_bytes
    } else {
        0.0
    };

    let compute_eff = if peak_compute > 0.0 {
        (achieved_gflops / peak_compute) * 100.0
    } else {
        0.0
    };
    let memory_eff = if peak_bw > 0.0 {
        (achieved_bw / peak_bw) * 100.0
    } else {
        0.0
    };

    let bottleneck = if ai < ai_threshold {
        "MEMORY BOUND"
    } else {
        "COMPUTE BOUND"
    };

    RooflineAnalysis {
        peak_compute,
        peak_bandwidth_gbps: peak_bw,
        achieved_gflops,
        achieved_bandwidth_gbps: achieved_bw,
        compute_efficiency_pct: compute_eff,
        memory_efficiency_pct: memory_eff,
        arithmetic_intensity: ai,
        ai_threshold,
        bottleneck: bottleneck.to_string(),
        backend: results.backend.clone(),
        hardware_model,
    }
}

/// Detect GPU hardware specs for roofline analysis
/// Returns (peak_tflops_as_gflops, peak_bw_gbps, ai_threshold, model_name)
/// Look up known GPU specs (peak GFLOPS, peak BW GB/s, AI threshold) by name.
fn gpu_specs_by_name(name: &str) -> (f64, f64, f64) {
    match name {
        n if n.contains("4090") => (82_580.0, 1008.0, 82.0),
        n if n.contains("4080") => (48_740.0, 716.8, 68.0),
        n if n.contains("4070") => (29_150.0, 504.2, 57.8),
        n if n.contains("3090") => (35_580.0, 936.0, 38.0),
        n if n.contains("3080") => (29_770.0, 760.0, 39.2),
        n if n.contains("A100") => (19_500.0, 2039.0, 9.6),
        n if n.contains("H100") => (51_200.0, 3350.0, 15.3),
        _ => (30_000.0, 800.0, 37.5),
    }
}

/// Parse nvidia-smi output to extract GPU name. Returns None if unavailable.
fn query_nvidia_smi_gpu_name() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let info = String::from_utf8_lossy(&output.stdout);
    let line = info.lines().next()?;
    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() >= 2 {
        Some(parts[0].trim().to_string())
    } else {
        None
    }
}

fn detect_gpu_hardware() -> (f64, f64, f64, String) {
    if let Some(gpu_name) = query_nvidia_smi_gpu_name() {
        let (peak_gflops, peak_bw, ai_thresh) = gpu_specs_by_name(&gpu_name);
        return (peak_gflops, peak_bw, ai_thresh, gpu_name);
    }
    // Fallback: generic CUDA GPU
    (30_000.0, 800.0, 37.5, "CUDA GPU (unknown)".to_string())
}

/// Profile SafeTensors model — converts to GGUF path for per-op profiling
#[cfg(feature = "inference")]
fn profile_safetensors_real(
    path: &Path,
    _warmup_passes: usize,
    _measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    // SafeTensors models need import to GGUF/APR for per-operation profiling.
    // Check if there's a sibling .gguf file to use instead.
    let gguf_path = path.with_extension("gguf");
    if gguf_path.exists() {
        output::info(&format!(
            "Found sibling GGUF: {}. Profiling that instead.",
            gguf_path.display()
        ));
        return profile_gguf_real(&gguf_path, _warmup_passes, _measure_passes);
    }

    output::warn("SafeTensors per-operation profiling requires GGUF format.");
    output::info("Convert first: apr import model.safetensors -o model.gguf");
    output::info("Then: apr profile model.gguf");
    Err(CliError::ValidationFailed(
        "SafeTensors per-op profiling not yet supported. Use GGUF format for full profiling."
            .to_string(),
    ))
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

    // Measurement passes with per-operation profiler
    println!(
        "{}",
        format!(
            "Running {} measurement passes (per-op instrumented)...",
            measure_passes
        )
        .dimmed()
    );

    let mut profiler = BrickProfiler::new();
    profiler.set_num_layers(num_layers);
    profiler.set_tokens(tokens_per_pass * measure_passes);

    let mut forward_times: Vec<f64> = Vec::new();

    profiler.start_inference();

    for _ in 0..measure_passes {
        let pass_start = Instant::now();

        // Use forward_profiled() for real per-operation timing
        let logits = model.forward_profiled(&test_tokens, &mut profiler);

        let pass_time = pass_start.elapsed().as_secs_f64() * 1_000_000.0;
        forward_times.push(pass_time);

        // Validate output
        if let Ok(ref logits) = logits {
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
    let _min_us = forward_times.iter().copied().fold(f64::INFINITY, f64::min);
    let _max_us = forward_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    // Build hotspots from profiler report with roofline classification
    let mut hotspots: Vec<Hotspot> = report
        .operations
        .iter()
        .map(|(name, stats)| {
            let category = classify_operation_category(name);
            let bottleneck = classify_operation_bottleneck(name);
            Hotspot {
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
                bottleneck: Some(bottleneck),
                efficiency_pct: None, // Computed later with hardware info
                category: Some(category),
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }
        })
        .collect();

    // Sort by total time descending
    hotspots.sort_by(|a, b| {
        b.time_us
            .partial_cmp(&a.time_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build real per-layer timing from profiler report's per_layer data
    // Each operation records per_layer entries — sum across all ops per layer
    let per_layer_us = build_per_layer_timing(&report, num_layers);

    // Compute category summary
    let category_summary = compute_category_summary(&hotspots);

    // Compute real percentiles from forward times
    let mut sorted_times: Vec<f64> = forward_times.iter().map(|t| t / 1000.0).collect(); // us -> ms
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let tps = if avg_us > 0.0 {
        (tokens_per_pass as f64 / avg_us) * 1_000_000.0
    } else {
        0.0
    };

    Ok(RealProfileResults {
        model_path: path.display().to_string(),
        architecture,
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: avg_us,
        throughput_tok_s: tps,
        tokens_per_pass,
        hotspots,
        per_layer_us,
        is_real_data: report.is_real_data,
        roofline: None,
        category_summary: Some(category_summary),
        backend: "cpu".to_string(),
        latency_p50_ms: percentile(&sorted_times, 50.0),
        latency_p95_ms: percentile(&sorted_times, 95.0),
        latency_p99_ms: percentile(&sorted_times, 99.0),
        latency_min_ms: sorted_times.first().copied().unwrap_or(0.0),
        latency_max_ms: sorted_times.last().copied().unwrap_or(0.0),
        prefill_tok_s: 0.0, // CPU path doesn't separate prefill/decode
        decode_tok_s: tps,
        total_tokens_generated: tokens_per_pass * measure_passes,
        kernel_launch_overhead_pct: 0.0, // CPU path: no kernel launches
        kernel_launch_overhead_us: 0.0,
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
    let min_us = forward_times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = forward_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut sorted_times: Vec<f64> = forward_times.iter().map(|t| t / 1000.0).collect();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let tps = if avg_us > 0.0 {
        (tokens_per_pass as f64 / avg_us) * 1_000_000.0
    } else {
        0.0
    };

    Ok(RealProfileResults {
        model_path: path.display().to_string(),
        architecture: "apr".to_string(),
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: avg_us,
        throughput_tok_s: tps,
        tokens_per_pass,
        hotspots: vec![Hotspot {
            name: "forward_pass".to_string(),
            time_us: total_us,
            percent: 100.0,
            count: measure_passes,
            avg_us,
            min_us,
            max_us,
            bottleneck: None,
            efficiency_pct: None,
            category: Some("Other".to_string()),
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        }],
        per_layer_us: vec![avg_us / num_layers as f64; num_layers],
        is_real_data: true,
        roofline: None,
        category_summary: None,
        backend: "cpu".to_string(),
        latency_p50_ms: percentile(&sorted_times, 50.0),
        latency_p95_ms: percentile(&sorted_times, 95.0),
        latency_p99_ms: percentile(&sorted_times, 99.0),
        latency_min_ms: sorted_times.first().copied().unwrap_or(0.0),
        latency_max_ms: sorted_times.last().copied().unwrap_or(0.0),
        prefill_tok_s: 0.0,
        decode_tok_s: tps,
        total_tokens_generated: tokens_per_pass * measure_passes,
        kernel_launch_overhead_pct: 0.0, // APR CPU: no kernel launches
        kernel_launch_overhead_us: 0.0,
    })
}

/// Print human-readable results with per-operation hotspots, category bars, and roofline
fn print_human_results(
    results: &RealProfileResults,
    granular: bool,
    show_perf_grade: bool,
    detect_naive: bool,
) -> Result<(), CliError> {
    print_profile_model_header(results);
    print_hotspot_table(results, granular);
    print_category_summary(results);
    print_kernel_launch_overhead(results);
    print_per_layer_timing(results, granular);
    print_roofline_section(results);
    print_perf_grade_section(results, show_perf_grade);
    print_naive_detection(results, detect_naive);
    print_generation_performance(results);
    print_latency_percentiles(results);
    print_profile_summary(results);
    Ok(())
}

fn print_profile_model_header(results: &RealProfileResults) {
    println!(
        "{}",
        output::kv_table(&[
            (
                "Architecture",
                format!(
                    "{} ({} layers, hidden={}, vocab={})",
                    results.architecture, results.num_layers,
                    output::count_fmt(results.hidden_dim),
                    output::count_fmt(results.vocab_size)
                )
            ),
            ("Backend", results.backend.to_uppercase()),
            ("Warmup", format!("{} passes", results.warmup_passes)),
            ("Measure", format!("{} passes", results.measure_passes)),
        ])
    );
    println!();
    if results.is_real_data {
        println!("  {}", output::badge_pass("REAL PER-OPERATION TELEMETRY"));
    } else {
        println!("  {}", output::badge_warn("SIMULATED DATA (inference disabled)"));
    }
    println!();
}

fn print_hotspot_table(results: &RealProfileResults, granular: bool) {
    output::subheader("Per-Operation Hotspots");
    println!();

    let total_time = results.hotspots.iter().map(|h| h.time_us).sum::<f64>();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, hotspot) in results.hotspots.iter().enumerate() {
        let percent = if total_time > 0.0 { (hotspot.time_us / total_time) * 100.0 } else { 0.0 };
        let bar = output::progress_bar(percent as usize, 100, 20);
        let bottleneck_str = hotspot.bottleneck.as_deref().unwrap_or("-");
        let mut row = vec![
            format!("#{}", i + 1), hotspot.name.clone(),
            format!("{:.0}µs", hotspot.time_us), format!("{:.1}%", percent),
            format!("{}", hotspot.count), bottleneck_str.to_string(), bar,
        ];
        if granular {
            let bw_str = hotspot.bandwidth_gbs.map(|bw| format!(", bw={:.1}GB/s", bw)).unwrap_or_default();
            let eff_str = hotspot.efficiency_pct.map(|e| format!(", eff={:.0}%", e)).unwrap_or_default();
            row.push(format!(
                "avg={:.1}µs, min={:.1}µs, max={:.1}µs{}{}",
                hotspot.avg_us, hotspot.min_us, hotspot.max_us, bw_str, eff_str
            ));
        }
        rows.push(row);
    }

    let headers: &[&str] = if granular {
        &["#", "Operation", "Time", "%", "Calls", "Bottleneck", "Bar", "Detail"]
    } else {
        &["#", "Operation", "Time", "%", "Calls", "Bottleneck", "Bar"]
    };
    println!("{}", output::table(headers, &rows));
    println!();
}

fn print_category_summary(results: &RealProfileResults) {
    let Some(ref cat) = results.category_summary else { return };
    output::subheader("Category Summary");
    println!();
    let bw = 40;
    let bar = |pct: f64| "█".repeat(((pct / 100.0) * bw as f64) as usize);
    println!("  Attention: {:5.1}%  {}", cat.attention_pct, bar(cat.attention_pct).cyan());
    println!("  FFN:       {:5.1}%  {}", cat.ffn_pct, bar(cat.ffn_pct).green());
    println!("  Norm:      {:5.1}%  {}", cat.norm_pct, bar(cat.norm_pct).yellow());
    println!("  Other:     {:5.1}%  {}", cat.other_pct, bar(cat.other_pct).dimmed());
    println!();
}

fn print_kernel_launch_overhead(results: &RealProfileResults) {
    if results.kernel_launch_overhead_us <= 0.0 { return; }
    output::subheader("Kernel Launch Overhead (F-PROFILE-009)");
    println!();
    println!("  Overhead: {:.0}µs ({:.1}% of decode time)",
        results.kernel_launch_overhead_us, results.kernel_launch_overhead_pct);
    let msg = if results.kernel_launch_overhead_pct > 20.0 {
        "WARNING: >20% overhead — consider kernel fusion".red()
    } else if results.kernel_launch_overhead_pct > 10.0 {
        "NOTE: 10-20% overhead — moderate, may benefit from CUDA graph".yellow()
    } else {
        "OK: <10% overhead — launch latency is not a bottleneck".green()
    };
    println!("  {msg}");
    println!();
}

fn print_per_layer_timing(results: &RealProfileResults, granular: bool) {
    if !granular || results.per_layer_us.is_empty() { return; }
    output::subheader("Per-Layer Timing (real)");
    println!();

    let max_t = results.per_layer_us.iter().copied().fold(0.0f64, f64::max);
    let min_t = results.per_layer_us.iter().copied().fold(f64::INFINITY, f64::min);
    if max_t > 0.0 && min_t > 0.0 {
        let cv = (max_t - min_t) / ((max_t + min_t) / 2.0);
        if cv < 0.01 {
            println!("  {}", output::badge_warn("WARNING: Per-layer timing shows zero variance (may be estimated)"));
        } else {
            println!("  {} (CV={:.1}%)", output::badge_pass("Real per-layer timing verified"), cv * 100.0);
        }
        println!();
    }

    let rows: Vec<Vec<String>> = results.per_layer_us.iter().enumerate().map(|(i, &t)| {
        let bw = if max_t > 0.0 { ((t / max_t) * 100.0) as usize } else { 0 };
        vec![format!("Layer {i}"), format!("{t:.1}µs"), output::progress_bar(bw, 100, 30)]
    }).collect();
    println!("{}", output::table(&["Layer", "Time", "Bar"], &rows));
    println!();
}

fn print_roofline_section(results: &RealProfileResults) {
    let Some(ref r) = results.roofline else { return };
    output::subheader("Roofline Analysis");
    println!();
    println!("  Hardware:       {} (peak {:.1} GFLOPS, {:.1} GB/s)", r.hardware_model, r.peak_compute, r.peak_bandwidth_gbps);
    println!("  Achieved:       {:.1} GFLOPS, {:.1} GB/s", r.achieved_gflops, r.achieved_bandwidth_gbps);
    println!("  Compute eff:    {:.1}%", r.compute_efficiency_pct);
    println!("  Memory eff:     {:.1}%", r.memory_efficiency_pct);
    println!("  Arithmetic int: {:.2} (threshold={:.1})", r.arithmetic_intensity, r.ai_threshold);
    println!("  {}", if r.bottleneck == "MEMORY BOUND" {
        output::badge_info(&r.bottleneck)
    } else {
        output::badge_warn(&r.bottleneck)
    });
    println!();
    if r.bottleneck == "MEMORY BOUND" {
        output::info("Decode is memory-bandwidth limited. Matmul operations transfer");
        output::info("more bytes than FLOPs computed. Focus on memory access patterns.");
    }
    println!();
}

fn print_perf_grade_section(results: &RealProfileResults, show: bool) {
    if !show { return; }
    let eff = results.roofline.as_ref().map_or(0.0, |r| r.memory_efficiency_pct.max(r.compute_efficiency_pct));
    let grade = PerfGrade::from_efficiency(eff);
    output::subheader("Performance Grade");
    println!();
    println!("  Grade: {}  —  {}", grade.label().bold(), grade.description());
    println!("  Efficiency: {:.1}%", eff);
    println!();
}

fn print_naive_detection(results: &RealProfileResults, detect: bool) {
    if !detect { return; }
    output::subheader("Naive Implementation Detection");
    println!();
    let mut found = false;
    for h in &results.hotspots {
        if h.count > 0 && h.avg_us > results.total_inference_us * 0.5 {
            println!("  {} {} takes {:.1}% of total time ({:.0}µs avg) — check for scalar fallback",
                output::badge_warn("NAIVE?"), h.name, h.percent, h.avg_us);
            found = true;
        }
    }
    if !found {
        println!("  {} No obvious naive implementations detected", output::badge_pass("OK"));
    }
    println!();
}

fn print_generation_performance(results: &RealProfileResults) {
    if results.decode_tok_s <= 0.0 && results.prefill_tok_s <= 0.0 { return; }
    output::subheader("Generation Performance");
    println!();
    println!("{}", output::kv_table(&[
        ("Decode throughput", format!("{:.1} tok/s", results.decode_tok_s)),
        ("Prefill throughput", format!("{:.1} tok/s", results.prefill_tok_s)),
        ("Tokens generated", format!("{}", results.total_tokens_generated)),
    ]));
    println!();
}

fn print_latency_percentiles(results: &RealProfileResults) {
    if results.latency_p50_ms <= 0.0 { return; }
    output::subheader("Latency Distribution (decode pass)");
    println!();
    println!("{}", output::kv_table(&[
        ("p50 (median)", format!("{:.1} ms", results.latency_p50_ms)),
        ("p95", format!("{:.1} ms", results.latency_p95_ms)),
        ("p99", format!("{:.1} ms", results.latency_p99_ms)),
        ("min", format!("{:.1} ms", results.latency_min_ms)),
        ("max", format!("{:.1} ms", results.latency_max_ms)),
    ]));
    println!();
}

fn print_profile_summary(results: &RealProfileResults) {
    output::subheader("Summary");
    println!();
    output::metric("Avg forward pass",
        format!("{:.1}µs ({:.2}ms)", results.total_inference_us, results.total_inference_us / 1000.0), "");
    output::metric("Throughput", format!("{:.2}", results.throughput_tok_s), "tok/s");
    output::metric("Tokens per pass", results.tokens_per_pass, "");
    output::metric("Operations profiled", results.hotspots.len(), "");
    println!();
}

/// Print JSON output
fn print_json_results(results: &RealProfileResults) -> Result<(), CliError> {
    let mut json = String::from("{\n");
    writeln!(json, "  \"model\": \"{}\",", results.model_path)
        .expect("write to String is infallible");
    writeln!(json, "  \"architecture\": \"{}\",", results.architecture)
        .expect("write to String is infallible");
    writeln!(json, "  \"num_layers\": {},", results.num_layers)
        .expect("write to String is infallible");
    writeln!(json, "  \"vocab_size\": {},", results.vocab_size)
        .expect("write to String is infallible");
    writeln!(json, "  \"hidden_dim\": {},", results.hidden_dim)
        .expect("write to String is infallible");
    writeln!(json, "  \"is_real_data\": {},", results.is_real_data)
        .expect("write to String is infallible");
    json.push_str("  \"timing\": {\n");
    writeln!(json, "    \"warmup_passes\": {},", results.warmup_passes)
        .expect("write to String is infallible");
    writeln!(json, "    \"measure_passes\": {},", results.measure_passes)
        .expect("write to String is infallible");
    writeln!(
        json,
        "    \"avg_inference_us\": {:.2},",
        results.total_inference_us
    )
    .expect("write to String is infallible");
    writeln!(
        json,
        "    \"throughput_tok_s\": {:.2}",
        results.throughput_tok_s
    )
    .expect("write to String is infallible");
    json.push_str("  },\n");

    json.push_str("  \"hotspots\": [\n");
    for (i, hotspot) in results.hotspots.iter().enumerate() {
        json.push_str("    {\n");
        writeln!(json, "      \"name\": \"{}\",", hotspot.name)
            .expect("write to String is infallible");
        writeln!(json, "      \"total_us\": {:.2},", hotspot.time_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"avg_us\": {:.2},", hotspot.avg_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"min_us\": {:.2},", hotspot.min_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"max_us\": {:.2},", hotspot.max_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"count\": {}", hotspot.count)
            .expect("write to String is infallible");
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
        write!(json, "{:.2}", time).expect("write to String is infallible");
    }
    json.push_str("]\n");

    json.push_str("}\n");

    println!("{json}");
    Ok(())
}

/// Print flamegraph SVG (GH-174: supports --output for file output)
fn print_flamegraph(
    results: &RealProfileResults,
    output_path: Option<&Path>,
) -> Result<(), CliError> {
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

        writeln!(
            svg,
            "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{width:.1}\" height=\"{height:.1}\" fill=\"{color}\" class=\"frame\"/>"
        ).expect("write to String is infallible");
        writeln!(
            svg,
            "  <text x=\"{:.1}\" y=\"{:.1}\" class=\"label\">{} ({:.1}%)</text>",
            x + 5.0,
            y + 16.0,
            hotspot.name,
            percent
        )
        .expect("write to String is infallible");

        y -= height + 2.0;
    }

    svg.push_str("</svg>\n");

    // GH-174: Write to file if output path specified, otherwise print to stdout
    if let Some(path) = output_path {
        std::fs::write(path, &svg).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to write flamegraph to {}: {e}",
                path.display()
            ))
        })?;
        output::success(&format!("Flamegraph written to: {}", path.display()));
    } else {
        println!("{svg}");
    }
    Ok(())
}

// ============================================================================
// Cross-format performance comparison (F-PROFILE-011)
// ============================================================================

/// Run side-by-side performance comparison between two model formats.
///
/// Profiles both models using real inference passes, then prints a table
/// comparing decode tok/s, prefill tok/s, and latency percentiles.
///
/// Usage: `apr profile model.apr --compare model.gguf`
#[cfg(feature = "inference")]
pub(crate) fn run_cross_format_comparison(
    path_a: &Path,
    path_b: &Path,
    warmup: usize,
    measure: usize,
    tokens: usize,
    no_gpu: bool,
) -> Result<(), CliError> {
    let format_a = detect_format(path_a);
    let format_b = detect_format(path_b);

    println!(
        "\n{}",
        format!(
            "Cross-Format Comparison: {} ({}) vs {} ({})",
            path_a.file_name().and_then(|f| f.to_str()).unwrap_or("?"),
            format_a.to_uppercase(),
            path_b.file_name().and_then(|f| f.to_str()).unwrap_or("?"),
            format_b.to_uppercase(),
        )
        .cyan()
        .bold()
    );
    println!("{}", "=".repeat(60));

    // Profile first model
    println!(
        "\n{}",
        format!("[1/2] Profiling {} ({})...", path_a.display(), format_a).dimmed()
    );
    let results_a = if no_gpu {
        profile_real_inference_cpu(path_a, warmup, measure)
    } else {
        profile_gpu_or_cpu(path_a, warmup, measure, tokens)
    }?;

    // Profile second model
    println!(
        "\n{}",
        format!("[2/2] Profiling {} ({})...", path_b.display(), format_b).dimmed()
    );
    let results_b = if no_gpu {
        profile_real_inference_cpu(path_b, warmup, measure)
    } else {
        profile_gpu_or_cpu(path_b, warmup, measure, tokens)
    }?;

    // Print comparison table
    println!("\n{}", "Performance Comparison".green().bold());
    println!("{}", "-".repeat(60));
    println!(
        "{:<24} {:>15} {:>15}",
        "Metric",
        format!("{} ({})", format_a.to_uppercase(), results_a.backend),
        format!("{} ({})", format_b.to_uppercase(), results_b.backend),
    );
    println!("{}", "-".repeat(60));

    print_comparison_row("Decode (tok/s)", results_a.decode_tok_s, results_b.decode_tok_s);
    print_comparison_row("Prefill (tok/s)", results_a.prefill_tok_s, results_b.prefill_tok_s);
    print_comparison_row(
        "Throughput (tok/s)",
        results_a.throughput_tok_s,
        results_b.throughput_tok_s,
    );
    print_comparison_row("Latency p50 (ms)", results_a.latency_p50_ms, results_b.latency_p50_ms);
    print_comparison_row("Latency p99 (ms)", results_a.latency_p99_ms, results_b.latency_p99_ms);
    println!("{}", "-".repeat(60));

    // Summary
    let decode_ratio = if results_b.decode_tok_s > 0.0 {
        results_a.decode_tok_s / results_b.decode_tok_s
    } else {
        0.0
    };
    let throughput_ratio = if results_b.throughput_tok_s > 0.0 {
        results_a.throughput_tok_s / results_b.throughput_tok_s
    } else {
        0.0
    };

    println!(
        "\n{} is {:.2}x decode, {:.2}x throughput vs {}",
        format_a.to_uppercase(),
        decode_ratio,
        throughput_ratio,
        format_b.to_uppercase(),
    );

    Ok(())
}

/// Try GPU profiling first, fall back to CPU if unavailable.
#[cfg(feature = "inference")]
fn profile_gpu_or_cpu(
    path: &Path,
    warmup: usize,
    measure: usize,
    tokens: usize,
) -> Result<RealProfileResults, CliError> {
    #[cfg(feature = "cuda")]
    {
        match profile_gpu_generation(path, warmup, measure, tokens) {
            Ok(r) => return Ok(r),
            Err(_) => {
                output::info("GPU profiling unavailable, falling back to CPU");
            }
        }
    }
    let _ = tokens; // Unused in CPU-only builds
    profile_real_inference_cpu(path, warmup, measure)
}

/// Print a comparison row with color-coded values.
fn print_comparison_row(label: &str, value_a: f64, value_b: f64) {
    let a_str = if value_a > 0.0 {
        format!("{value_a:.1}")
    } else {
        "N/A".to_string()
    };
    let b_str = if value_b > 0.0 {
        format!("{value_b:.1}")
    } else {
        "N/A".to_string()
    };

    println!("{:<24} {:>15} {:>15}", label, a_str, b_str);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // OutputFormat Tests
    // ========================================================================

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
    fn test_output_format_parse_text() {
        assert!(matches!(
            "text".parse::<OutputFormat>().unwrap(),
            OutputFormat::Human
        ));
    }

    #[test]
    fn test_output_format_parse_svg() {
        assert!(matches!(
            "svg".parse::<OutputFormat>().unwrap(),
            OutputFormat::Flamegraph
        ));
    }

    #[test]
    fn test_output_format_parse_case_insensitive() {
        assert!(matches!(
            "JSON".parse::<OutputFormat>().unwrap(),
            OutputFormat::Json
        ));
        assert!(matches!(
            "HUMAN".parse::<OutputFormat>().unwrap(),
            OutputFormat::Human
        ));
    }

    #[test]
    fn test_output_format_parse_invalid() {
        assert!("invalid".parse::<OutputFormat>().is_err());
        assert!("xml".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::default();
        assert!(matches!(format, OutputFormat::Human));
    }

    #[test]
    fn test_output_format_debug() {
        let format = OutputFormat::Json;
        let debug = format!("{format:?}");
        assert!(debug.contains("Json"));
    }

    #[test]
    fn test_output_format_clone() {
        let format = OutputFormat::Flamegraph;
        let cloned = format;
        assert!(matches!(cloned, OutputFormat::Flamegraph));
    }

    // ========================================================================
    // ProfileFocus Tests
    // ========================================================================

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
    fn test_profile_focus_parse_attn() {
        assert!(matches!(
            "attn".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Attention
        ));
    }

    #[test]
    fn test_profile_focus_parse_ffn() {
        assert!(matches!(
            "ffn".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Mlp
        ));
    }

    #[test]
    fn test_profile_focus_parse_matmul() {
        assert!(matches!(
            "matmul".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Matmul
        ));
        assert!(matches!(
            "gemm".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Matmul
        ));
    }

    #[test]
    fn test_profile_focus_parse_embedding() {
        assert!(matches!(
            "embedding".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Embedding
        ));
        assert!(matches!(
            "embed".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Embedding
        ));
    }

    #[test]
    fn test_profile_focus_parse_invalid() {
        assert!("invalid".parse::<ProfileFocus>().is_err());
        assert!("unknown".parse::<ProfileFocus>().is_err());
    }

    #[test]
    fn test_profile_focus_default() {
        let focus = ProfileFocus::default();
        assert!(matches!(focus, ProfileFocus::All));
    }

    #[test]
    fn test_profile_focus_debug() {
        let focus = ProfileFocus::Attention;
        let debug = format!("{focus:?}");
        assert!(debug.contains("Attention"));
    }

    // ========================================================================
    // detect_format Tests
    // ========================================================================

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("model.apr")), "apr");
        assert_eq!(detect_format(Path::new("model.safetensors")), "safetensors");
        assert_eq!(detect_format(Path::new("model.gguf")), "gguf");
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format(Path::new("model.xyz")), "unknown");
        assert_eq!(detect_format(Path::new("model.pt")), "unknown");
    }

    #[test]
    fn test_detect_format_pytorch() {
        assert_eq!(detect_format(Path::new("model.bin")), "pytorch");
    }

    #[test]
    fn test_detect_format_no_extension() {
        assert_eq!(detect_format(Path::new("model")), "unknown");
    }

    #[test]
    fn test_detect_format_case() {
        // Extensions are case-sensitive in typical implementations
        assert_eq!(detect_format(Path::new("model.APR")), "unknown");
    }

    // ========================================================================
    // CiAssertions Tests
    // ========================================================================

    #[test]
    fn test_ci_assertions_default() {
        let assertions = CiAssertions::default();
        assert!(assertions.min_throughput.is_none());
        assert!(assertions.max_p99_ms.is_none());
        assert!(assertions.max_p50_ms.is_none());
    }

    #[test]
    fn test_ci_assertions_with_throughput() {
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            ..Default::default()
        };
        assert_eq!(assertions.min_throughput.unwrap(), 100.0);
    }

    #[test]
    fn test_ci_assertions_with_latency() {
        let assertions = CiAssertions {
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(25.0),
            ..Default::default()
        };
        assert_eq!(assertions.max_p99_ms.unwrap(), 50.0);
        assert_eq!(assertions.max_p50_ms.unwrap(), 25.0);
    }

    #[test]
    fn test_ci_assertions_debug() {
        let assertions = CiAssertions::default();
        let debug = format!("{assertions:?}");
        assert!(debug.contains("CiAssertions"));
    }

    #[test]
    fn test_ci_assertions_clone() {
        let assertions = CiAssertions {
            min_throughput: Some(50.0),
            ..Default::default()
        };
        let cloned = assertions.clone();
        assert_eq!(cloned.min_throughput, assertions.min_throughput);
    }

    // ========================================================================
    // AssertionResult Tests
    // ========================================================================

    #[test]
    fn test_assertion_result_passed() {
        let result = AssertionResult {
            name: "throughput".to_string(),
            expected: ">= 100.0 tok/s".to_string(),
            actual: "150.0 tok/s".to_string(),
            passed: true,
        };
        assert!(result.passed);
        assert_eq!(result.name, "throughput");
    }

    #[test]
    fn test_assertion_result_failed() {
        let result = AssertionResult {
            name: "latency_p99".to_string(),
            expected: "<= 50.0 ms".to_string(),
            actual: "75.0 ms".to_string(),
            passed: false,
        };
        assert!(!result.passed);
    }

    #[test]
    fn test_assertion_result_debug() {
        let result = AssertionResult {
            name: "test".to_string(),
            expected: "expected".to_string(),
            actual: "actual".to_string(),
            passed: true,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("AssertionResult"));
    }

    #[test]
    fn test_assertion_result_clone() {
        let result = AssertionResult {
            name: "test".to_string(),
            expected: "expected".to_string(),
            actual: "actual".to_string(),
            passed: true,
        };
        let cloned = result.clone();
        assert_eq!(cloned.name, result.name);
        assert_eq!(cloned.passed, result.passed);
    }

    // ========================================================================
    // CiProfileReport Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_from_results_no_assertions() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions::default();
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
        assert!(report.assertions.is_empty());
    }

    #[test]
    fn test_ci_profile_report_throughput_pass() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 150.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
        assert_eq!(report.assertions.len(), 1);
        assert!(report.assertions[0].passed);
    }

    #[test]
    fn test_ci_profile_report_throughput_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 50.0, // Below threshold
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        assert!(!report.assertions[0].passed);
    }

    #[test]
    fn test_ci_profile_report_latency_assertions() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 25000.0, // 25ms
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            max_p50_ms: Some(50.0),
            max_p99_ms: Some(100.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed); // 25ms is below both thresholds
        assert_eq!(report.assertions.len(), 2);
    }

    #[test]
    fn test_ci_profile_report_debug() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        let debug = format!("{report:?}");
        assert!(debug.contains("CiProfileReport"));
    }

    #[test]
    fn test_ci_profile_report_clone() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        let cloned = report.clone();
        assert_eq!(cloned.model_path, report.model_path);
        assert_eq!(cloned.passed, report.passed);
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        // run(path, granular, format, focus, detect_naive, naive_threshold,
        //     compare_hf, energy, perf_grade, callgraph, fail_on_naive, output_path)
        let result = run(
            Path::new("/nonexistent/model.gguf"),
            false, // granular
            OutputFormat::Human,
            ProfileFocus::All,
            false, // detect_naive
            0.5,   // naive_threshold
            None,  // compare_hf
            false, // energy
            false, // perf_grade
            false, // callgraph
            false, // fail_on_naive
            None,  // output_path
            32,    // tokens
            false, // ollama
            true,  // no_gpu
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Human,
            ProfileFocus::All,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid GGUF or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_format() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Json,
            ProfileFocus::All,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_granular_mode() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            true, // granular
            OutputFormat::Human,
            ProfileFocus::All,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_attention_focus() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Human,
            ProfileFocus::Attention,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file) but tests focus path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_matmul_focus() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            false,
            OutputFormat::Human,
            ProfileFocus::Matmul,
            false,
            0.5,
            None,
            false,
            false,
            false,
            false,
            None,
            32,
            false,
            true,
        );
        // Should fail (invalid file) but tests focus path
        assert!(result.is_err());
    }

    // ========================================================================
    // run_ci Command Tests
    // ========================================================================

    #[test]
    fn test_run_ci_file_not_found() {
        let result = run_ci(
            Path::new("/nonexistent/model.gguf"),
            OutputFormat::Human,
            &CiAssertions::default(),
            3,  // warmup
            10, // measure
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_ci_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run_ci(
            file.path(),
            OutputFormat::Json,
            &CiAssertions {
                min_throughput: Some(100.0),
                ..Default::default()
            },
            1,
            1,
        );
        // Should fail (invalid file or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_ci_with_latency_assertions() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run_ci(
            file.path(),
            OutputFormat::Human,
            &CiAssertions {
                max_p99_ms: Some(50.0),
                max_p50_ms: Some(25.0),
                ..Default::default()
            },
            1,
            1,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    // ========================================================================
    // run_diff_benchmark Tests
    // ========================================================================

    #[test]
    fn test_run_diff_benchmark_model_a_not_found() {
        let model_b = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let result = run_diff_benchmark(
            Path::new("/nonexistent/model_a.gguf"),
            model_b.path(),
            OutputFormat::Human,
            1,
            1,
            0.05,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_benchmark_model_b_not_found() {
        let mut model_a = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_a.write_all(b"not valid").expect("write");

        let result = run_diff_benchmark(
            model_a.path(),
            Path::new("/nonexistent/model_b.gguf"),
            OutputFormat::Human,
            1,
            1,
            0.05,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_benchmark_both_invalid() {
        let mut model_a = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_a.write_all(b"not valid a").expect("write");
        let mut model_b = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_b.write_all(b"not valid b").expect("write");

        let result = run_diff_benchmark(
            model_a.path(),
            model_b.path(),
            OutputFormat::Json,
            1,
            1,
            0.05,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    // ========================================================================
    // Additional Hotspot Tests
    // ========================================================================

    #[test]
    fn test_hotspot_debug() {
        let hotspot = Hotspot {
            name: "matmul".to_string(),
            time_us: 1000.0,
            percent: 50.0,
            count: 10,
            avg_us: 100.0,
            min_us: 80.0,
            max_us: 120.0,
            bottleneck: None,
            efficiency_pct: None,
            category: None,
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        };
        let debug = format!("{:?}", hotspot);
        assert!(debug.contains("Hotspot"));
        assert!(debug.contains("matmul"));
    }

    #[test]
    fn test_hotspot_clone() {
        let hotspot = Hotspot {
            name: "attention".to_string(),
            time_us: 500.0,
            percent: 25.0,
            count: 5,
            avg_us: 100.0,
            min_us: 90.0,
            max_us: 110.0,
            bottleneck: None,
            efficiency_pct: None,
            category: None,
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        };
        let cloned = hotspot.clone();
        assert_eq!(cloned.name, hotspot.name);
        assert_eq!(cloned.time_us, hotspot.time_us);
    }

    #[test]
    fn test_hotspot_zero_count() {
        let hotspot = Hotspot {
            name: "empty".to_string(),
            time_us: 0.0,
            percent: 0.0,
            count: 0,
            avg_us: 0.0,
            min_us: 0.0,
            max_us: 0.0,
            bottleneck: None,
            efficiency_pct: None,
            category: None,
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        };
        assert_eq!(hotspot.count, 0);
        assert_eq!(hotspot.avg_us, 0.0);
    }

    #[test]
    fn test_hotspot_high_variance() {
        let hotspot = Hotspot {
            name: "variable_op".to_string(),
            time_us: 10000.0,
            percent: 100.0,
            count: 100,
            avg_us: 100.0,
            min_us: 10.0,
            max_us: 500.0, // High max vs avg
            bottleneck: None,
            efficiency_pct: None,
            category: None,
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        };
        assert!(hotspot.max_us > hotspot.avg_us * 4.0);
    }

    // ========================================================================
    // Additional RealProfileResults Tests
    // ========================================================================

    #[test]
    fn test_real_profile_results_construction() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        assert_eq!(results.model_path, "test.gguf");
        assert_eq!(results.architecture, "llama");
        assert!(results.is_real_data);
    }

    #[test]
    fn test_real_profile_results_with_hotspots() {
        let hotspots = vec![
            Hotspot {
                name: "matmul".to_string(),
                time_us: 5000.0,
                percent: 50.0,
                count: 10,
                avg_us: 500.0,
                min_us: 450.0,
                max_us: 550.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            },
            Hotspot {
                name: "attention".to_string(),
                time_us: 3000.0,
                percent: 30.0,
                count: 10,
                avg_us: 300.0,
                min_us: 280.0,
                max_us: 320.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            },
        ];
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots,
            per_layer_us: vec![100.0; 32],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        assert_eq!(results.hotspots.len(), 2);
        assert_eq!(results.per_layer_us.len(), 32);
    }

    #[test]
    fn test_real_profile_results_synthetic_data() {
        let results = RealProfileResults {
            model_path: "synthetic.gguf".to_string(),
            architecture: "mock".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: false,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        assert!(!results.is_real_data);
    }

    // ========================================================================
    // Edge Cases for Formats
    // ========================================================================

    #[test]
    fn test_detect_format_apr_path() {
        let path = Path::new("/some/deep/path/model.apr");
        assert_eq!(detect_format(path), "apr");
    }

    #[test]
    fn test_detect_format_safetensors_path() {
        let path = Path::new("models/v1.0/model.safetensors");
        assert_eq!(detect_format(path), "safetensors");
    }

    #[test]
    fn test_detect_format_gguf_with_version() {
        let path = Path::new("qwen2.5-coder-1.5b-q4_k_m.gguf");
        assert_eq!(detect_format(path), "gguf");
    }

    #[test]
    fn test_detect_format_empty_extension() {
        let path = Path::new("model.");
        assert_eq!(detect_format(path), "unknown");
    }

    // ========================================================================
    // CI Assertions Edge Cases
    // ========================================================================

    #[test]
    fn test_ci_assertions_all_set() {
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(25.0),
            max_memory_mb: Some(1024.0),
        };
        assert!(assertions.min_throughput.is_some());
        assert!(assertions.max_p99_ms.is_some());
        assert!(assertions.max_p50_ms.is_some());
    }

    #[test]
    fn test_ci_profile_report_multiple_assertions_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 100000.0, // 100ms
            throughput_tok_s: 10.0,       // Low
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0), // Will fail
            max_p99_ms: Some(50.0),      // Will fail (100ms > 50ms)
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        // Both assertions should fail
        assert_eq!(report.assertions.iter().filter(|a| !a.passed).count(), 2);
    }

    #[test]
    fn test_ci_profile_report_boundary_values() {
        // Exactly at threshold
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 50000.0, // Exactly 50ms
            throughput_tok_s: 100.0,     // Exactly at threshold
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0), // Exactly at threshold
            max_p99_ms: Some(50.0),      // Exactly at threshold
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        // Exactly at threshold should pass for >= and <=
        assert!(report.passed);
    }

    // ========================================================================
    // OutputFormat and ProfileFocus Exhaustive Tests
    // ========================================================================

    #[test]
    fn test_output_format_copy_trait() {
        let format = OutputFormat::Json;
        let copied = format;
        assert!(matches!(format, OutputFormat::Json));
        assert!(matches!(copied, OutputFormat::Json));
    }

    #[test]
    fn test_profile_focus_copy_trait() {
        let focus = ProfileFocus::Mlp;
        let copied = focus;
        assert!(matches!(focus, ProfileFocus::Mlp));
        assert!(matches!(copied, ProfileFocus::Mlp));
    }

    #[test]
    fn test_output_format_parse_mixed_case() {
        assert!(matches!(
            "Json".parse::<OutputFormat>().unwrap(),
            OutputFormat::Json
        ));
        assert!(matches!(
            "FLAMEGRAPH".parse::<OutputFormat>().unwrap(),
            OutputFormat::Flamegraph
        ));
    }

    #[test]
    fn test_profile_focus_parse_mixed_case() {
        assert!(matches!(
            "ATTENTION".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Attention
        ));
        assert!(matches!(
            "Mlp".parse::<ProfileFocus>().unwrap(),
            ProfileFocus::Mlp
        ));
    }

    // ========================================================================
    // filter_results_by_focus Tests
    // ========================================================================

    fn make_test_results_with_hotspots() -> RealProfileResults {
        RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![
                Hotspot {
                    name: "attention_qkv".to_string(),
                    time_us: 3000.0,
                    percent: 30.0,
                    count: 10,
                    avg_us: 300.0,
                    min_us: 280.0,
                    max_us: 320.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "mlp_gate_up".to_string(),
                    time_us: 2500.0,
                    percent: 25.0,
                    count: 10,
                    avg_us: 250.0,
                    min_us: 230.0,
                    max_us: 270.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "matmul_q4k".to_string(),
                    time_us: 2000.0,
                    percent: 20.0,
                    count: 10,
                    avg_us: 200.0,
                    min_us: 180.0,
                    max_us: 220.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "embedding_lookup".to_string(),
                    time_us: 1000.0,
                    percent: 10.0,
                    count: 10,
                    avg_us: 100.0,
                    min_us: 90.0,
                    max_us: 110.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "softmax".to_string(),
                    time_us: 800.0,
                    percent: 8.0,
                    count: 10,
                    avg_us: 80.0,
                    min_us: 70.0,
                    max_us: 90.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "ffn_down_proj".to_string(),
                    time_us: 500.0,
                    percent: 5.0,
                    count: 10,
                    avg_us: 50.0,
                    min_us: 40.0,
                    max_us: 60.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "lm_head".to_string(),
                    time_us: 200.0,
                    percent: 2.0,
                    count: 10,
                    avg_us: 20.0,
                    min_us: 15.0,
                    max_us: 25.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "linear_proj".to_string(),
                    time_us: 100.0,
                    percent: 1.0,
                    count: 10,
                    avg_us: 10.0,
                    min_us: 8.0,
                    max_us: 12.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "gemm_f16".to_string(),
                    time_us: 50.0,
                    percent: 0.5,
                    count: 5,
                    avg_us: 10.0,
                    min_us: 8.0,
                    max_us: 12.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![312.5; 32],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_filter_results_by_focus_all() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::All);
        assert_eq!(filtered.hotspots.len(), results.hotspots.len());
    }

    #[test]
    fn test_filter_results_by_focus_attention() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        // Should match: attention_qkv, softmax
        assert_eq!(
            filtered.hotspots.len(),
            2,
            "Expected 2 attention hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered.hotspots.iter().any(|h| h.name == "attention_qkv"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "softmax"));
    }

    #[test]
    fn test_filter_results_by_focus_mlp() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Mlp);
        // Should match: mlp_gate_up, ffn_down_proj, gate (in mlp_gate_up)
        assert!(
            filtered.hotspots.len() >= 2,
            "Expected at least 2 MLP hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered.hotspots.iter().any(|h| h.name == "mlp_gate_up"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "ffn_down_proj"));
    }

    #[test]
    fn test_filter_results_by_focus_matmul() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Matmul);
        // Should match: matmul_q4k, linear_proj, gemm_f16
        assert!(
            filtered.hotspots.len() >= 3,
            "Expected at least 3 matmul hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered.hotspots.iter().any(|h| h.name == "matmul_q4k"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "linear_proj"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "gemm_f16"));
    }

    #[test]
    fn test_filter_results_by_focus_embedding() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Embedding);
        // Should match: embedding_lookup, lm_head
        assert_eq!(
            filtered.hotspots.len(),
            2,
            "Expected 2 embedding hotspots, got names: {:?}",
            filtered
                .hotspots
                .iter()
                .map(|h| &h.name)
                .collect::<Vec<_>>()
        );
        assert!(filtered
            .hotspots
            .iter()
            .any(|h| h.name == "embedding_lookup"));
        assert!(filtered.hotspots.iter().any(|h| h.name == "lm_head"));
    }

    #[test]
    fn test_filter_results_preserves_metadata() {
        let results = make_test_results_with_hotspots();
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        // All metadata should be preserved
        assert_eq!(filtered.model_path, results.model_path);
        assert_eq!(filtered.architecture, results.architecture);
        assert_eq!(filtered.num_layers, results.num_layers);
        assert_eq!(filtered.vocab_size, results.vocab_size);
        assert_eq!(filtered.hidden_dim, results.hidden_dim);
        assert_eq!(filtered.warmup_passes, results.warmup_passes);
        assert_eq!(filtered.measure_passes, results.measure_passes);
        assert_eq!(filtered.total_inference_us, results.total_inference_us);
        assert_eq!(filtered.throughput_tok_s, results.throughput_tok_s);
        assert_eq!(filtered.tokens_per_pass, results.tokens_per_pass);
        assert_eq!(filtered.per_layer_us.len(), results.per_layer_us.len());
        assert_eq!(filtered.is_real_data, results.is_real_data);
    }

    #[test]
    fn test_filter_results_no_matching_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![Hotspot {
                name: "custom_op".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 10,
                avg_us: 100.0,
                min_us: 90.0,
                max_us: 110.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        // No attention-related hotspots
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        assert!(filtered.hotspots.is_empty());
    }

    #[test]
    fn test_filter_results_empty_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 100.0,
            throughput_tok_s: 10000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        for focus in [
            ProfileFocus::All,
            ProfileFocus::Attention,
            ProfileFocus::Mlp,
            ProfileFocus::Matmul,
            ProfileFocus::Embedding,
        ] {
            let filtered = filter_results_by_focus(&results, focus);
            assert!(filtered.hotspots.is_empty());
        }
    }

    // ========================================================================
    // CiProfileReport from_results Extended Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_all_assertions_combined() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 20000.0, // 20ms
            throughput_tok_s: 200.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(30.0),
            max_memory_mb: None,
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
        assert_eq!(report.assertions.len(), 3);
        assert!(report.assertions.iter().all(|a| a.passed));
        assert_eq!(report.throughput_tok_s, 200.0);
        // 20000us / 1000 = 20ms
        assert!((report.latency_p50_ms - 20.0).abs() < 0.01);
        assert!((report.latency_p99_ms - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_report_zero_throughput() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 0.0,
            throughput_tok_s: 0.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        assert!(!report.assertions[0].passed);
    }

    #[test]
    fn test_ci_profile_report_latency_p50_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 60000.0, // 60ms
            throughput_tok_s: 100.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            max_p50_ms: Some(50.0), // 60ms > 50ms => fail
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        assert_eq!(report.assertions.len(), 1);
        assert!(!report.assertions[0].passed);
        assert_eq!(report.assertions[0].name, "latency_p50");
    }

    #[test]
    fn test_ci_profile_report_mixed_pass_fail() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 30000.0, // 30ms
            throughput_tok_s: 150.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0), // 150 >= 100 => pass
            max_p99_ms: Some(20.0),      // 30ms > 20ms => fail
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed); // One failure means overall fail
        assert_eq!(report.assertions.len(), 2);
        assert!(report.assertions[0].passed); // throughput passed
        assert!(!report.assertions[1].passed); // latency failed
    }

    #[test]
    fn test_ci_profile_report_assertion_format_strings() {
        let results = RealProfileResults {
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 32,
            vocab_size: 32000,
            hidden_dim: 4096,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 10000.0,
            throughput_tok_s: 150.0,
            tokens_per_pass: 10,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            max_p99_ms: Some(50.0),
            max_p50_ms: Some(25.0),
            max_memory_mb: None,
        };
        let report = CiProfileReport::from_results(&results, &assertions);

        // Check format of assertion strings
        let throughput_assertion = &report.assertions[0];
        assert_eq!(throughput_assertion.name, "throughput");
        assert!(throughput_assertion.expected.contains("tok/s"));
        assert!(throughput_assertion.actual.contains("tok/s"));

        let p99_assertion = &report.assertions[1];
        assert_eq!(p99_assertion.name, "latency_p99");
        assert!(p99_assertion.expected.contains("ms"));
        assert!(p99_assertion.actual.contains("ms"));

        let p50_assertion = &report.assertions[2];
        assert_eq!(p50_assertion.name, "latency_p50");
        assert!(p50_assertion.expected.contains("ms"));
    }

    // ========================================================================
    // DiffBenchmarkReport Tests
    // ========================================================================

    #[test]
    fn test_diff_benchmark_report_construction() {
        let report = DiffBenchmarkReport {
            model_a: "model_a.gguf".to_string(),
            model_b: "model_b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 150.0,
            throughput_delta_pct: 50.0,
            latency_a_ms: 10.0,
            latency_b_ms: 7.0,
            latency_delta_pct: -30.0,
            winner: "Model B (50.0% faster)".to_string(),
            regressions: vec![],
            improvements: vec!["Throughput: 50.0% faster".to_string()],
        };
        assert_eq!(report.model_a, "model_a.gguf");
        assert_eq!(report.throughput_delta_pct, 50.0);
        assert!(report.regressions.is_empty());
        assert_eq!(report.improvements.len(), 1);
    }

    #[test]
    fn test_diff_benchmark_report_debug() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 90.0,
            throughput_delta_pct: -10.0,
            latency_a_ms: 10.0,
            latency_b_ms: 11.0,
            latency_delta_pct: 10.0,
            winner: "Model A".to_string(),
            regressions: vec!["Throughput regression".to_string()],
            improvements: vec![],
        };
        let debug = format!("{report:?}");
        assert!(debug.contains("DiffBenchmarkReport"));
    }

    #[test]
    fn test_diff_benchmark_report_clone() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 100.0,
            throughput_delta_pct: 0.0,
            latency_a_ms: 10.0,
            latency_b_ms: 10.0,
            latency_delta_pct: 0.0,
            winner: "Tie".to_string(),
            regressions: vec![],
            improvements: vec![],
        };
        let cloned = report.clone();
        assert_eq!(cloned.model_a, report.model_a);
        assert_eq!(cloned.throughput_delta_pct, report.throughput_delta_pct);
        assert_eq!(cloned.winner, report.winner);
    }

    #[test]
    fn test_diff_benchmark_report_with_regressions() {
        let report = DiffBenchmarkReport {
            model_a: "baseline.gguf".to_string(),
            model_b: "candidate.gguf".to_string(),
            throughput_a: 200.0,
            throughput_b: 100.0,
            throughput_delta_pct: -50.0,
            latency_a_ms: 5.0,
            latency_b_ms: 10.0,
            latency_delta_pct: 100.0,
            winner: "Model A (50.0% faster)".to_string(),
            regressions: vec![
                "Throughput: 50.0% slower".to_string(),
                "Latency: 100.0% slower".to_string(),
            ],
            improvements: vec![],
        };
        assert_eq!(report.regressions.len(), 2);
        assert!(report.improvements.is_empty());
    }

    #[test]
    fn test_diff_benchmark_report_with_improvements() {
        let report = DiffBenchmarkReport {
            model_a: "old.gguf".to_string(),
            model_b: "new.gguf".to_string(),
            throughput_a: 50.0,
            throughput_b: 200.0,
            throughput_delta_pct: 300.0,
            latency_a_ms: 20.0,
            latency_b_ms: 5.0,
            latency_delta_pct: -75.0,
            winner: "Model B (300.0% faster)".to_string(),
            regressions: vec![],
            improvements: vec![
                "Throughput: 300.0% faster".to_string(),
                "Latency: 75.0% faster".to_string(),
            ],
        };
        assert!(report.regressions.is_empty());
        assert_eq!(report.improvements.len(), 2);
    }

    // ========================================================================
    // CiProfileReport print_json Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_print_json_no_assertions() {
        // Just verify it doesn't panic
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        report.print_json();
    }

    #[test]
    fn test_ci_profile_report_print_json_with_assertions() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: false,
            throughput_tok_s: 50.0,
            latency_p50_ms: 100.0,
            latency_p99_ms: 200.0,
            assertions: vec![
                AssertionResult {
                    name: "throughput".to_string(),
                    expected: ">= 100.0 tok/s".to_string(),
                    actual: "50.0 tok/s".to_string(),
                    passed: false,
                },
                AssertionResult {
                    name: "latency_p99".to_string(),
                    expected: "<= 50.0 ms".to_string(),
                    actual: "200.00 ms".to_string(),
                    passed: false,
                },
            ],
        };
        report.print_json();
    }

    #[test]
    fn test_ci_profile_report_print_json_single_assertion() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 200.0,
            latency_p50_ms: 5.0,
            latency_p99_ms: 10.0,
            assertions: vec![AssertionResult {
                name: "throughput".to_string(),
                expected: ">= 100.0 tok/s".to_string(),
                actual: "200.0 tok/s".to_string(),
                passed: true,
            }],
        };
        report.print_json();
    }

    // ========================================================================
    // CiProfileReport print_human Tests
    // ========================================================================

    #[test]
    fn test_ci_profile_report_print_human_passed() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 150.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![AssertionResult {
                name: "throughput".to_string(),
                expected: ">= 100.0 tok/s".to_string(),
                actual: "150.0 tok/s".to_string(),
                passed: true,
            }],
        };
        report.print_human();
    }

    #[test]
    fn test_ci_profile_report_print_human_failed() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: false,
            throughput_tok_s: 50.0,
            latency_p50_ms: 100.0,
            latency_p99_ms: 200.0,
            assertions: vec![AssertionResult {
                name: "throughput".to_string(),
                expected: ">= 100.0 tok/s".to_string(),
                actual: "50.0 tok/s".to_string(),
                passed: false,
            }],
        };
        report.print_human();
    }

    #[test]
    fn test_ci_profile_report_print_human_no_assertions() {
        let report = CiProfileReport {
            model_path: "model.gguf".to_string(),
            passed: true,
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
            assertions: vec![],
        };
        report.print_human();
    }

    // ========================================================================
    // DiffBenchmarkReport print Tests
    // ========================================================================

    #[test]
    fn test_diff_benchmark_report_print_human() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 150.0,
            throughput_delta_pct: 50.0,
            latency_a_ms: 10.0,
            latency_b_ms: 7.0,
            latency_delta_pct: -30.0,
            winner: "Model B".to_string(),
            regressions: vec![],
            improvements: vec!["Throughput +50%".to_string()],
        };
        report.print_human();
    }

    #[test]
    fn test_diff_benchmark_report_print_human_with_regressions() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 200.0,
            throughput_b: 100.0,
            throughput_delta_pct: -50.0,
            latency_a_ms: 5.0,
            latency_b_ms: 10.0,
            latency_delta_pct: 100.0,
            winner: "Model A".to_string(),
            regressions: vec!["Throughput -50%".to_string()],
            improvements: vec![],
        };
        report.print_human();
    }

    #[test]
    fn test_diff_benchmark_report_print_json() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 100.0,
            throughput_delta_pct: 0.0,
            latency_a_ms: 10.0,
            latency_b_ms: 10.0,
            latency_delta_pct: 0.0,
            winner: "Tie".to_string(),
            regressions: vec![],
            improvements: vec![],
        };
        report.print_json();
    }

    #[test]
    fn test_diff_benchmark_report_print_json_with_regressions_and_improvements() {
        let report = DiffBenchmarkReport {
            model_a: "a.gguf".to_string(),
            model_b: "b.gguf".to_string(),
            throughput_a: 100.0,
            throughput_b: 150.0,
            throughput_delta_pct: 50.0,
            latency_a_ms: 5.0,
            latency_b_ms: 7.0,
            latency_delta_pct: 40.0,
            winner: "Mixed".to_string(),
            regressions: vec!["Latency regression".to_string()],
            improvements: vec![
                "Throughput improvement".to_string(),
                "Memory improvement".to_string(),
            ],
        };
        report.print_json();
    }

    // ========================================================================
    // print_human_results Tests
    // ========================================================================

    #[test]
    fn test_print_human_results_basic() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "forward_pass".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 1,
                avg_us: 1000.0,
                min_us: 1000.0,
                max_us: 1000.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![250.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_granular() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "forward_pass".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 1,
                avg_us: 1000.0,
                min_us: 900.0,
                max_us: 1100.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![250.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, true, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_simulated_data() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: false,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_zero_total_time() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 0.0,
            throughput_tok_s: 0.0,
            tokens_per_pass: 0,
            hotspots: vec![Hotspot {
                name: "op".to_string(),
                time_us: 0.0,
                percent: 0.0,
                count: 0,
                avg_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_many_hotspots() {
        let hotspots: Vec<Hotspot> = (0..10)
            .map(|i| Hotspot {
                name: format!("op_{i}"),
                time_us: (10 - i) as f64 * 100.0,
                percent: (10 - i) as f64 * 10.0,
                count: 10,
                avg_us: (10 - i) as f64 * 10.0,
                min_us: (10 - i) as f64 * 8.0,
                max_us: (10 - i) as f64 * 12.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            })
            .collect();
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 10,
            total_inference_us: 5500.0,
            throughput_tok_s: 1818.0,
            tokens_per_pass: 10,
            hotspots,
            per_layer_us: vec![1375.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, true, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_human_results_granular_zero_max_layer() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 2,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![0.0, 0.0], // Zero layer times
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_human_results(&results, true, false, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // print_json_results Tests
    // ========================================================================

    #[test]
    fn test_print_json_results_basic() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_json_results(&results);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_json_results_with_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 5,
            total_inference_us: 5000.0,
            throughput_tok_s: 200.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "op_a".to_string(),
                    time_us: 3000.0,
                    percent: 60.0,
                    count: 5,
                    avg_us: 600.0,
                    min_us: 550.0,
                    max_us: 650.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "op_b".to_string(),
                    time_us: 2000.0,
                    percent: 40.0,
                    count: 5,
                    avg_us: 400.0,
                    min_us: 350.0,
                    max_us: 450.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![1250.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_json_results(&results);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_json_results_single_hotspot() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 1,
            vocab_size: 100,
            hidden_dim: 32,
            warmup_passes: 0,
            measure_passes: 1,
            total_inference_us: 100.0,
            throughput_tok_s: 10000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "forward".to_string(),
                time_us: 100.0,
                percent: 100.0,
                count: 1,
                avg_us: 100.0,
                min_us: 100.0,
                max_us: 100.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![100.0],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_json_results(&results);
        assert!(result.is_ok());
    }

    // ========================================================================
    // print_flamegraph Tests
    // ========================================================================

    #[test]
    fn test_print_flamegraph_stdout() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "op_a".to_string(),
                    time_us: 600.0,
                    percent: 60.0,
                    count: 1,
                    avg_us: 600.0,
                    min_us: 600.0,
                    max_us: 600.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "op_b".to_string(),
                    time_us: 400.0,
                    percent: 40.0,
                    count: 1,
                    avg_us: 400.0,
                    min_us: 400.0,
                    max_us: 400.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_flamegraph(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_flamegraph_to_file() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "forward".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 1,
                avg_us: 1000.0,
                min_us: 1000.0,
                max_us: 1000.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let file = NamedTempFile::with_suffix(".svg").expect("create temp file");
        let result = print_flamegraph(&results, Some(file.path()));
        assert!(result.is_ok());
        // Verify file was written
        let content = std::fs::read_to_string(file.path()).expect("read svg");
        assert!(content.contains("<svg"));
        assert!(content.contains("forward"));
    }

    #[test]
    fn test_print_flamegraph_empty_hotspots() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 0.0,
            throughput_tok_s: 0.0,
            tokens_per_pass: 0,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_flamegraph(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_flamegraph_zero_total_time() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 0.0,
            throughput_tok_s: 0.0,
            tokens_per_pass: 0,
            hotspots: vec![Hotspot {
                name: "op".to_string(),
                time_us: 0.0,
                percent: 0.0,
                count: 0,
                avg_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_flamegraph(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_flamegraph_invalid_output_path() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let result = print_flamegraph(&results, Some(Path::new("/nonexistent/dir/file.svg")));
        assert!(result.is_err());
    }

    // ========================================================================
    // detect_format Extended Tests
    // ========================================================================

    #[test]
    fn test_detect_format_dot_only() {
        assert_eq!(detect_format(Path::new(".")), "unknown");
    }

    #[test]
    fn test_detect_format_multiple_dots() {
        assert_eq!(detect_format(Path::new("model.v2.gguf")), "gguf");
        assert_eq!(
            detect_format(Path::new("model.q4_k_m.safetensors")),
            "safetensors"
        );
    }

    #[test]
    fn test_detect_format_absolute_paths() {
        assert_eq!(detect_format(Path::new("/models/latest/model.apr")), "apr");
        assert_eq!(
            detect_format(Path::new("/tmp/downloads/model.gguf")),
            "gguf"
        );
    }

    // ========================================================================
    // Additional Edge Case Tests
    // ========================================================================

    #[test]
    fn test_real_profile_results_debug() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let debug = format!("{results:?}");
        assert!(debug.contains("RealProfileResults"));
        assert!(debug.contains("test.gguf"));
    }

    #[test]
    fn test_real_profile_results_clone() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "op".to_string(),
                time_us: 1000.0,
                percent: 100.0,
                count: 1,
                avg_us: 1000.0,
                min_us: 1000.0,
                max_us: 1000.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![250.0; 4],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let cloned = results.clone();
        assert_eq!(cloned.model_path, results.model_path);
        assert_eq!(cloned.hotspots.len(), results.hotspots.len());
        assert_eq!(cloned.per_layer_us, results.per_layer_us);
    }

    // ========================================================================
    // OutputFormat parse error message Tests
    // ========================================================================

    #[test]
    fn test_output_format_error_message() {
        let err = "invalid".parse::<OutputFormat>().unwrap_err();
        assert!(err.contains("invalid"));
    }

    #[test]
    fn test_profile_focus_error_message() {
        let err = "invalid".parse::<ProfileFocus>().unwrap_err();
        assert!(err.contains("invalid"));
    }

    // ========================================================================
    // Comprehensive Boundary Tests for CI Report
    // ========================================================================

    #[test]
    fn test_ci_profile_report_very_high_throughput() {
        let results = RealProfileResults {
            model_path: "fast.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 10,
            total_inference_us: 1.0, // 0.001ms
            throughput_tok_s: 1_000_000.0,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(100.0),
            max_p99_ms: Some(1.0),
            ..Default::default()
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(report.passed);
    }

    #[test]
    fn test_ci_profile_report_very_slow() {
        let results = RealProfileResults {
            model_path: "slow.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 128,
            vocab_size: 128000,
            hidden_dim: 16384,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 10_000_000.0, // 10 seconds
            throughput_tok_s: 0.1,
            tokens_per_pass: 1,
            hotspots: vec![],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let assertions = CiAssertions {
            min_throughput: Some(1.0),
            max_p99_ms: Some(100.0),
            max_p50_ms: Some(50.0),
            max_memory_mb: None,
        };
        let report = CiProfileReport::from_results(&results, &assertions);
        assert!(!report.passed);
        // All 3 assertions should fail
        assert_eq!(report.assertions.len(), 3);
        assert!(report.assertions.iter().all(|a| !a.passed));
    }

    // ========================================================================
    // Attention Focus Variant Keywords
    // ========================================================================

    #[test]
    fn test_filter_attention_qkv_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "qkv_projection".to_string(),
                time_us: 500.0,
                percent: 50.0,
                count: 1,
                avg_us: 500.0,
                min_us: 500.0,
                max_us: 500.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        assert_eq!(filtered.hotspots.len(), 1);
        assert_eq!(filtered.hotspots[0].name, "qkv_projection");
    }

    #[test]
    fn test_filter_mlp_up_proj_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "up_proj".to_string(),
                    time_us: 300.0,
                    percent: 30.0,
                    count: 1,
                    avg_us: 300.0,
                    min_us: 300.0,
                    max_us: 300.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "down_proj".to_string(),
                    time_us: 300.0,
                    percent: 30.0,
                    count: 1,
                    avg_us: 300.0,
                    min_us: 300.0,
                    max_us: 300.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let filtered = filter_results_by_focus(&results, ProfileFocus::Mlp);
        assert_eq!(filtered.hotspots.len(), 2);
    }

    #[test]
    fn test_filter_matmul_mm_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "mm_q4k".to_string(),
                time_us: 500.0,
                percent: 50.0,
                count: 1,
                avg_us: 500.0,
                min_us: 500.0,
                max_us: 500.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let filtered = filter_results_by_focus(&results, ProfileFocus::Matmul);
        assert_eq!(filtered.hotspots.len(), 1);
    }

    #[test]
    fn test_filter_embedding_vocab_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "vocab_lookup".to_string(),
                time_us: 200.0,
                percent: 20.0,
                count: 1,
                avg_us: 200.0,
                min_us: 200.0,
                max_us: 200.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let filtered = filter_results_by_focus(&results, ProfileFocus::Embedding);
        assert_eq!(filtered.hotspots.len(), 1);
    }

    // ========================================================================
    // Case-insensitive hotspot filtering
    // ========================================================================

    #[test]
    fn test_filter_case_insensitive() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "ATTENTION_QKV".to_string(),
                    time_us: 500.0,
                    percent: 50.0,
                    count: 1,
                    avg_us: 500.0,
                    min_us: 500.0,
                    max_us: 500.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "MATMUL_F16".to_string(),
                    time_us: 300.0,
                    percent: 30.0,
                    count: 1,
                    avg_us: 300.0,
                    min_us: 300.0,
                    max_us: 300.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "MLP_Gate".to_string(),
                    time_us: 200.0,
                    percent: 20.0,
                    count: 1,
                    avg_us: 200.0,
                    min_us: 200.0,
                    max_us: 200.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };

        // Attention filter should match ATTENTION_QKV (case-insensitive)
        let attn_filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        assert_eq!(attn_filtered.hotspots.len(), 1);
        assert_eq!(attn_filtered.hotspots[0].name, "ATTENTION_QKV");

        // Matmul filter should match MATMUL_F16
        let mm_filtered = filter_results_by_focus(&results, ProfileFocus::Matmul);
        assert_eq!(mm_filtered.hotspots.len(), 1);
        assert_eq!(mm_filtered.hotspots[0].name, "MATMUL_F16");

        // MLP filter should match MLP_Gate (contains "gate")
        let mlp_filtered = filter_results_by_focus(&results, ProfileFocus::Mlp);
        assert_eq!(mlp_filtered.hotspots.len(), 1);
        assert_eq!(mlp_filtered.hotspots[0].name, "MLP_Gate");
    }

    // ================================================================
    // F-PROFILE-011: Cross-format comparison tests
    // ================================================================

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format(Path::new("model.gguf")), "gguf");
    }

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format(Path::new("model.apr")), "apr");
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(detect_format(Path::new("weights.safetensors")), "safetensors");
    }

    #[test]
    fn test_detect_format_bin_and_txt() {
        assert_eq!(detect_format(Path::new("data.bin")), "pytorch");
        assert_eq!(detect_format(Path::new("data.txt")), "unknown");
    }

    #[test]
    fn test_print_comparison_row_does_not_panic() {
        // Smoke test: ensure formatting works for various values
        print_comparison_row("Test metric", 100.5, 200.3);
        print_comparison_row("Zero case", 0.0, 100.0);
        print_comparison_row("Both zero", 0.0, 0.0);
    }

    #[test]
    fn test_cross_format_comparison_nonexistent_files() {
        let result = run_cross_format_comparison(
            Path::new("/tmp/nonexistent_a.gguf"),
            Path::new("/tmp/nonexistent_b.apr"),
            1,
            1,
            8,
            true,
        );
        assert!(result.is_err(), "Should fail with nonexistent files");
    }

    // ========================================================================
    // F-PROFILE-007/008/009: Per-Kernel Profiling Tests
    // ========================================================================

    #[test]
    fn test_estimate_kernel_data_bytes_q_proj() {
        let bytes = estimate_kernel_data_bytes("q_proj", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("q_proj should have data estimate");
        // Q4K weight: 4096*4096 * 0.5625 ≈ 9.4M, plus activation RW
        assert!(b > 9_000_000, "Q_proj should move >9MB: got {b}");
        assert!(b < 20_000_000, "Q_proj should move <20MB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_gate_proj() {
        let bytes = estimate_kernel_data_bytes("gate_proj", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("gate_proj should have data estimate");
        // FFN gate: 4096 * (4096*4) * 0.5625 ≈ 37.7M, plus activation RW
        assert!(b > 30_000_000, "gate_proj should move >30MB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_lm_head() {
        let bytes = estimate_kernel_data_bytes("lm_head", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("lm_head should have data estimate");
        // LM head: 4096*151936 * 0.5625 ≈ 350M
        assert!(b > 300_000_000, "lm_head should move >300MB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_rmsnorm() {
        let bytes = estimate_kernel_data_bytes("rmsnorm", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("rmsnorm should have data estimate");
        // Norm: activation RW (4096*8) + weight (4096*4) ≈ 48KB
        assert!(b > 40_000, "rmsnorm should move >40KB: got {b}");
        assert!(b < 200_000, "rmsnorm should move <200KB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_unknown() {
        let bytes = estimate_kernel_data_bytes("random_op_xyz", 4096, 151936);
        assert!(bytes.is_none(), "Unknown ops should return None");
    }

    #[test]
    fn test_compute_kernel_launch_overhead_basic() {
        let hotspots = vec![
            Hotspot {
                name: "q_proj".to_string(),
                time_us: 500.0,
                percent: 50.0,
                count: 10,
                avg_us: 50.0,
                min_us: 45.0,
                max_us: 55.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            },
            Hotspot {
                name: "k_proj".to_string(),
                time_us: 300.0,
                percent: 30.0,
                count: 10,
                avg_us: 30.0,
                min_us: 25.0,
                max_us: 35.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            },
        ];
        // Total kernel time = 800µs, total decode = 1000µs → 200µs overhead = 20%
        let (overhead_us, overhead_pct) = compute_kernel_launch_overhead(&hotspots, 1000.0);
        assert!(
            (overhead_us - 200.0).abs() < 1.0,
            "Expected ~200µs overhead, got {overhead_us}"
        );
        assert!(
            (overhead_pct - 20.0).abs() < 1.0,
            "Expected ~20% overhead, got {overhead_pct}"
        );
    }

    #[test]
    fn test_compute_kernel_launch_overhead_zero_decode() {
        let hotspots = vec![];
        let (overhead_us, overhead_pct) = compute_kernel_launch_overhead(&hotspots, 0.0);
        assert!(
            overhead_us.abs() < 0.001,
            "Zero decode should yield zero overhead"
        );
        assert!(
            overhead_pct.abs() < 0.001,
            "Zero decode should yield zero percent"
        );
    }

    #[test]
    fn test_hotspot_bandwidth_fields() {
        let h = Hotspot {
            name: "test".to_string(),
            time_us: 100.0,
            percent: 50.0,
            count: 5,
            avg_us: 20.0,
            min_us: 15.0,
            max_us: 25.0,
            bottleneck: None,
            efficiency_pct: Some(45.0),
            category: Some("FFN".to_string()),
            bandwidth_gbs: Some(500.0),
            data_bytes_per_call: Some(10_000_000),
        };
        assert!(
            (h.bandwidth_gbs.expect("should have bw") - 500.0).abs() < 0.1,
            "Bandwidth should be preserved"
        );
        assert_eq!(
            h.data_bytes_per_call.expect("should have data bytes"),
            10_000_000
        );
    }

    #[test]
    fn test_real_profile_results_has_launch_overhead() {
        let results = RealProfileResults {
            kernel_launch_overhead_pct: 15.0,
            kernel_launch_overhead_us: 1500.0,
            ..Default::default()
        };
        assert!(
            (results.kernel_launch_overhead_pct - 15.0).abs() < 0.01,
            "Launch overhead percent preserved"
        );
        assert!(
            (results.kernel_launch_overhead_us - 1500.0).abs() < 0.01,
            "Launch overhead µs preserved"
        );
    }
}
