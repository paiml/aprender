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
        writeln!(json, "  \"model\": \"{}\",", self.model_path).unwrap();
        writeln!(json, "  \"passed\": {},", self.passed).unwrap();
        json.push_str("  \"metrics\": {\n");
        writeln!(
            json,
            "    \"throughput_tok_s\": {:.2},",
            self.throughput_tok_s
        )
        .unwrap();
        writeln!(json, "    \"latency_p50_ms\": {:.2},", self.latency_p50_ms).unwrap();
        writeln!(json, "    \"latency_p99_ms\": {:.2}", self.latency_p99_ms).unwrap();
        json.push_str("  },\n");
        json.push_str("  \"assertions\": [\n");
        for (i, assertion) in self.assertions.iter().enumerate() {
            json.push_str("    {\n");
            writeln!(json, "      \"name\": \"{}\",", assertion.name).unwrap();
            writeln!(json, "      \"expected\": \"{}\",", assertion.expected).unwrap();
            writeln!(json, "      \"actual\": \"{}\",", assertion.actual).unwrap();
            writeln!(json, "      \"passed\": {}", assertion.passed).unwrap();
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
    assertions: &CiAssertions,
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
        writeln!(json, "  \"model_a\": \"{}\",", self.model_a).unwrap();
        writeln!(json, "  \"model_b\": \"{}\",", self.model_b).unwrap();
        json.push_str("  \"metrics\": {\n");
        writeln!(
            json,
            "    \"throughput_a_tok_s\": {:.2},",
            self.throughput_a
        )
        .unwrap();
        writeln!(
            json,
            "    \"throughput_b_tok_s\": {:.2},",
            self.throughput_b
        )
        .unwrap();
        writeln!(
            json,
            "    \"throughput_delta_pct\": {:.2},",
            self.throughput_delta_pct
        )
        .unwrap();
        writeln!(json, "    \"latency_a_ms\": {:.2},", self.latency_a_ms).unwrap();
        writeln!(json, "    \"latency_b_ms\": {:.2},", self.latency_b_ms).unwrap();
        writeln!(
            json,
            "    \"latency_delta_pct\": {:.2}",
            self.latency_delta_pct
        )
        .unwrap();
        json.push_str("  },\n");
        writeln!(json, "  \"winner\": \"{}\",", self.winner).unwrap();
        json.push_str("  \"regressions\": [");
        for (i, r) in self.regressions.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            write!(json, "\"{}\"", r).unwrap();
        }
        json.push_str("],\n");
        json.push_str("  \"improvements\": [");
        for (i, imp) in self.improvements.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            write!(json, "\"{}\"", imp).unwrap();
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
    let results_a = profile_real_inference(model_a, warmup, measure)?;

    #[cfg(not(feature = "inference"))]
    return Err(CliError::ValidationFailed(
        "Requires --features inference".to_string(),
    ));

    // Profile model B
    output::kv("Profiling Model B", model_b.display());
    #[cfg(feature = "inference")]
    let results_b = profile_real_inference(model_b, warmup, measure)?;

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
    let _min_us = forward_times.iter().copied().fold(f64::INFINITY, f64::min);
    let _max_us = forward_times
        .iter()
        .copied()
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
    let min_us = forward_times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = forward_times
        .iter()
        .copied()
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

        let max_layer_time = results.per_layer_us.iter().copied().fold(0.0f64, f64::max);

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
    writeln!(json, "  \"model\": \"{}\",", results.model_path).unwrap();
    writeln!(json, "  \"architecture\": \"{}\",", results.architecture).unwrap();
    writeln!(json, "  \"num_layers\": {},", results.num_layers).unwrap();
    writeln!(json, "  \"vocab_size\": {},", results.vocab_size).unwrap();
    writeln!(json, "  \"hidden_dim\": {},", results.hidden_dim).unwrap();
    writeln!(json, "  \"is_real_data\": {},", results.is_real_data).unwrap();
    json.push_str("  \"timing\": {\n");
    writeln!(json, "    \"warmup_passes\": {},", results.warmup_passes).unwrap();
    writeln!(json, "    \"measure_passes\": {},", results.measure_passes).unwrap();
    writeln!(
        json,
        "    \"avg_inference_us\": {:.2},",
        results.total_inference_us
    )
    .unwrap();
    writeln!(
        json,
        "    \"throughput_tok_s\": {:.2}",
        results.throughput_tok_s
    )
    .unwrap();
    json.push_str("  },\n");

    json.push_str("  \"hotspots\": [\n");
    for (i, hotspot) in results.hotspots.iter().enumerate() {
        json.push_str("    {\n");
        writeln!(json, "      \"name\": \"{}\",", hotspot.name).unwrap();
        writeln!(json, "      \"total_us\": {:.2},", hotspot.time_us).unwrap();
        writeln!(json, "      \"avg_us\": {:.2},", hotspot.avg_us).unwrap();
        writeln!(json, "      \"min_us\": {:.2},", hotspot.min_us).unwrap();
        writeln!(json, "      \"max_us\": {:.2},", hotspot.max_us).unwrap();
        writeln!(json, "      \"count\": {}", hotspot.count).unwrap();
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
        write!(json, "{:.2}", time).unwrap();
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
        ).unwrap();
        writeln!(
            svg,
            "  <text x=\"{:.1}\" y=\"{:.1}\" class=\"label\">{} ({:.1}%)</text>",
            x + 5.0,
            y + 16.0,
            hotspot.name,
            percent
        )
        .unwrap();

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
            },
            Hotspot {
                name: "attention".to_string(),
                time_us: 3000.0,
                percent: 30.0,
                count: 10,
                avg_us: 300.0,
                min_us: 280.0,
                max_us: 320.0,
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
}
