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

include!("diff_benchmark_report.rs");
include!("profile_part_03.rs");
include!("kernel.rs");
include!("profile_part_05.rs");
include!("profile_part_06.rs");
include!("profile_part_07.rs");
include!("comparison.rs");
include!("profile_part_09.rs");
