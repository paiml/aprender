//! QA Command Implementation - Falsifiable Quality Assurance Checklist
//!
//! Implements a scientific QA process for model releases. Every claim must be
//! falsifiable - if a test can't fail, it doesn't provide information.
//!
//! # Gates
//!
//! 1. **Golden Output Test** (Correctness Gate)
//!    - Run model with known prompts, verify expected patterns in output
//!    - Falsifiable: Output must match expected pattern or test fails
//!
//! 2. **Throughput Falsification** (Performance Gate)
//!    - Run benchmark with statistical rigor (CV < 5%)
//!    - Assert minimum tok/s threshold
//!    - Falsifiable: If tok/s < threshold, test fails
//!
//! 3. **Ollama Parity Test** (Parity Gate)
//!    - Compare against Ollama baseline (if available)
//!    - Assert speedup factor >= target
//!    - Falsifiable: If speedup < target, test fails
//!
//! 4. **GPU vs CPU Speedup Test** (F-PERF-042)
//!    - Measure throughput on both GPU and CPU
//!    - Assert GPU >= 2x CPU (default threshold)
//!    - Falsifiable: If GPU speedup < threshold, test fails
//!    - Toyota Way: Genchi Genbutsu - measure real performance
//!
//! 5. **Cross-Format Parity Test** (F-QUAL-032)
//!    - Compare argmax between GGUF and SafeTensors for same model
//!    - Invariant: argmax(forward_gguf) == argmax(forward_safetensors)
//!    - Falsifiable: If argmax differs, cross-format parity is BROKEN
//!    - Cornerstone of architecture's logical validity
//!
//! 6. **PTX Parity Test** (GH-219, F-PTX-001)
//!    - Validate batched GPU kernels maintain structural parity with single-vector references
//!    - Checks: batch dispatch mechanism, u64 shared memory addressing, dispatch strategy
//!    - Falsifiable: If any of 6 kernel pairs fails structural validation, test fails
//!    - Toyota Way: Poka-Yoke - error-proof PTX generation at compile time
//!
//! # Usage
//!
//! ```bash
//! apr qa model.gguf                           # Run all gates
//! apr qa model.gguf --assert-tps 100          # Custom throughput threshold
//! apr qa model.gguf --assert-speedup 2.0      # Custom Ollama speedup
//! apr qa model.gguf --assert-gpu-speedup 3.0  # Custom GPU vs CPU speedup
//! apr qa model.gguf --skip-ollama             # Skip Ollama comparison
//! apr qa model.gguf --skip-gpu-speedup        # Skip GPU vs CPU test
//! apr qa model.gguf --skip-format-parity      # Skip cross-format test
//! apr qa model.gguf --safetensors-path m.st   # Compare with SafeTensors model
//! apr qa model.gguf --json                    # JSON output for CI
//! ```
//!
//! # Exit Codes
//!
//! - 0: All gates passed
//! - 5: One or more gates failed (ValidationFailed)
//!
//! Toyota Way: Jidoka - Stop and fix quality issues immediately.
//! Scientific Method: Claims must be falsifiable to have meaning.

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

/// QA configuration
#[derive(Debug, Clone)]
pub struct QaConfig {
    /// Minimum throughput in tok/s (default: 100 for GPU, 10 for CPU)
    pub min_tps: f64,
    /// Minimum speedup vs Ollama (default: 2.0x)
    pub min_speedup: f64,
    /// Minimum GPU vs CPU speedup (default: 2.0x) - F-PERF-042
    pub min_gpu_speedup: f64,
    /// Skip golden output test
    pub skip_golden: bool,
    /// Skip throughput test
    pub skip_throughput: bool,
    /// Skip Ollama parity test
    pub skip_ollama: bool,
    /// Skip GPU vs CPU speedup test (F-PERF-042)
    pub skip_gpu_speedup: bool,
    /// Skip tensor contract validation (PMAT-235)
    pub skip_contract: bool,
    /// Skip cross-format parity test (F-QUAL-032)
    pub skip_format_parity: bool,
    /// Skip PTX parity validation (GH-219, F-PTX-001)
    pub skip_ptx_parity: bool,
    /// SafeTensors model path for cross-format parity (F-QUAL-032)
    pub safetensors_path: Option<std::path::PathBuf>,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Number of warmup iterations
    pub warmup: usize,
    /// Max tokens for generation
    pub max_tokens: usize,
    /// Output as JSON
    pub json: bool,
    /// Verbose output
    pub verbose: bool,
    /// Minimum number of gates that must execute (not be skipped)
    pub min_executed: Option<usize>,
    /// Path to previous QA report for regression comparison
    pub previous_report: Option<std::path::PathBuf>,
    /// Maximum allowed performance regression (0.10 = 10%)
    pub regression_threshold: f64,
    /// Skip GPU state isolation test
    pub skip_gpu_state: bool,
    /// Skip metadata plausibility validation (Bug 210, GH-222)
    pub skip_metadata: bool,
    /// Skip GPU capability match gate (GH-280)
    pub skip_capability: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            min_tps: 100.0,       // GPU target
            min_speedup: 0.2, // Ollama uses llama.cpp optimized kernels; 0.2x is realistic floor
            min_gpu_speedup: 2.0, // GPU must be 2x faster than CPU (F-PERF-042)
            skip_golden: false,
            skip_throughput: false,
            skip_ollama: false,
            skip_gpu_speedup: false,
            skip_contract: false,
            skip_format_parity: false,
            skip_ptx_parity: false,
            safetensors_path: None,
            iterations: 10,
            warmup: 3,
            max_tokens: 32,
            json: false,
            verbose: false,
            min_executed: None,
            previous_report: None,
            regression_threshold: 0.10,
            skip_gpu_state: false,
            skip_metadata: false,
            skip_capability: false,
        }
    }
}

/// Result of a single QA gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Gate name
    pub name: String,
    /// Whether the gate passed
    pub passed: bool,
    /// Human-readable result message
    pub message: String,
    /// Measured value (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Expected/threshold value (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Time taken to run the gate
    pub duration_ms: u64,
    /// Whether the gate was skipped
    pub skipped: bool,
}

impl GateResult {
    pub(crate) fn passed(
        name: &str,
        message: &str,
        value: Option<f64>,
        threshold: Option<f64>,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            message: message.to_string(),
            value,
            threshold,
            duration_ms: duration.as_millis() as u64,
            skipped: false,
        }
    }

    pub(crate) fn failed(
        name: &str,
        message: &str,
        value: Option<f64>,
        threshold: Option<f64>,
        duration: Duration,
    ) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            message: message.to_string(),
            value,
            threshold,
            duration_ms: duration.as_millis() as u64,
            skipped: false,
        }
    }

    fn skipped(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true, // Skipped gates don't fail
            message: format!("Skipped: {reason}"),
            value: None,
            threshold: None,
            duration_ms: 0,
            skipped: true,
        }
    }
}

/// System information captured during QA run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// CPU model name
    pub cpu_model: String,
    /// GPU model name (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_model: Option<String>,
    /// GPU driver version (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_driver: Option<String>,
}

impl SystemInfo {
    fn capture() -> Self {
        let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string());

        let (gpu_model, gpu_driver) = Self::detect_gpu();

        Self {
            cpu_model,
            gpu_model,
            gpu_driver,
        }
    }

    fn detect_gpu() -> (Option<String>, Option<String>) {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
            .output()
            .ok();
        if let Some(out) = output {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                let parts: Vec<&str> = text.trim().splitn(2, ',').collect();
                return (
                    parts.first().map(|s| s.trim().to_string()),
                    parts.get(1).map(|s| s.trim().to_string()),
                );
            }
        }
        (None, None)
    }
}

/// Full QA report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaReport {
    /// Model path
    pub model: String,
    /// Whether all gates passed
    pub passed: bool,
    /// Individual gate results
    pub gates: Vec<GateResult>,
    /// Number of gates that actually executed (not skipped)
    #[serde(default)]
    pub gates_executed: usize,
    /// Number of gates that were skipped
    #[serde(default)]
    pub gates_skipped: usize,
    /// Total duration
    pub total_duration_ms: u64,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// Summary message
    pub summary: String,
    /// System information
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_info: Option<SystemInfo>,
}

/// Run the QA command
#[allow(clippy::too_many_arguments)]
pub fn run(
    path: &Path,
    min_tps: Option<f64>,
    min_speedup: Option<f64>,
    min_gpu_speedup: Option<f64>,
    skip_golden: bool,
    skip_throughput: bool,
    skip_ollama: bool,
    skip_gpu_speedup: bool,
    skip_contract: bool,
    skip_format_parity: bool,
    skip_ptx_parity: bool,
    safetensors_path: Option<std::path::PathBuf>,
    iterations: usize,
    warmup: usize,
    max_tokens: usize,
    json: bool,
    verbose: bool,
    min_executed: Option<usize>,
    previous_report: Option<std::path::PathBuf>,
    regression_threshold: Option<f64>,
    skip_gpu_state: bool,
    skip_metadata: bool,
    skip_capability: bool,
) -> Result<()> {
    let config = QaConfig {
        min_tps: min_tps.unwrap_or(100.0),
        min_speedup: min_speedup.unwrap_or(0.2), // Ollama uses llama.cpp optimized kernels
        min_gpu_speedup: min_gpu_speedup.unwrap_or(2.0), // GPU must be 2x faster (F-PERF-042)
        skip_golden,
        skip_throughput,
        skip_ollama,
        skip_gpu_speedup,
        skip_contract,
        skip_format_parity,
        skip_ptx_parity,
        safetensors_path,
        iterations,
        warmup,
        max_tokens,
        json,
        verbose,
        min_executed,
        previous_report,
        regression_threshold: regression_threshold.unwrap_or(0.10),
        skip_gpu_state,
        skip_metadata,
        skip_capability,
    };

    let report = run_qa(path, &config)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
    }

    if !report.passed {
        return Err(CliError::ValidationFailed(report.summary));
    }

    Ok(())
}

/// Dispatch a single QA gate: skip if flagged, otherwise run, then print and collect.
fn dispatch_gate(
    gates: &mut Vec<GateResult>,
    json: bool,
    skip: bool,
    name: &str,
    skip_reason: &str,
    runner: impl FnOnce() -> Result<GateResult>,
) -> Result<()> {
    let result = if skip {
        GateResult::skipped(name, skip_reason)
    } else {
        runner()?
    };
    if !json {
        print_gate_result(&result);
    }
    gates.push(result);
    Ok(())
}

/// Run all QA gates and produce a report
/// Human-readable gate name for display.
fn gate_display_name(name: &str) -> &str {
    match name {
        "capability_match" => "Capability Match",
        "tensor_contract" => "Tensor Contract",
        "golden_output" => "Golden Output",
        "throughput" => "Throughput",
        "ollama_parity" => "Ollama Parity",
        "gpu_speedup" => "GPU Speedup",
        "format_parity" => "Format Parity",
        "ptx_parity" => "PTX Parity",
        "gpu_state_isolation" => "GPU State Isolation",
        "performance_regression" => "Perf Regression",
        "metadata_plausibility" => "Metadata Plausibility",
        other => other,
    }
}

/// Print the QA summary table and pass/fail badges.
fn print_qa_summary(gates: &[GateResult], passed: bool, total_duration: Duration) {
    output::header("QA Summary");

    let gate_rows: Vec<Vec<String>> = gates
        .iter()
        .map(|g| {
            let badge = if g.skipped {
                output::badge_skip("SKIP")
            } else if g.passed {
                output::badge_pass("PASS")
            } else {
                output::badge_fail("FAIL")
            };
            let measured = g.value.map_or("—".to_string(), |v| format!("{v:.2}"));
            let threshold = g.threshold.map_or("—".to_string(), |v| format!("{v:.2}"));
            vec![
                gate_display_name(&g.name).to_string(),
                badge,
                measured,
                threshold,
                output::duration_fmt(g.duration_ms),
            ]
        })
        .collect();
    println!(
        "{}",
        output::table(
            &["Gate", "Status", "Measured", "Threshold", "Duration"],
            &gate_rows,
        )
    );

    println!();
    if passed {
        println!("  {}", output::badge_pass("ALL GATES PASSED"));
    } else {
        println!("  {}", output::badge_fail("GATES FAILED"));
        for gate in gates.iter().filter(|g| !g.passed && !g.skipped) {
            println!("    {} {}", "✗".red(), gate.name);
        }
    }
    output::metric(
        "Total Duration",
        output::duration_fmt(total_duration.as_millis() as u64),
        "",
    );
}

include!("qa_gguf.rs");
include!("output_verification.rs");
include!("golden_output.rs");
include!("speedup.rs");
include!("forward_error.rs");
include!("gpu_isolation_result.rs");
include!("qa_08.rs");
