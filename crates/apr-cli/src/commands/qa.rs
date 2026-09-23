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

/// QA configuration
#[derive(Debug, Clone)]
pub struct QaConfig {
    /// Throughput floor asserted by the user via `--assert-tps`, in tok/s.
    ///
    /// `None` means the user asserted nothing, and the throughput gate picks a
    /// format-aware default instead (see `speedup.rs`). This has to stay an
    /// `Option`: collapsing it to a plain `f64` with a default is what let the
    /// gate confuse "the user demanded 100 tok/s" with "nobody said anything",
    /// and then quietly substitute its own much lower number for both.
    pub min_tps: Option<f64>,
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
    /// Assert classifier head presence and shape (F-CLASS-004)
    pub assert_classifier_head: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            min_tps: None,        // no assertion; the gate picks a format-aware default
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
            assert_classifier_head: false,
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

    /// A gate that did not run. #3965: it is NOT `passed`. `passed: true` on an unrun
    /// check told every JSON consumer that read `passed` alone that the check had
    /// passed: `gpu_speedup` "Skipped: CUDA not available" read as a GPU pass. A skip
    /// is Unknown(NotRun), never a pass. Whether a skip fails the RUN is a separate
    /// rule, and it lives in one place: [`gates_pass`].
    pub(crate) fn skipped(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            message: format!("Skipped: {reason}"),
            value: None,
            threshold: None,
            duration_ms: 0,
            skipped: true,
        }
    }
}

/// #3965: the gate registry: every gate `run_qa` dispatches, by the name it reports.
///
/// `run_qa` checks the gates it actually emitted against this on every run, and a
/// mismatch is a FAILED `gate_registry` gate, not a warning, so the list cannot
/// drift from the pipeline silently. It is published in the report as
/// `gates_registered`, so a JSON consumer derives what to expect from the binary
/// it ran instead of from a number it once counted.
pub(crate) const QA_GATES: [&str; 12] = [
    "capability_match",
    "tensor_contract",
    "metadata_plausibility",
    "classifier_head",
    "golden_output",
    "throughput",
    "ollama_parity",
    "gpu_speedup",
    "format_parity",
    "ptx_parity",
    "gpu_state_isolation",
    "performance_regression",
];

/// `None` when the emitted gates are exactly [`QA_GATES`] (order-free, each once),
/// otherwise the reason.
#[must_use]
pub(crate) fn gate_registry_mismatch(gates: &[GateResult]) -> Option<String> {
    let mut emitted: Vec<&str> = gates.iter().map(|g| g.name.as_str()).collect();
    emitted.sort_unstable();
    let mut want: Vec<&str> = QA_GATES.to_vec();
    want.sort_unstable();
    if emitted == want {
        return None;
    }
    let missing: Vec<&str> = want
        .iter()
        .copied()
        .filter(|w| !emitted.contains(w))
        .collect();
    let extra: Vec<&str> = emitted
        .iter()
        .copied()
        .filter(|e| !want.contains(e))
        .collect();
    Some(format!(
        "emitted gates do not match the registry: missing {missing:?}, unregistered or repeated {extra:?}"
    ))
}

/// #3965: the RUN verdict over a set of gates, in ONE place.
///
/// Every executed gate must pass. A skipped gate does not fail the run: that rule is
/// unchanged, and `check_min_executed` still bounds how many may skip. What changed
/// is that a skip no longer claims `passed` at the gate level, so this rule has to
/// say `skipped` explicitly. Before, it was hidden inside `all(|g| g.passed)`.
/// Several tests re-implemented that expression inline instead of calling
/// production; they now call this, so they test the code that ships.
#[must_use]
pub(crate) fn gates_pass(gates: &[GateResult]) -> bool {
    gates.iter().all(|g| g.passed || g.skipped)
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
    /// #3965 / SHIP-006: every gate this binary runs, from [`QA_GATES`]. A consumer
    /// compares `gates` against THIS instead of a count it hardcoded. SHIP-006
    /// required exactly 8 while apr qa emitted 12, so it could never go green again.
    /// A report from a binary predating the registry deserializes to an empty list,
    /// which consumers must treat as "cannot check", never as "nothing required".
    #[serde(default)]
    pub gates_registered: Vec<String>,
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
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
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
    assert_classifier_head: bool,
) -> Result<()> {
    contract_pre_qa_gate_composition!();
    // GH-2391: every QA gate is `observed >= asserted`. A NaN or negative
    // assertion makes that comparison unable to distinguish pass from fail, so
    // the release gate reports a verdict it never reached. Refuse the value.
    use crate::commands::threshold_arg;
    threshold_arg::guard_opt("--assert-tps", min_tps, threshold_arg::TOLERANCE)?;
    threshold_arg::guard_opt("--assert-speedup", min_speedup, threshold_arg::TOLERANCE)?;
    threshold_arg::guard_opt(
        "--assert-gpu-speedup",
        min_gpu_speedup,
        threshold_arg::TOLERANCE,
    )?;
    threshold_arg::guard_opt(
        "--regression-threshold",
        regression_threshold,
        threshold_arg::FRACTION,
    )?;

    let config = QaConfig {
        min_tps,
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
        assert_classifier_head,
    };

    // `run_qa(...)?` used to propagate straight past the `if json` block below, so
    // ANY error meant `apr qa --json` exited having written ZERO BYTES. Measured
    // 2026-09-22 on gx10 (#3842): on qwen35-27b-q4km it ran 78s of GPU work and
    // its own stderr says `F2 guard: passed in 78442 ms on 20 positions` — then
    // wrote nothing. `model_ladder.sh` appended the empty result as an empty row,
    // the receipt assembler dropped it, and a RED REQUIRED RUNG disappeared from
    // its own receipt while the red counter went on counting: the receipt said
    // `red: 3` with two reds in its rows.
    //
    // A gate that produces NO DOCUMENT is strictly worse than one that fails: a
    // failure is evidence, an absence is not, and no per-row judging can find a
    // row that was never written. So the document is emitted on BOTH paths and
    // the error is still returned, preserving the exit code.
    let report = match run_qa(path, &config) {
        Ok(report) => report,
        Err(e) => {
            if json {
                emit_qa_json(&qa_report_for_error(path, &e));
            }
            return Err(e);
        }
    };

    if json {
        emit_qa_json(&report);
    }

    if !report.passed {
        return Err(CliError::ValidationFailed(report.summary));
    }

    contract_post_qa_gate_composition!(&());
    Ok(())
}

/// Minimal JSON string escaper, so the last-resort document cannot itself fail.
///
/// Deliberately not `serde_json` — the fallback below exists precisely for the
/// case where serialization did not work, and reaching for the thing that just
/// failed is how a fallback becomes decoration.
fn json_escaped(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Emit the report as JSON. NEVER emits an empty document.
///
/// The old call site was `serde_json::to_string_pretty(&report).unwrap_or_default()`,
/// which turns a serialization failure into an empty String — a bare newline on
/// stdout. That is the same zero-byte outcome as printing nothing, with the extra
/// property that it looks like output, so a reader sees a truncated file rather
/// than a missing one (#3842). A serializer that cannot describe the report is
/// itself a finding and is reported as one.
pub(crate) fn emit_qa_json(report: &QaReport) {
    println!("{}", qa_json_document(report));
}

/// Build the `--json` document. Pure, and the return value is NEVER empty.
///
/// Separated from the printing so the invariant is a value a test can assert on.
/// The defect this replaces was covered by a test called `test_run_with_json_output`
/// which drove `--json` down the error path and asserted only `result.is_err()` —
/// the test named for the output never looked at the output. A pure function makes
/// "never empty" checkable without capturing stdout.
pub(crate) fn qa_json_document(report: &QaReport) -> String {
    match serde_json::to_string_pretty(report) {
        Ok(s) if !s.trim().is_empty() => s,
        other => {
            let why = match other {
                Ok(_) => "the serializer produced an empty document".to_string(),
                Err(e) => e.to_string(),
            };
            format!(
                "{{{}:{},{}:false,{}:[],{}:0,{}:0,{}:0,{}:{},{}:{}}}",
                json_escaped("model"),
                json_escaped(&report.model),
                json_escaped("passed"),
                json_escaped("gates"),
                json_escaped("gates_executed"),
                json_escaped("gates_skipped"),
                json_escaped("total_duration_ms"),
                json_escaped("timestamp"),
                json_escaped(&report.timestamp),
                json_escaped("summary"),
                json_escaped(&format!("apr qa could not serialize its report: {why}")),
            )
        }
    }
}

/// The report `apr qa --json` emits when the run itself could not complete.
///
/// It carries a FAILED gate rather than an empty `gates` list, because an empty
/// list reads as "nothing failed" to anything counting failures — the same
/// ambiguity that let a missing row look like an absent model instead of a red
/// one (#3842).
pub(crate) fn qa_report_for_error(path: &Path, e: &CliError) -> QaReport {
    let why = format!("apr qa did not complete: {e}");
    QaReport {
        model: path.display().to_string(),
        passed: false,
        gates: vec![GateResult::failed(
            "qa_run",
            &why,
            None,
            None,
            Duration::ZERO,
        )],
        gates_executed: 0,
        gates_skipped: 0,
        gates_registered: Vec::new(),
        total_duration_ms: 0,
        timestamp: chrono::Utc::now().to_rfc3339(),
        summary: why,
        system_info: None,
    }
}

/// Dispatch a single QA gate: skip if flagged, otherwise run, then print and collect.
///
/// A gate whose runner returns `Err` is recorded as a FAILED row carrying the
/// error, and the remaining gates still run (#3714 done_when 3). It used to be
/// `runner()?`, which abandoned `run_qa` before the report existed: on a
/// qwen3moe file the golden gate's dense CPU path errored and `apr qa --json`
/// printed ZERO bytes and exited 5 — every gate that had already passed, and
/// the one that failed, were lost. An error is a FAIL, never a skip and never
/// a pass; the report is always written.
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
        // #3817: a gate that cannot RUN is a FAILED gate, never an aborted
        // report. `apr qa` on a qwen3moe GGUF used to propagate the dense
        // loader's error out of `run_qa`, so the process exited 5 having printed
        // **zero bytes of JSON** — `capability_match` and `golden_output` were
        // absent rather than red, and absence reads as conformance to anything
        // parsing the report. The error is now the gate's message. (#3714 R2
        // fixed the same abort independently; the fold keeps this message and
        // takes its measured duration rather than a zero.)
        let start = Instant::now();
        match runner() {
            Ok(result) => result,
            Err(e) => GateResult::failed(
                name,
                &format!("{name} could not run: {e}"),
                None,
                None,
                start.elapsed(),
            ),
        }
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
        "classifier_head" => "Classifier Head",
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
include!("qa_json_never_empty.rs");
