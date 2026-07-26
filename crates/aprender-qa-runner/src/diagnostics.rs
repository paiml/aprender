//! Fail-Fast Diagnostic Report Generation (FF-REPORT-001)
//!
//! Generates comprehensive diagnostic reports on test failure using apr's rich tooling.
//! Reports are designed for immediate GitHub issue creation with full reproduction context.

use crate::evidence::Evidence;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Timeout for the `apr check` diagnostic command
const CHECK_TIMEOUT: Duration = Duration::from_secs(30);
/// Timeout for the `apr inspect` diagnostic command
const INSPECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Timeout for the `apr trace` diagnostic command
const TRACE_TIMEOUT: Duration = Duration::from_secs(60);
/// Timeout for the `apr tensors` diagnostic command
const TENSORS_TIMEOUT: Duration = Duration::from_secs(10);
/// Timeout for the `apr explain` diagnostic command
const EXPLAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of a diagnostic command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// Command that was run
    pub command: String,
    /// Whether the command succeeded
    pub success: bool,
    /// Stdout output
    pub stdout: String,
    /// Stderr output
    pub stderr: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether the command timed out
    pub timed_out: bool,
}

/// Environment context for the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    /// Operating system (e.g., "linux", "macos", "windows")
    pub os: String,
    /// CPU architecture (e.g., "x86_64", "aarch64")
    pub arch: String,
    /// apr-qa version
    pub aprender_qa_version: String,
    /// apr CLI version
    pub apr_cli_version: String,
    /// Git commit hash (short form)
    pub git_commit: String,
    /// Git branch name
    pub git_branch: String,
    /// Whether working directory has uncommitted changes
    pub git_dirty: bool,
    /// Rust compiler version
    pub rustc_version: String,
}

/// Collection and construction for environment context
impl EnvironmentContext {
    /// Collect environment context
    #[must_use]
    pub fn collect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            aprender_qa_version: env!("CARGO_PKG_VERSION").to_string(),
            apr_cli_version: get_apr_version(),
            git_commit: get_git_commit(),
            git_branch: get_git_branch(),
            git_dirty: get_git_dirty(),
            rustc_version: get_rustc_version(),
        }
    }
}

/// Failure details from the evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetails {
    /// Gate ID that failed (e.g., "G3-STABLE")
    pub gate_id: String,
    /// Model identifier (HuggingFace repo path)
    pub model: String,
    /// Model format (e.g., "Apr", "SafeTensors", "Gguf")
    pub format: String,
    /// Backend used (e.g., "Cpu", "Metal", "Cuda")
    pub backend: String,
    /// Test outcome (e.g., "Crashed", "Falsified", "Timeout")
    pub outcome: String,
    /// Human-readable failure reason
    pub reason: String,
    /// Process exit code if available
    pub exit_code: Option<i32>,
    /// Test duration in milliseconds
    pub duration_ms: u64,
    /// Standard error output if captured
    pub stderr: Option<String>,
}

/// Convert Evidence into FailureDetails for diagnostic reports
impl From<&Evidence> for FailureDetails {
    fn from(evidence: &Evidence) -> Self {
        Self {
            gate_id: evidence.gate_id.clone(),
            model: evidence.scenario.model.hf_repo(),
            format: format!("{:?}", evidence.scenario.format),
            backend: format!("{:?}", evidence.scenario.backend),
            outcome: format!("{:?}", evidence.outcome),
            reason: evidence.reason.clone(),
            exit_code: evidence.exit_code,
            duration_ms: evidence.metrics.duration_ms,
            stderr: evidence.stderr.clone(),
        }
    }
}

/// Complete fail-fast diagnostic report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailFastReport {
    /// Report version
    pub version: String,
    /// Timestamp
    pub timestamp: String,
    /// Failure details
    pub failure: FailureDetails,
    /// Environment context
    pub environment: EnvironmentContext,
    /// Diagnostic results
    pub diagnostics: DiagnosticsBundle,
    /// Reproduction information
    pub reproduction: ReproductionInfo,
}

/// Bundle of all diagnostic results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsBundle {
    /// Results from `apr check` - pipeline integrity
    pub check: Option<DiagnosticResult>,
    /// Results from `apr inspect` - model metadata
    pub inspect: Option<DiagnosticResult>,
    /// Results from `apr trace` - layer-by-layer analysis
    pub trace: Option<DiagnosticResult>,
    /// Results from `apr tensors` - tensor names and shapes
    pub tensors: Option<DiagnosticResult>,
    /// Results from `apr explain` - error code explanation
    pub explain: Option<DiagnosticResult>,
}

/// Information for reproducing the failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproductionInfo {
    /// Command to reproduce the failure
    pub command: String,
    /// Path to the model file used
    pub model_path: String,
    /// Path to the playbook file if applicable
    pub playbook: Option<String>,
}

/// Fail-fast diagnostic reporter
pub struct FailFastReporter {
    output_dir: PathBuf,
    binary: String,
}

/// Report generation and diagnostic command execution
impl FailFastReporter {
    /// Create a new reporter
    #[must_use]
    pub fn new(output_dir: &Path) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
            binary: "apr".to_string(),
        }
    }

    /// Create with custom binary path
    #[must_use]
    pub fn with_binary(mut self, binary: &str) -> Self {
        self.binary = binary.to_string();
        self
    }

    /// Generate full diagnostic report on failure
    ///
    /// # Errors
    ///
    /// Returns an error if report generation fails.
    pub fn generate_report(
        &self,
        evidence: &Evidence,
        model_path: &Path,
        playbook: Option<&str>,
    ) -> std::io::Result<FailFastReport> {
        let report_dir = self.output_dir.join("fail-fast-report");
        std::fs::create_dir_all(&report_dir)?;

        eprintln!("[FAIL-FAST] Generating diagnostic report...");

        // Collect diagnostics
        let check = self.run_check(model_path);
        let inspect = self.run_inspect(model_path);
        let trace = self.run_trace(model_path);
        let tensors = self.run_tensors(model_path);
        let explain = self.run_explain(&evidence.gate_id);

        // Save individual diagnostic files first (before moving into report)
        if let Some(ref c) = check {
            self.save_json(&report_dir.join("check.json"), c)?;
        }
        if let Some(ref i) = inspect {
            self.save_json(&report_dir.join("inspect.json"), i)?;
        }
        if let Some(ref t) = trace {
            self.save_json(&report_dir.join("trace.json"), t)?;
        }
        if let Some(ref t) = tensors {
            self.save_json(&report_dir.join("tensors.json"), t)?;
        }

        // Build report (moves diagnostic values)
        let report = FailFastReport {
            version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            failure: FailureDetails::from(evidence),
            environment: EnvironmentContext::collect(),
            diagnostics: DiagnosticsBundle {
                check,
                inspect,
                trace,
                tensors,
                explain,
            },
            reproduction: ReproductionInfo {
                command: format!(
                    "apr-qa run {} --fail-fast",
                    playbook.unwrap_or("playbook.yaml")
                ),
                model_path: model_path.to_string_lossy().to_string(),
                playbook: playbook.map(String::from),
            },
        };

        // Save full diagnostics JSON
        self.save_json(&report_dir.join("diagnostics.json"), &report)?;

        // Save environment
        self.save_json(&report_dir.join("environment.json"), &report.environment)?;

        // Save stderr log
        if let Some(ref stderr) = evidence.stderr {
            std::fs::write(report_dir.join("stderr.log"), stderr)?;
        }

        // Generate markdown summary
        let summary = self.generate_markdown(&report);
        std::fs::write(report_dir.join("summary.md"), &summary)?;

        eprintln!("[FAIL-FAST] Report saved to: {}", report_dir.display());
        eprintln!("[FAIL-FAST] Summary: {}/summary.md", report_dir.display());
        eprintln!("[FAIL-FAST] GitHub issue body ready for paste");

        Ok(report)
    }

    /// Run apr check and capture output
    fn run_check(&self, model_path: &Path) -> Option<DiagnosticResult> {
        eprint!("[FAIL-FAST] Running apr check... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "check",
                &model_path.to_string_lossy(),
                "--json",
            ],
            CHECK_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr inspect and capture output
    fn run_inspect(&self, model_path: &Path) -> Option<DiagnosticResult> {
        eprint!("[FAIL-FAST] Running apr inspect... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "inspect",
                &model_path.to_string_lossy(),
                "--json",
            ],
            INSPECT_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr trace and capture output
    fn run_trace(&self, model_path: &Path) -> Option<DiagnosticResult> {
        // Only run trace for .apr files
        if model_path.extension().is_none_or(|e| e != "apr") {
            return None;
        }

        eprint!("[FAIL-FAST] Running apr trace... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "trace",
                &model_path.to_string_lossy(),
                "--payload",
                "--json",
            ],
            TRACE_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr tensors and capture output
    fn run_tensors(&self, model_path: &Path) -> Option<DiagnosticResult> {
        eprint!("[FAIL-FAST] Running apr tensors... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "tensors",
                &model_path.to_string_lossy(),
                "--json",
            ],
            TENSORS_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr explain for the error code
    fn run_explain(&self, error_code: &str) -> Option<DiagnosticResult> {
        // Extract error code pattern (e.g., "G3-STABLE" -> try explaining common errors)
        eprint!("[FAIL-FAST] Running apr explain... ");
        let result =
            self.run_command_with_timeout(&[&self.binary, "explain", error_code], EXPLAIN_TIMEOUT);
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run an external command with enforced timeout.
    ///
    /// Spawns the process and polls `try_wait` with 100ms intervals. If the
    /// process exceeds `timeout`, it is killed and reaped — the caller gets
    /// `timed_out = true` instead of blocking indefinitely.
    fn run_command_with_timeout(&self, args: &[&str], timeout: Duration) -> DiagnosticResult {
        use std::io::Read;
        use std::process::Stdio;

        let start = Instant::now();
        let command_str = args.join(" ");

        let mut child = match Command::new(args[0])
            .args(&args[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return DiagnosticResult {
                    command: command_str,
                    success: false,
                    stdout: String::new(),
                    stderr: format!("Failed to execute: {e}"),
                    duration_ms: 0,
                    timed_out: false,
                };
            }
        };

        let poll_interval = Duration::from_millis(100);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    // Process exited — read output from pipes
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(ref mut out) = child.stdout {
                        let _ = out.read_to_string(&mut stdout);
                    }
                    if let Some(ref mut err) = child.stderr {
                        let _ = err.read_to_string(&mut stderr);
                    }
                    return DiagnosticResult {
                        command: command_str,
                        success: status.success(),
                        stdout,
                        stderr,
                        duration_ms: start.elapsed().as_millis() as u64,
                        timed_out: false,
                    };
                }
                Ok(None) => {
                    // Still running — check timeout
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        let _ = child.wait(); // reap zombie
                        return DiagnosticResult {
                            command: command_str,
                            success: false,
                            stdout: String::new(),
                            stderr: format!(
                                "Process killed after {}ms timeout",
                                timeout.as_millis()
                            ),
                            duration_ms: start.elapsed().as_millis() as u64,
                            timed_out: true,
                        };
                    }
                    std::thread::sleep(poll_interval);
                }
                Err(e) => {
                    return DiagnosticResult {
                        command: command_str,
                        success: false,
                        stdout: String::new(),
                        stderr: format!("Error waiting for process: {e}"),
                        duration_ms: start.elapsed().as_millis() as u64,
                        timed_out: false,
                    };
                }
            }
        }
    }

    /// Serialize data to pretty-printed JSON and write to a file
    fn save_json<T: Serialize>(&self, path: &Path, data: &T) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(data).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }
}

include!("diagnostics_markdown_gen.rs");
