//! Command execution abstraction for testability
//!
//! This module provides a trait-based abstraction over subprocess execution,
//! allowing the executor code to be tested with mock implementations.

use std::path::Path;

/// Result of executing a command
#[derive(Debug, Clone)]
pub struct CommandOutput {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code (negative for signals)
    pub exit_code: i32,
    /// Whether the command succeeded
    pub success: bool,
}

impl CommandOutput {
    /// Create a successful command output
    #[must_use]
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: String::new(),
            exit_code: 0,
            success: true,
        }
    }

    /// Create a failed command output
    #[must_use]
    pub fn failure(exit_code: i32, stderr: impl Into<String>) -> Self {
        Self {
            stdout: String::new(),
            stderr: stderr.into(),
            exit_code,
            success: false,
        }
    }

    /// Create output with both stdout and stderr
    #[must_use]
    pub fn with_output(
        stdout: impl Into<String>,
        stderr: impl Into<String>,
        exit_code: i32,
    ) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: stderr.into(),
            exit_code,
            success: exit_code == 0,
        }
    }
}

/// Trait for executing shell commands
///
/// This abstraction allows for mocking subprocess execution in tests.
pub trait CommandRunner: Send + Sync {
    /// Execute an apr run command
    fn run_inference(
        &self,
        model_path: &Path,
        prompt: &str,
        max_tokens: u32,
        no_gpu: bool,
        extra_args: &[&str],
    ) -> CommandOutput;

    /// Execute an apr convert command
    fn convert_model(&self, source: &Path, target: &Path) -> CommandOutput;

    /// Execute an apr rosetta inspect command
    fn inspect_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr validate command
    fn validate_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr bench command
    fn bench_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr check command
    fn check_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr profile command
    fn profile_model(&self, model_path: &Path, warmup: u32, measure: u32) -> CommandOutput;

    /// Execute apr profile in CI mode
    fn profile_ci(
        &self,
        model_path: &Path,
        min_throughput: Option<f64>,
        max_p99: Option<f64>,
        warmup: u32,
        measure: u32,
        no_gpu: bool,
    ) -> CommandOutput;

    /// Execute apr rosetta diff-tensors
    fn diff_tensors(&self, model_a: &Path, model_b: &Path, json: bool) -> CommandOutput;

    /// Execute apr rosetta compare-inference
    fn compare_inference(
        &self,
        model_a: &Path,
        model_b: &Path,
        prompt: &str,
        max_tokens: u32,
        tolerance: f64,
    ) -> CommandOutput;

    /// Execute apr run with --profile and --profile-output for flamegraph
    fn profile_with_flamegraph(
        &self,
        model_path: &Path,
        output_path: &Path,
        no_gpu: bool,
    ) -> CommandOutput;

    /// Execute apr run with --profile and --focus
    fn profile_with_focus(&self, model_path: &Path, focus: &str, no_gpu: bool) -> CommandOutput;

    /// Execute an apr validate command with --strict --json flags
    ///
    /// Runs physics-level validation: detects NaN, Inf, and all-zeros tensors
    /// in model weights. Used by the G0-VALIDATE pre-flight gate.
    fn validate_model_strict(&self, model_path: &Path) -> CommandOutput;

    /// Execute apr rosetta fingerprint to capture tensor statistics
    fn fingerprint_model(&self, model_path: &Path, json: bool) -> CommandOutput;

    /// Execute apr rosetta validate-stats to compare tensor statistics
    fn validate_stats(&self, fp_a: &Path, fp_b: &Path) -> CommandOutput;

    /// Execute `apr pull --json <hf_repo>` to acquire model from cache or remote
    fn pull_model(&self, hf_repo: &str) -> CommandOutput;

    /// Execute `apr rosetta inspect --json` to get model metadata including tensor names
    ///
    /// Returns JSON output with tensor_count, tensor_names, and other model metadata.
    /// Used by G0-TENSOR-001 for tensor template validation (PMAT-271).
    fn inspect_model_json(&self, model_path: &Path) -> CommandOutput;

    /// Execute `ollama run <model_tag>` for parity testing (GH-6/AC-2)
    fn run_ollama_inference(
        &self,
        model_tag: &str,
        prompt: &str,
        temperature: f64,
    ) -> CommandOutput;

    /// Execute `ollama pull <model_tag>` to acquire model (GH-6/AC-2)
    fn pull_ollama_model(&self, model_tag: &str) -> CommandOutput;

    /// Execute `ollama create <tag> -f <modelfile>` to register a GGUF with ollama (F-OLLAMA-005)
    fn create_ollama_model(&self, model_tag: &str, modelfile_path: &Path) -> CommandOutput;

    /// Execute `apr serve` and return immediately (F-OLLAMA-004)
    ///
    /// The returned output contains the PID or server info in stdout.
    fn serve_model(&self, model_path: &Path, port: u16) -> CommandOutput;

    /// Execute an HTTP GET request (F-OLLAMA-004)
    fn http_get(&self, url: &str) -> CommandOutput;

    /// Execute `apr profile --memory` for memory usage (F-PERF-005)
    fn profile_memory(&self, model_path: &Path) -> CommandOutput;

    /// Execute `apr chat` command with prompt piped via stdin (Bug 200)
    fn run_chat(
        &self,
        model_path: &Path,
        prompt: &str,
        no_gpu: bool,
        extra_args: &[&str],
    ) -> CommandOutput;

    /// Execute an HTTP POST request (Bug 200: serve modality)
    fn http_post(&self, url: &str, body: &str) -> CommandOutput;

    /// Spawn `apr serve` in background and return the child process PID (Bug 200)
    ///
    /// Unlike `serve_model` which blocks, this spawns the server process
    /// and returns immediately with the PID in stdout.
    fn spawn_serve(&self, model_path: &Path, port: u16, no_gpu: bool) -> CommandOutput;

    /// Execute `apr quantize --scheme <scheme> --json <model_path> -o <output>`
    fn quantize_model(&self, model_path: &Path, output_path: &Path, scheme: &str) -> CommandOutput;

    /// Execute `apr import --json <source> -o <output>`
    fn import_model(&self, source_path: &Path, output_path: &Path) -> CommandOutput;

    /// Execute `apr prune --method <method> --target-ratio <ratio> --json <model_path> -o <output>`
    fn prune_model(
        &self,
        model_path: &Path,
        output_path: &Path,
        method: &str,
        target_ratio: f64,
    ) -> CommandOutput;

    /// Execute `apr distill --student <student> --data <data> --json <teacher_path> -o <output>`
    fn distill_model(
        &self,
        teacher_path: &Path,
        student_path: &Path,
        output_path: &Path,
        data_path: &str,
    ) -> CommandOutput;
}

/// Real command runner that executes actual subprocess commands
#[derive(Debug, Clone)]
pub struct RealCommandRunner {
    /// Path to apr binary (default: "apr")
    pub apr_binary: String,
}

impl Default for RealCommandRunner {
    /// Create a default RealCommandRunner with "apr" as the binary
    fn default() -> Self {
        Self::new()
    }
}

impl RealCommandRunner {
    /// Create a new real command runner
    #[must_use]
    pub fn new() -> Self {
        Self {
            apr_binary: "apr".to_string(),
        }
    }

    /// Create with custom apr binary path
    #[must_use]
    pub fn with_binary(apr_binary: impl Into<String>) -> Self {
        Self {
            apr_binary: apr_binary.into(),
        }
    }

    /// Execute an apr command with the given arguments and return output
    fn execute(&self, args: &[&str]) -> CommandOutput {
        use std::process::Command;

        match Command::new(&self.apr_binary).args(args).output() {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute command: {e}")),
        }
    }
}

include!("command_runner_impl.rs");
include!("mock_command_runner.rs");
include!("simulate.rs");
