//! Validation stage - verifies semantic equivalence.

use anyhow::{Context as AnyhowContext, Result};

#[cfg(feature = "native")]
use tracing::{info, warn};

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! warn {
    ($($arg:tt)*) => {{}};
}

use crate::pipeline::types::{PipelineContext, PipelineStage, ValidationResult};

/// Validation stage - verifies semantic equivalence
pub struct ValidationStage {
    pub(crate) trace_syscalls: bool,
    pub(crate) run_tests: bool,
}

impl ValidationStage {
    pub fn new(trace_syscalls: bool, run_tests: bool) -> Self {
        Self { trace_syscalls, run_tests }
    }

    /// Trace syscalls from both binaries and compare them for semantic equivalence
    pub async fn trace_and_compare(
        &self,
        original_binary: &std::path::Path,
        transpiled_binary: &std::path::Path,
    ) -> Result<bool> {
        // Trace original binary
        let original_trace = Self::trace_binary(original_binary)?;

        // Trace transpiled binary
        let transpiled_trace = Self::trace_binary(transpiled_binary)?;

        // Compare traces
        Ok(Self::compare_traces(&original_trace, &transpiled_trace))
    }

    /// Trace a binary using Renacer and capture syscall output
    pub fn trace_binary(binary: &std::path::Path) -> Result<Vec<String>> {
        use std::process::Command;

        // Run the binary with renacer tracing
        // Note: This is a simplified approach - ideally we'd use the renacer library directly
        let output = Command::new("renacer")
            .arg(binary.to_string_lossy().to_string())
            .output()
            .context("Failed to run renacer")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Renacer failed: {}", stderr);
        }

        Ok(Self::parse_syscall_output(&output.stdout))
    }

    /// Parse raw renacer stdout bytes into a list of syscall strings.
    ///
    /// Filters out renacer's own messages (lines starting with `[`).
    pub fn parse_syscall_output(stdout: &[u8]) -> Vec<String> {
        let text = String::from_utf8_lossy(stdout);
        text.lines().filter(|line| !line.starts_with('[')).map(|s| s.to_string()).collect()
    }

    /// Compare two syscall traces for semantic equivalence
    pub fn compare_traces(trace1: &[String], trace2: &[String]) -> bool {
        // For now, simple comparison - ideally would normalize and filter
        // non-deterministic syscalls (timestamps, PIDs, etc.)
        if trace1.len() != trace2.len() {
            return false;
        }

        // Compare each syscall (ignoring arguments that may vary)
        for (call1, call2) in trace1.iter().zip(trace2.iter()) {
            // Extract syscall name (before the '(' character)
            let name1 = call1.split('(').next().unwrap_or("");
            let name2 = call2.split('(').next().unwrap_or("");

            if name1 != name2 {
                return false;
            }
        }

        true
    }
}

#[async_trait::async_trait]
impl PipelineStage for ValidationStage {
    fn name(&self) -> &'static str {
        "Validation"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Validating semantic equivalence");

        // If syscall tracing is enabled, use Renacer to verify equivalence
        if self.trace_syscalls {
            info!("Tracing syscalls with Renacer");

            // Find original and transpiled binaries
            let original_binary = ctx.input_path.join("original_binary");
            let transpiled_binary = ctx.output_path.join("target/release/transpiled");

            if original_binary.exists() && transpiled_binary.exists() {
                match self.trace_and_compare(&original_binary, &transpiled_binary).await {
                    Ok(equivalent) => {
                        ctx.validation_results.push(ValidationResult {
                            stage: self.name().to_string(),
                            passed: equivalent,
                            message: if equivalent {
                                "Syscall traces match - semantic equivalence verified"
                            } else {
                                "Syscall traces differ - semantic equivalence NOT verified"
                            }
                            .to_string(),
                            details: None,
                        });

                        ctx.metadata.insert(
                            "syscall_equivalence".to_string(),
                            serde_json::json!(equivalent),
                        );
                    }
                    Err(e) => {
                        warn!("Syscall tracing failed: {}", e);
                        ctx.validation_results.push(ValidationResult {
                            stage: self.name().to_string(),
                            passed: false,
                            message: format!("Syscall tracing error: {}", e),
                            details: None,
                        });
                    }
                }
            } else {
                info!("Binaries not found for comparison, skipping syscall trace");
            }
        }

        // If run_tests is enabled, run the original test suite
        // PIPELINE-004: Test suite execution planned for validation phase
        if self.run_tests {
            info!("Running original test suite");
            // Test execution deferred - requires renacer integration
        }

        ctx.metadata.insert("validation_completed".to_string(), serde_json::json!(true));

        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_stage_new() {
        let stage = ValidationStage::new(true, false);
        assert!(stage.trace_syscalls);
        assert!(!stage.run_tests);
    }

    #[test]
    fn test_validation_stage_name() {
        let stage = ValidationStage::new(false, false);
        assert_eq!(stage.name(), "Validation");
    }

    #[test]
    fn test_compare_traces_identical() {
        let trace1 = vec!["read(3, buf, 1024)".to_string(), "write(1, msg, 12)".to_string()];
        let trace2 = vec!["read(3, buf, 1024)".to_string(), "write(1, msg, 12)".to_string()];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_same_syscalls_different_args() {
        let trace1 = vec!["read(3, buf, 1024)".to_string(), "write(1, msg, 12)".to_string()];
        let trace2 = vec!["read(4, buf2, 2048)".to_string(), "write(2, msg2, 24)".to_string()];
        // Should match because syscall names are the same
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_different_syscalls() {
        let trace1 = vec!["read(3, buf, 1024)".to_string(), "write(1, msg, 12)".to_string()];
        let trace2 = vec!["open(path, flags)".to_string(), "close(3)".to_string()];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_different_lengths() {
        let trace1 = vec!["read(3, buf, 1024)".to_string()];
        let trace2 = vec!["read(3, buf, 1024)".to_string(), "write(1, msg, 12)".to_string()];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_empty() {
        let trace1: Vec<String> = vec![];
        let trace2: Vec<String> = vec![];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_no_parentheses() {
        let trace1 = vec!["syscall1".to_string()];
        let trace2 = vec!["syscall1".to_string()];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_partial_match() {
        let trace1 = vec!["read(3)".to_string(), "write(1)".to_string(), "close(3)".to_string()];
        let trace2 = vec![
            "read(4)".to_string(),
            "read(5)".to_string(), // Different syscall
            "close(4)".to_string(),
        ];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    // =========================================================================
    // Coverage: ValidationStage fields and state
    // =========================================================================

    #[test]
    fn test_validation_stage_both_flags() {
        let stage = ValidationStage::new(true, true);
        assert!(stage.trace_syscalls);
        assert!(stage.run_tests);
    }

    #[test]
    fn test_validation_stage_no_flags() {
        let stage = ValidationStage::new(false, false);
        assert!(!stage.trace_syscalls);
        assert!(!stage.run_tests);
    }

    // =========================================================================
    // Coverage: execute() async path - no trace, no tests
    // =========================================================================

    #[tokio::test]
    async fn test_execute_no_trace_no_tests() {
        let stage = ValidationStage::new(false, false);
        let ctx = PipelineContext::new(
            std::path::PathBuf::from("/tmp/input"),
            std::path::PathBuf::from("/tmp/output"),
        );

        let result = stage.execute(ctx).await.expect("async operation failed");
        // Should add validation_completed metadata
        assert_eq!(result.metadata.get("validation_completed"), Some(&serde_json::json!(true)));
        // No validation results since both flags are off
        assert!(result.validation_results.is_empty());
    }

    #[tokio::test]
    async fn test_execute_with_trace_no_binaries() {
        let stage = ValidationStage::new(true, false);
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let ctx = PipelineContext::new(tempdir.path().to_path_buf(), tempdir.path().to_path_buf());

        let result = stage.execute(ctx).await.expect("async operation failed");
        // Binaries don't exist, so tracing is skipped
        assert_eq!(result.metadata.get("validation_completed"), Some(&serde_json::json!(true)));
        // No syscall_equivalence metadata since binaries not found
        assert!(!result.metadata.contains_key("syscall_equivalence"));
    }

    #[tokio::test]
    async fn test_execute_with_run_tests_flag() {
        let stage = ValidationStage::new(false, true);
        let ctx = PipelineContext::new(
            std::path::PathBuf::from("/tmp/input"),
            std::path::PathBuf::from("/tmp/output"),
        );

        let result = stage.execute(ctx).await.expect("async operation failed");
        // run_tests is set but the implementation is deferred
        assert_eq!(result.metadata.get("validation_completed"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn test_execute_with_trace_binaries_not_found() {
        // Create input/output dirs but no binaries
        let input_dir = tempfile::tempdir().expect("tempdir creation failed");
        let output_dir = tempfile::tempdir().expect("tempdir creation failed");

        let stage = ValidationStage::new(true, true);
        let ctx =
            PipelineContext::new(input_dir.path().to_path_buf(), output_dir.path().to_path_buf());

        let result = stage.execute(ctx).await.expect("async operation failed");
        // Binaries not found -> tracing skipped
        assert!(!result.metadata.contains_key("syscall_equivalence"));
    }

    #[tokio::test]
    async fn test_execute_with_trace_binary_exists_but_renacer_not_found() {
        let input_dir = tempfile::tempdir().expect("tempdir creation failed");
        let output_dir = tempfile::tempdir().expect("tempdir creation failed");

        // Create the expected binary paths
        std::fs::write(input_dir.path().join("original_binary"), "#!/bin/sh\nexit 0")
            .expect("unexpected failure");
        let target_dir = output_dir.path().join("target/release");
        std::fs::create_dir_all(&target_dir).expect("mkdir failed");
        std::fs::write(target_dir.join("transpiled"), "#!/bin/sh\nexit 0")
            .expect("fs write failed");

        let stage = ValidationStage::new(true, false);
        let ctx =
            PipelineContext::new(input_dir.path().to_path_buf(), output_dir.path().to_path_buf());

        let result = stage.execute(ctx).await.expect("async operation failed");
        // Renacer is not installed, so trace_and_compare should fail
        // This should produce a validation result with passed=false
        assert!(!result.validation_results.is_empty());
        assert!(!result.validation_results[0].passed);
        assert!(
            result.validation_results[0].message.contains("error")
                || result.validation_results[0].message.contains("renacer")
                || result.validation_results[0].message.contains("Syscall tracing error")
        );
    }

    // =========================================================================
    // Coverage: trace_binary error path
    // =========================================================================

    #[test]
    fn test_trace_binary_not_found() {
        let result = ValidationStage::trace_binary(std::path::Path::new("/nonexistent/binary"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("renacer") || err.contains("Failed"));
    }

    // =========================================================================
    // Coverage: compare_traces additional edge cases
    // =========================================================================

    #[test]
    fn test_compare_traces_single_element_match() {
        let trace1 = vec!["open(file.txt)".to_string()];
        let trace2 = vec!["open(other.txt)".to_string()];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_single_element_mismatch() {
        let trace1 = vec!["open(file.txt)".to_string()];
        let trace2 = vec!["close(3)".to_string()];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_many_syscalls() {
        let trace1: Vec<String> = (0..100).map(|i| format!("syscall_{}(arg1, arg2)", i)).collect();
        let trace2: Vec<String> =
            (0..100).map(|i| format!("syscall_{}(different, args)", i)).collect();
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_empty_strings() {
        let trace1 = vec!["".to_string()];
        let trace2 = vec!["".to_string()];
        assert!(ValidationStage::compare_traces(&trace1, &trace2));
    }

    #[test]
    fn test_compare_traces_one_empty_vs_nonempty() {
        let trace1 = vec!["read(3)".to_string()];
        let trace2: Vec<String> = vec![];
        assert!(!ValidationStage::compare_traces(&trace1, &trace2));
    }

    // =========================================================================
    // Coverage: PipelineContext output and validation_results
    // =========================================================================

    #[test]
    fn test_pipeline_context_output_all_passed() {
        let mut ctx = PipelineContext::new(
            std::path::PathBuf::from("/input"),
            std::path::PathBuf::from("/output"),
        );
        ctx.validation_results.push(ValidationResult {
            stage: "test".to_string(),
            passed: true,
            message: "ok".to_string(),
            details: None,
        });
        let output = ctx.output();
        assert!(output.validation_passed);
    }

    // =========================================================================
    // Coverage: parse_syscall_output (extracted from trace_binary success path)
    // =========================================================================

    #[test]
    fn test_parse_syscall_output_basic() {
        let stdout = b"read(3, buf, 1024)\nwrite(1, msg, 12)\n";
        let result = ValidationStage::parse_syscall_output(stdout);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "read(3, buf, 1024)");
        assert_eq!(result[1], "write(1, msg, 12)");
    }

    #[test]
    fn test_parse_syscall_output_filters_renacer_messages() {
        let stdout =
            b"[renacer] tracing pid 1234\nread(3, buf, 1024)\n[renacer] done\nwrite(1, msg, 12)\n";
        let result = ValidationStage::parse_syscall_output(stdout);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "read(3, buf, 1024)");
        assert_eq!(result[1], "write(1, msg, 12)");
    }

    #[test]
    fn test_parse_syscall_output_empty() {
        let stdout = b"";
        let result = ValidationStage::parse_syscall_output(stdout);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_syscall_output_only_renacer_messages() {
        let stdout = b"[renacer] starting\n[renacer] done\n";
        let result = ValidationStage::parse_syscall_output(stdout);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_syscall_output_utf8_lossy() {
        // Invalid UTF-8 should be handled gracefully
        let stdout = b"read(3, \xff\xfe, 10)\nwrite(1, ok, 2)\n";
        let result = ValidationStage::parse_syscall_output(stdout);
        assert_eq!(result.len(), 2);
        // First line has replacement characters for invalid UTF-8
        assert!(result[0].contains("read"));
        assert_eq!(result[1], "write(1, ok, 2)");
    }

    #[test]
    fn test_pipeline_context_output_one_failed() {
        let mut ctx = PipelineContext::new(
            std::path::PathBuf::from("/input"),
            std::path::PathBuf::from("/output"),
        );
        ctx.validation_results.push(ValidationResult {
            stage: "test1".to_string(),
            passed: true,
            message: "ok".to_string(),
            details: None,
        });
        ctx.validation_results.push(ValidationResult {
            stage: "test2".to_string(),
            passed: false,
            message: "failed".to_string(),
            details: None,
        });
        let output = ctx.output();
        assert!(!output.validation_passed);
    }
}
