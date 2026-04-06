//! `ValidationEngine` for Batuta Transpilation Validation
//!
//! Orchestrates end-to-end validation of transpiled programs by:
//! 1. Tracing both original and transpiled binaries
//! 2. Extracting unified traces with all observability layers
//! 3. Comparing semantic equivalence using `SemanticValidator`
//! 4. Generating comprehensive validation reports
//!
//! This enables Batuta Phase 4 to verify that Python→Rust (or other)
//! transpilations preserve observable program behavior.

use crate::semantic_equivalence::{SemanticValidator, ValidationResult};
use crate::unified_trace::UnifiedTrace;
use std::path::Path;
use std::time::Duration;

/// Validation report for transpilation verification
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub result: ValidationResult,
    pub original_trace: TraceSummary,
    pub transpiled_trace: TraceSummary,
    pub comparison: TraceComparison,
}

/// Summary statistics for a trace
#[derive(Debug, Clone, PartialEq)]
pub struct TraceSummary {
    pub total_syscalls: usize,
    pub total_duration_nanos: u64,
    pub exit_code: Option<i32>,
    pub gpu_kernels: usize,
    pub simd_blocks: usize,
    pub transpiler_decisions: usize,
}

/// Detailed comparison between two traces
#[derive(Debug, Clone, PartialEq)]
pub struct TraceComparison {
    pub syscall_delta: i64,       // positive if transpiled has more
    pub runtime_delta_nanos: i64, // positive if transpiled is slower
    pub speedup_factor: f64,      // transpiled_time / original_time
    pub gpu_kernel_delta: i64,
    pub simd_block_delta: i64,
}

/// Main validation engine for Batuta integration
pub struct ValidationEngine {
    validator: SemanticValidator,
    tracer_timeout: Duration,
}

impl Default for ValidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationEngine {
    /// Create a new `ValidationEngine` with default settings
    pub fn new() -> Self {
        Self {
            validator: SemanticValidator::new(),
            tracer_timeout: Duration::from_secs(300), // 5 minutes default
        }
    }

    /// Create a `ValidationEngine` with custom tolerance
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            validator: SemanticValidator::with_tolerance(tolerance),
            tracer_timeout: Duration::from_secs(300),
        }
    }

    /// Set the tracer timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.tracer_timeout = timeout;
        self
    }

    /// Validate transpilation by tracing both binaries and comparing
    ///
    /// This is the main entry point for Batuta Phase 4 validation.
    ///
    /// # Arguments
    /// * `original_binary` - Path to the original program binary
    /// * `transpiled_binary` - Path to the transpiled program binary
    /// * `args` - Command-line arguments to pass to both programs
    ///
    /// # Returns
    /// `ValidationReport` containing the comparison results
    ///
    /// # Errors
    /// Returns error if tracing fails or binaries cannot be executed
    pub fn validate_transpilation(
        &self,
        original_binary: &Path,
        transpiled_binary: &Path,
        args: &[String],
    ) -> Result<ValidationReport, ValidationError> {
        // 1. Trace original program
        let original_trace = self.trace_binary(original_binary, args)?;

        // 2. Trace transpiled program
        let transpiled_trace = self.trace_binary(transpiled_binary, args)?;

        // 3. Validate semantic equivalence
        let result = self.validator.validate(&original_trace, &transpiled_trace);

        // 4. Generate comprehensive report
        Ok(ValidationReport {
            result,
            original_trace: Self::summarize_trace(&original_trace),
            transpiled_trace: Self::summarize_trace(&transpiled_trace),
            comparison: Self::compare_traces(&original_trace, &transpiled_trace),
        })
    }

    /// Trace a binary and return a `UnifiedTrace`
    ///
    /// This would integrate with the existing Renacer tracer infrastructure.
    /// For now, this is a placeholder that returns a mock trace for testing.
    fn trace_binary(
        &self,
        _binary_path: &Path,
        _args: &[String],
    ) -> Result<UnifiedTrace, ValidationError> {
        // Future: Integration with actual Tracer
        // This would call: Tracer::new().trace(binary_path, args, self.tracer_timeout)
        Err(ValidationError::TracingNotImplemented)
    }

    /// Generate summary statistics for a trace
    fn summarize_trace(trace: &UnifiedTrace) -> TraceSummary {
        TraceSummary {
            total_syscalls: trace.syscall_spans.len(),
            total_duration_nanos: trace
                .process_span
                .end_timestamp_nanos
                .unwrap_or(trace.process_span.start_timestamp_nanos)
                - trace.process_span.start_timestamp_nanos,
            exit_code: trace.process_span.exit_code,
            gpu_kernels: trace.gpu_spans.len(),
            simd_blocks: trace.simd_spans.len(),
            transpiler_decisions: trace.transpiler_spans.len(),
        }
    }

    /// Compare two traces and calculate deltas
    fn compare_traces(original: &UnifiedTrace, transpiled: &UnifiedTrace) -> TraceComparison {
        let original_runtime = original
            .process_span
            .end_timestamp_nanos
            .unwrap_or(original.process_span.start_timestamp_nanos)
            - original.process_span.start_timestamp_nanos;

        let transpiled_runtime = transpiled
            .process_span
            .end_timestamp_nanos
            .unwrap_or(transpiled.process_span.start_timestamp_nanos)
            - transpiled.process_span.start_timestamp_nanos;

        let speedup_factor = if transpiled_runtime > 0 {
            original_runtime as f64 / transpiled_runtime as f64
        } else {
            1.0
        };

        TraceComparison {
            syscall_delta: transpiled.syscall_spans.len() as i64
                - original.syscall_spans.len() as i64,
            runtime_delta_nanos: transpiled_runtime as i64 - original_runtime as i64,
            speedup_factor,
            gpu_kernel_delta: transpiled.gpu_spans.len() as i64 - original.gpu_spans.len() as i64,
            simd_block_delta: transpiled.simd_spans.len() as i64 - original.simd_spans.len() as i64,
        }
    }
}

/// Errors that can occur during validation
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Binary file not found
    BinaryNotFound { path: String },

    /// Binary is not executable
    NotExecutable { path: String },

    /// Tracing failed with error
    TracingFailed { binary: String, error: String },

    /// Tracer timeout exceeded
    TracerTimeout { binary: String, timeout_secs: u64 },

    /// Integration with Tracer not yet implemented
    TracingNotImplemented,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::BinaryNotFound { path } => {
                write!(f, "Binary not found: {path}")
            }
            ValidationError::NotExecutable { path } => {
                write!(f, "Binary is not executable: {path}")
            }
            ValidationError::TracingFailed { binary, error } => {
                write!(f, "Tracing failed for {binary}: {error}")
            }
            ValidationError::TracerTimeout { binary, timeout_secs } => {
                write!(f, "Tracer timeout exceeded for {binary} after {timeout_secs} seconds")
            }
            ValidationError::TracingNotImplemented => {
                write!(f, "Tracer integration not yet implemented")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(ValidationReport: Send, Sync);
static_assertions::assert_impl_all!(TraceSummary: Send, Sync);
static_assertions::assert_impl_all!(TraceComparison: Send, Sync);
static_assertions::assert_impl_all!(ValidationError: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace_context::LamportClock;
    use crate::unified_trace::SyscallSpan;
    use std::borrow::Cow;

    // Helper: Create a minimal UnifiedTrace for testing
    fn create_test_trace(
        pid: i32,
        name: &str,
        syscalls: Vec<(&'static str, i64, u64)>,
    ) -> UnifiedTrace {
        let clock = LamportClock::new();
        let mut trace = UnifiedTrace::new(pid, name.to_string());

        for (syscall_name, ret_val, duration) in syscalls {
            let span = SyscallSpan::new(
                trace.process_span.span_id,
                Cow::Borrowed(syscall_name),
                vec![],
                ret_val,
                clock.now(),
                duration,
                None,
                &clock,
            );
            trace.add_syscall(span);
        }
        trace
    }

    #[test]
    fn test_validation_engine_new() {
        let engine = ValidationEngine::new();
        assert_eq!(engine.tracer_timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_validation_engine_with_tolerance() {
        let engine = ValidationEngine::with_tolerance(0.10);
        // Validator tolerance is not publicly accessible, so we verify via behavior
        assert_eq!(engine.tracer_timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_validation_engine_with_timeout() {
        let engine = ValidationEngine::new().with_timeout(Duration::from_secs(60));
        assert_eq!(engine.tracer_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_summarize_trace_basic() {
        let trace = create_test_trace(
            100,
            "test_program",
            vec![("open", 3, 1000), ("read", 1024, 2000), ("close", 0, 500)],
        );

        let summary = ValidationEngine::summarize_trace(&trace);

        assert_eq!(summary.total_syscalls, 3);
        assert_eq!(summary.exit_code, None);
        assert_eq!(summary.gpu_kernels, 0);
        assert_eq!(summary.simd_blocks, 0);
        assert_eq!(summary.transpiler_decisions, 0);
    }

    #[test]
    fn test_summarize_trace_with_exit_code() {
        let mut trace = UnifiedTrace::new(100, "test_program".to_string());
        trace.process_span.exit_code = Some(0);
        trace.process_span.end_timestamp_nanos =
            Some(trace.process_span.start_timestamp_nanos + 10000);

        let summary = ValidationEngine::summarize_trace(&trace);

        assert_eq!(summary.exit_code, Some(0));
        assert_eq!(summary.total_duration_nanos, 10000);
    }

    #[test]
    fn test_compare_traces_identical() {
        let trace1 =
            create_test_trace(100, "original", vec![("open", 3, 1000), ("read", 1024, 2000)]);
        let trace2 =
            create_test_trace(200, "transpiled", vec![("open", 3, 1000), ("read", 1024, 2000)]);

        let comparison = ValidationEngine::compare_traces(&trace1, &trace2);

        assert_eq!(comparison.syscall_delta, 0);
        assert_eq!(comparison.gpu_kernel_delta, 0);
        assert_eq!(comparison.simd_block_delta, 0);
    }

    #[test]
    fn test_compare_traces_transpiled_has_more_syscalls() {
        let trace1 = create_test_trace(100, "original", vec![("open", 3, 1000)]);
        let trace2 = create_test_trace(
            200,
            "transpiled",
            vec![("open", 3, 1000), ("mmap", 0, 500), ("munmap", 0, 300)],
        );

        let comparison = ValidationEngine::compare_traces(&trace1, &trace2);

        assert_eq!(comparison.syscall_delta, 2); // transpiled has 2 more
    }

    #[test]
    fn test_compare_traces_speedup_calculation() {
        // Original: 10ms runtime
        let mut trace1 = UnifiedTrace::new(100, "original".to_string());
        trace1.process_span.end_timestamp_nanos =
            Some(trace1.process_span.start_timestamp_nanos + 10_000_000);

        // Transpiled: 5ms runtime (2x speedup)
        let mut trace2 = UnifiedTrace::new(200, "transpiled".to_string());
        trace2.process_span.end_timestamp_nanos =
            Some(trace2.process_span.start_timestamp_nanos + 5_000_000);

        let comparison = ValidationEngine::compare_traces(&trace1, &trace2);

        assert_eq!(comparison.runtime_delta_nanos, -5_000_000); // transpiled is 5ms faster
        assert!((comparison.speedup_factor - 2.0).abs() < 0.001); // 2x speedup
    }

    #[test]
    fn test_compare_traces_slowdown() {
        // Original: 5ms runtime
        let mut trace1 = UnifiedTrace::new(100, "original".to_string());
        trace1.process_span.end_timestamp_nanos =
            Some(trace1.process_span.start_timestamp_nanos + 5_000_000);

        // Transpiled: 10ms runtime (0.5x slowdown)
        let mut trace2 = UnifiedTrace::new(200, "transpiled".to_string());
        trace2.process_span.end_timestamp_nanos =
            Some(trace2.process_span.start_timestamp_nanos + 10_000_000);

        let comparison = ValidationEngine::compare_traces(&trace1, &trace2);

        assert_eq!(comparison.runtime_delta_nanos, 5_000_000); // transpiled is 5ms slower
        assert!((comparison.speedup_factor - 0.5).abs() < 0.001); // 0.5x (slowdown)
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::BinaryNotFound { path: "/tmp/test".to_string() };
        assert_eq!(err.to_string(), "Binary not found: /tmp/test");

        let err = ValidationError::NotExecutable { path: "/tmp/test".to_string() };
        assert_eq!(err.to_string(), "Binary is not executable: /tmp/test");

        let err = ValidationError::TracingFailed {
            binary: "test".to_string(),
            error: "segfault".to_string(),
        };
        assert_eq!(err.to_string(), "Tracing failed for test: segfault");

        let err = ValidationError::TracerTimeout { binary: "test".to_string(), timeout_secs: 300 };
        assert_eq!(err.to_string(), "Tracer timeout exceeded for test after 300 seconds");

        let err = ValidationError::TracingNotImplemented;
        assert_eq!(err.to_string(), "Tracer integration not yet implemented");
    }

    #[test]
    fn test_validate_transpilation_not_implemented() {
        let engine = ValidationEngine::new();
        let result = engine.validate_transpilation(
            Path::new("/tmp/original"),
            Path::new("/tmp/transpiled"),
            &[],
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ValidationError::TracingNotImplemented);
    }

    // Integration test: When tracing is implemented, this will test the full workflow
    #[test]
    fn test_validation_report_structure() {
        // This test verifies the ValidationReport structure is correct
        // Once tracing is implemented, we'll add end-to-end tests

        use crate::semantic_equivalence::{PerformanceComparison, ValidationResult};

        let trace1 = create_test_trace(100, "original", vec![("open", 3, 1000)]);
        let trace2 = create_test_trace(200, "transpiled", vec![("open", 3, 1000)]);

        let result = ValidationResult::Pass {
            confidence: 1.0,
            matched_syscalls: 1,
            performance: PerformanceComparison {
                original_runtime_nanos: 1000,
                transpiled_runtime_nanos: 1000,
                speedup: 1.0,
                memory_delta: None,
            },
        };

        let report = ValidationReport {
            result,
            original_trace: ValidationEngine::summarize_trace(&trace1),
            transpiled_trace: ValidationEngine::summarize_trace(&trace2),
            comparison: ValidationEngine::compare_traces(&trace1, &trace2),
        };

        assert!(matches!(report.result, ValidationResult::Pass { .. }));
        assert_eq!(report.original_trace.total_syscalls, 1);
        assert_eq!(report.transpiled_trace.total_syscalls, 1);
        assert_eq!(report.comparison.syscall_delta, 0);
    }

    #[test]
    fn test_validation_report_with_failure() {
        use crate::semantic_equivalence::{DivergencePoint, ValidationResult};

        let trace1 = create_test_trace(100, "original", vec![("open", 3, 1000)]);
        let trace2 = create_test_trace(200, "transpiled", vec![("openat", 3, 1000)]);

        let result = ValidationResult::Fail {
            divergence_point: DivergencePoint {
                syscall_index: 0,
                original_syscall: "open".to_string(),
                transpiled_syscall: "openat".to_string(),
            },
            explanation: "Syscall mismatch at index 0".to_string(),
        };

        let report = ValidationReport {
            result,
            original_trace: ValidationEngine::summarize_trace(&trace1),
            transpiled_trace: ValidationEngine::summarize_trace(&trace2),
            comparison: ValidationEngine::compare_traces(&trace1, &trace2),
        };

        assert!(matches!(report.result, ValidationResult::Fail { .. }));
    }

    #[test]
    fn test_trace_summary_counts_all_layers() {
        // Future: Once we have GPU/SIMD spans, verify they're counted correctly
        let trace = create_test_trace(100, "test", vec![("open", 3, 1000)]);
        let summary = ValidationEngine::summarize_trace(&trace);

        assert_eq!(summary.total_syscalls, 1);
        assert_eq!(summary.gpu_kernels, 0);
        assert_eq!(summary.simd_blocks, 0);
    }
}
