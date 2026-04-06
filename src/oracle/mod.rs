//! Verification oracle for transpilation correctness
//!
//! The oracle executes source and target code and compares outputs
//! to verify transpilation correctness.
//!
//! # Verification Strategy
//!
//! From spec Section 5.1:
//! 1. **Fast path (I/O oracle)**: Execute source & target, diff outputs - catches 95%+ of bugs
//! 2. **Slow path (SMT/Z3)**: For critical paths (security-sensitive, memory ops)
//! 3. **Property proofs**: Encode transpilation invariants as SMT constraints
//!
//! # Example
//!
//! ```rust,ignore
//! use verificar::oracle::{IoOracle, DiffOptions};
//!
//! let oracle = IoOracle::new();
//! let result = oracle.verify_python("print(1+1)", "2\n")?;
//! assert!(result.matches);
//! ```

mod diff;
mod executor;
mod sandbox;
mod semantic;

pub use diff::{diff_results, format_diff, DiffOptions, DiffResult, Difference, DifferenceKind};
pub use executor::{executor_for, Executor, PythonExecutor, RustExecutor};
pub use sandbox::{SandboxConfig, SandboxedPythonExecutor};
pub use semantic::{
    AstSemanticOracle, CombinedSemanticOracle, Complexity, DifferenceCategory, DifferenceDetails,
    FormalVerificationOracle, HeapAllocation, MemoryLayout, PerformanceProfile, SemanticDifference,
    SemanticNode, SemanticOracle, SemanticVerdict,
};

use crate::{Language, Result};

/// Result of executing code
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: i32,
    /// Execution time in milliseconds
    pub duration_ms: u64,
}

/// Verdict from verification
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Verdict {
    /// I/O equivalent - transpilation is correct
    Pass,
    /// Output mismatch - transpilation bug detected
    OutputMismatch {
        /// Expected output (from source)
        expected: String,
        /// Actual output (from target)
        actual: String,
    },
    /// Timeout during execution
    Timeout {
        /// Which phase timed out
        phase: Phase,
        /// Timeout limit in milliseconds
        limit_ms: u64,
    },
    /// Runtime error
    RuntimeError {
        /// Phase where error occurred
        phase: Phase,
        /// Error message
        error: String,
    },
}

/// Verification result with full metadata
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Source code
    pub source_code: String,
    /// Source language
    pub source_language: Language,
    /// Target code
    pub target_code: String,
    /// Target language
    pub target_language: Language,
    /// Verification verdict
    pub verdict: Verdict,
    /// Source execution result (if available)
    pub source_result: Option<ExecutionResult>,
    /// Target execution result (if available)
    pub target_result: Option<ExecutionResult>,
}

/// Phase of execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Phase {
    /// Executing source code
    Source,
    /// Executing target code
    Target,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source => write!(f, "source"),
            Self::Target => write!(f, "target"),
        }
    }
}

/// Verification oracle trait
///
/// Standardized oracle interface enables cross-transpiler comparison.
pub trait Oracle: Send + Sync {
    /// Execute source code and return result
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    fn execute_source(&self, code: &str, input: &str) -> Result<ExecutionResult>;

    /// Execute target code and return result
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    fn execute_target(&self, code: &str, input: &str) -> Result<ExecutionResult>;

    /// Compare source and target execution results
    fn compare(&self, source: &ExecutionResult, target: &ExecutionResult) -> Verdict;

    /// Get the timeout for execution in milliseconds
    fn timeout_ms(&self) -> u64 {
        5000 // 5 second default
    }
}

/// Default I/O-based verification oracle
pub struct IoOracle {
    timeout_ms: u64,
    diff_options: DiffOptions,
    source_executor: Box<dyn Executor>,
    target_executor: Option<Box<dyn Executor>>,
}

impl std::fmt::Debug for IoOracle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoOracle")
            .field("timeout_ms", &self.timeout_ms)
            .field("diff_options", &self.diff_options)
            .field(
                "source_executor",
                &format!("<{}>", self.source_executor.language()),
            )
            .field(
                "target_executor",
                &self
                    .target_executor
                    .as_ref()
                    .map(|e| format!("<{}>", e.language())),
            )
            .finish()
    }
}

impl Default for IoOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl IoOracle {
    /// Create a new I/O oracle with default settings (Python source)
    #[must_use]
    pub fn new() -> Self {
        Self {
            timeout_ms: 5000,
            diff_options: DiffOptions::default(),
            source_executor: Box::new(PythonExecutor::new()),
            target_executor: None,
        }
    }

    /// Create an I/O oracle with custom timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set diff options for comparison
    #[must_use]
    pub fn with_diff_options(mut self, options: DiffOptions) -> Self {
        self.diff_options = options;
        self
    }

    /// Set the source language executor
    #[must_use]
    pub fn with_source_executor(mut self, executor: Box<dyn Executor>) -> Self {
        self.source_executor = executor;
        self
    }

    /// Set the target language executor
    #[must_use]
    pub fn with_target_executor(mut self, executor: Box<dyn Executor>) -> Self {
        self.target_executor = Some(executor);
        self
    }

    /// Verify transpilation correctness
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails
    pub fn verify(
        &self,
        source_code: &str,
        target_code: &str,
        input: &str,
        source_lang: Language,
        target_lang: Language,
    ) -> Result<VerificationResult> {
        let source_result = self.execute_source(source_code, input)?;

        let target_result = if let Some(ref target_exec) = self.target_executor {
            target_exec.execute(target_code, input, self.timeout_ms)?
        } else {
            self.execute_target(target_code, input)?
        };

        let verdict = self.compare(&source_result, &target_result);

        Ok(VerificationResult {
            source_code: source_code.to_string(),
            source_language: source_lang,
            target_code: target_code.to_string(),
            target_language: target_lang,
            verdict,
            source_result: Some(source_result),
            target_result: Some(target_result),
        })
    }

    /// Execute Python code and verify against expected output
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    pub fn verify_python(&self, code: &str, expected_output: &str) -> Result<DiffResult> {
        let result = self.source_executor.execute(code, "", self.timeout_ms)?;

        let expected = ExecutionResult {
            stdout: expected_output.to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 0,
        };

        Ok(diff_results(&expected, &result, &self.diff_options))
    }

    /// Execute Python code and verify against expected output with input
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    pub fn verify_python_with_input(
        &self,
        code: &str,
        input: &str,
        expected_output: &str,
    ) -> Result<DiffResult> {
        let result = self.source_executor.execute(code, input, self.timeout_ms)?;

        let expected = ExecutionResult {
            stdout: expected_output.to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 0,
        };

        Ok(diff_results(&expected, &result, &self.diff_options))
    }

    /// Execute code and return raw result
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    pub fn execute_python(&self, code: &str, input: &str) -> Result<ExecutionResult> {
        self.source_executor.execute(code, input, self.timeout_ms)
    }

    /// Get diff options
    #[must_use]
    pub fn diff_options(&self) -> &DiffOptions {
        &self.diff_options
    }
}

impl Oracle for IoOracle {
    fn execute_source(&self, code: &str, input: &str) -> Result<ExecutionResult> {
        self.source_executor.execute(code, input, self.timeout_ms)
    }

    fn execute_target(&self, code: &str, input: &str) -> Result<ExecutionResult> {
        if let Some(ref executor) = self.target_executor {
            executor.execute(code, input, self.timeout_ms)
        } else {
            // Default: use same executor as source (for same-language testing)
            self.source_executor.execute(code, input, self.timeout_ms)
        }
    }

    fn compare(&self, source: &ExecutionResult, target: &ExecutionResult) -> Verdict {
        let diff = diff_results(source, target, &self.diff_options);

        if diff.matches {
            Verdict::Pass
        } else {
            Verdict::OutputMismatch {
                expected: source.stdout.clone(),
                actual: target.stdout.clone(),
            }
        }
    }

    fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verdict_pass() {
        let oracle = IoOracle::new();
        let source = ExecutionResult {
            stdout: "hello".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 10,
        };
        let target = ExecutionResult {
            stdout: "hello".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 5,
        };

        let verdict = oracle.compare(&source, &target);
        assert_eq!(verdict, Verdict::Pass);
    }

    #[test]
    fn test_verdict_mismatch() {
        let oracle = IoOracle::new();
        let source = ExecutionResult {
            stdout: "hello".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 10,
        };
        let target = ExecutionResult {
            stdout: "world".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 5,
        };

        let verdict = oracle.compare(&source, &target);
        assert!(matches!(verdict, Verdict::OutputMismatch { .. }));
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Source), "source");
        assert_eq!(format!("{}", Phase::Target), "target");
    }

    #[test]
    fn test_io_oracle_verify_python() {
        let oracle = IoOracle::new();
        let executor = PythonExecutor::new();

        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let result = oracle
            .verify_python("print('hello')", "hello\n")
            .expect("verification should succeed");

        assert!(result.matches);
    }

    #[test]
    fn test_io_oracle_verify_python_mismatch() {
        let oracle = IoOracle::new();
        let executor = PythonExecutor::new();

        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let result = oracle
            .verify_python("print('hello')", "world\n")
            .expect("verification should succeed");

        assert!(!result.matches);
    }

    #[test]
    fn test_io_oracle_execute_python() {
        let oracle = IoOracle::new();
        let executor = PythonExecutor::new();

        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let result = oracle
            .execute_python("print(2 + 2)", "")
            .expect("execution should succeed");

        assert_eq!(result.stdout.trim(), "4");
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_io_oracle_with_diff_options() {
        let oracle = IoOracle::new()
            .with_diff_options(DiffOptions::lenient())
            .with_timeout(10000);

        assert_eq!(oracle.timeout_ms(), 10000);
        assert!(oracle.diff_options().normalize_whitespace);
    }

    #[test]
    fn test_io_oracle_same_code_verification() {
        let oracle = IoOracle::new();
        let executor = PythonExecutor::new();

        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        // Verify same code produces same output
        let code = "x = 5\nprint(x * 2)";
        let result = oracle
            .verify(code, code, "", Language::Python, Language::Python)
            .expect("verification should succeed");

        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_io_oracle_default() {
        let oracle = IoOracle::default();
        assert_eq!(oracle.timeout_ms(), 5000);
    }

    #[test]
    fn test_io_oracle_debug() {
        let oracle = IoOracle::new();
        let debug = format!("{:?}", oracle);
        assert!(debug.contains("IoOracle"));
        assert!(debug.contains("timeout_ms"));
    }

    #[test]
    fn test_io_oracle_with_source_executor() {
        let oracle = IoOracle::new().with_source_executor(Box::new(PythonExecutor::new()));
        assert_eq!(oracle.source_executor.language(), Language::Python);
    }

    #[test]
    fn test_io_oracle_with_target_executor() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            return;
        }

        let oracle = IoOracle::new().with_target_executor(Box::new(PythonExecutor::new()));
        assert!(oracle.target_executor.is_some());

        // Test execute_target uses target executor
        let result = oracle.execute_target("print(1)", "");
        assert!(result.is_ok());
    }

    #[test]
    fn test_io_oracle_verify_python_with_input() {
        let oracle = IoOracle::new();
        let executor = PythonExecutor::new();

        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        let code = "x = input()\nprint(f'Hello {x}')";
        let result = oracle
            .verify_python_with_input(code, "World", "Hello World\n")
            .expect("verification should succeed");

        assert!(result.matches);
    }

    #[test]
    fn test_io_oracle_verify_with_target_executor() {
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            return;
        }

        let oracle = IoOracle::new().with_target_executor(Box::new(PythonExecutor::new()));

        let code = "print(42)";
        let result = oracle
            .verify(code, code, "", Language::Python, Language::Python)
            .expect("verification should succeed");

        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_execution_result_debug() {
        let result = ExecutionResult {
            stdout: "test".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 100,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("ExecutionResult"));
    }

    #[test]
    fn test_execution_result_clone() {
        let result = ExecutionResult {
            stdout: "test".to_string(),
            stderr: "err".to_string(),
            exit_code: 1,
            duration_ms: 100,
        };
        let cloned = result.clone();
        assert_eq!(cloned.stdout, result.stdout);
        assert_eq!(cloned.exit_code, result.exit_code);
    }

    #[test]
    fn test_verdict_debug() {
        let verdict = Verdict::Pass;
        let debug = format!("{:?}", verdict);
        assert!(debug.contains("Pass"));

        let mismatch = Verdict::OutputMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
        };
        let debug2 = format!("{:?}", mismatch);
        assert!(debug2.contains("OutputMismatch"));

        let timeout = Verdict::Timeout {
            phase: Phase::Source,
            limit_ms: 5000,
        };
        let debug3 = format!("{:?}", timeout);
        assert!(debug3.contains("Timeout"));

        let error = Verdict::RuntimeError {
            phase: Phase::Target,
            error: "error".to_string(),
        };
        let debug4 = format!("{:?}", error);
        assert!(debug4.contains("RuntimeError"));
    }

    #[test]
    fn test_verdict_clone() {
        let verdict = Verdict::OutputMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
        };
        let cloned = verdict.clone();
        assert_eq!(cloned, verdict);
    }

    #[test]
    fn test_verification_result_debug() {
        let result = VerificationResult {
            source_code: "print(1)".to_string(),
            source_language: Language::Python,
            target_code: "fn main() {}".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Pass,
            source_result: None,
            target_result: None,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("VerificationResult"));
    }

    #[test]
    fn test_verification_result_clone() {
        let result = VerificationResult {
            source_code: "print(1)".to_string(),
            source_language: Language::Python,
            target_code: "fn main() {}".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Pass,
            source_result: None,
            target_result: None,
        };
        let cloned = result.clone();
        assert_eq!(cloned.source_code, result.source_code);
    }

    #[test]
    fn test_phase_debug() {
        let phase = Phase::Source;
        let debug = format!("{:?}", phase);
        assert!(debug.contains("Source"));
    }

    #[test]
    fn test_phase_clone() {
        let phase = Phase::Target;
        let cloned = phase.clone();
        assert_eq!(cloned, phase);
    }

    #[test]
    fn test_phase_copy() {
        let phase = Phase::Source;
        let copied = phase;
        assert_eq!(copied, Phase::Source);
    }
}
