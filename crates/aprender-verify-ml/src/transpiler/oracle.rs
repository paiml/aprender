//! Transpiler verification oracle
//!
//! Provides end-to-end verification of transpiler correctness by:
//! 1. Generating test programs
//! 2. Transpiling source to target
//! 3. Executing both in sandboxed environments
//! 4. Comparing I/O behavior

use crate::generator::Generator;
use crate::oracle::{
    diff_results, DiffOptions, DiffResult, ExecutionResult, Executor, PythonExecutor, RustExecutor,
    SandboxedPythonExecutor,
};
use crate::Language;

use super::Transpiler;

/// Verification result for a single test case
#[derive(Debug, Clone)]
pub struct TranspilerVerification {
    /// Original source code
    pub source_code: String,
    /// Transpiled target code (if successful)
    pub target_code: Option<String>,
    /// Source execution result
    pub source_result: Option<ExecutionResult>,
    /// Target execution result
    pub target_result: Option<ExecutionResult>,
    /// Diff result comparing outputs
    pub diff: Option<DiffResult>,
    /// Overall verdict
    pub verdict: TranspilerVerdict,
    /// Test input used
    pub input: String,
}

/// Verdict from transpiler verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranspilerVerdict {
    /// I/O equivalent - transpilation is correct
    Pass,
    /// Transpilation failed
    TranspileError(String),
    /// Source execution failed
    SourceError(String),
    /// Target execution failed
    TargetError(String),
    /// Output mismatch
    OutputMismatch,
    /// Timeout during execution
    Timeout,
}

/// Statistics from verification run
#[derive(Debug, Clone, Default)]
pub struct VerificationStats {
    /// Total test cases run
    pub total: usize,
    /// Passed (I/O equivalent)
    pub passed: usize,
    /// Transpilation errors
    pub transpile_errors: usize,
    /// Source execution errors
    pub source_errors: usize,
    /// Target execution errors
    pub target_errors: usize,
    /// Output mismatches
    pub mismatches: usize,
    /// Timeouts
    pub timeouts: usize,
}

impl VerificationStats {
    /// Calculate pass rate as percentage
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.passed as f64 / self.total as f64) * 100.0
    }

    /// Calculate transpilation success rate
    #[must_use]
    pub fn transpile_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let successful = self.total - self.transpile_errors;
        (successful as f64 / self.total as f64) * 100.0
    }
}

/// Oracle for verifying transpiler correctness
pub struct TranspilerOracle<T: Transpiler> {
    transpiler: T,
    source_executor: Box<dyn Executor>,
    target_executor: Box<dyn Executor>,
    diff_options: DiffOptions,
    timeout_ms: u64,
    use_sandbox: bool,
}

impl<T: Transpiler> TranspilerOracle<T> {
    /// Create a new transpiler oracle
    pub fn new(transpiler: T) -> Self {
        let source_executor: Box<dyn Executor> = match transpiler.source_language() {
            Language::Python => Box::new(SandboxedPythonExecutor::new()),
            _ => Box::new(PythonExecutor::new()), // Fallback
        };

        let target_executor: Box<dyn Executor> = match transpiler.target_language() {
            Language::Python => Box::new(PythonExecutor::new()),
            // Default to Rust executor for Rust and other languages
            _ => Box::new(RustExecutor::new()),
        };

        Self {
            transpiler,
            source_executor,
            target_executor,
            diff_options: DiffOptions::default(),
            timeout_ms: 5000,
            use_sandbox: true,
        }
    }

    /// Set custom timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set diff options
    #[must_use]
    pub fn with_diff_options(mut self, options: DiffOptions) -> Self {
        self.diff_options = options;
        self
    }

    /// Disable sandboxing (use with caution)
    #[must_use]
    pub fn without_sandbox(mut self) -> Self {
        self.use_sandbox = false;
        if matches!(self.transpiler.source_language(), Language::Python) {
            self.source_executor = Box::new(PythonExecutor::new());
        }
        self
    }

    /// Verify a single source program
    pub fn verify(&self, source: &str, input: &str) -> TranspilerVerification {
        // Step 1: Transpile
        let target_code = match self.transpiler.transpile(source) {
            Ok(code) => code,
            Err(e) => {
                return TranspilerVerification {
                    source_code: source.to_string(),
                    target_code: None,
                    source_result: None,
                    target_result: None,
                    diff: None,
                    verdict: TranspilerVerdict::TranspileError(e.to_string()),
                    input: input.to_string(),
                };
            }
        };

        // Step 2: Execute source
        let source_result = match self.source_executor.execute(source, input, self.timeout_ms) {
            Ok(result) => result,
            Err(e) => {
                let msg = e.to_string();
                let verdict = if msg.contains("timeout") || msg.contains("timed out") {
                    TranspilerVerdict::Timeout
                } else {
                    TranspilerVerdict::SourceError(msg)
                };
                return TranspilerVerification {
                    source_code: source.to_string(),
                    target_code: Some(target_code),
                    source_result: None,
                    target_result: None,
                    diff: None,
                    verdict,
                    input: input.to_string(),
                };
            }
        };

        // Step 3: Execute target
        let target_result = match self
            .target_executor
            .execute(&target_code, input, self.timeout_ms)
        {
            Ok(result) => result,
            Err(e) => {
                let msg = e.to_string();
                let verdict = if msg.contains("timeout") || msg.contains("timed out") {
                    TranspilerVerdict::Timeout
                } else {
                    TranspilerVerdict::TargetError(msg)
                };
                return TranspilerVerification {
                    source_code: source.to_string(),
                    target_code: Some(target_code),
                    source_result: Some(source_result),
                    target_result: None,
                    diff: None,
                    verdict,
                    input: input.to_string(),
                };
            }
        };

        // Step 4: Compare outputs
        let diff = diff_results(&source_result, &target_result, &self.diff_options);
        let verdict = if diff.matches {
            TranspilerVerdict::Pass
        } else {
            TranspilerVerdict::OutputMismatch
        };

        TranspilerVerification {
            source_code: source.to_string(),
            target_code: Some(target_code),
            source_result: Some(source_result),
            target_result: Some(target_result),
            diff: Some(diff),
            verdict,
            input: input.to_string(),
        }
    }

    /// Verify multiple source programs
    pub fn verify_batch(
        &self,
        sources: &[(String, String)],
    ) -> (Vec<TranspilerVerification>, VerificationStats) {
        let mut results = Vec::with_capacity(sources.len());
        let mut stats = VerificationStats::default();

        for (source, input) in sources {
            let verification = self.verify(source, input);
            stats.total += 1;

            match &verification.verdict {
                TranspilerVerdict::Pass => stats.passed += 1,
                TranspilerVerdict::TranspileError(_) => stats.transpile_errors += 1,
                TranspilerVerdict::SourceError(_) => stats.source_errors += 1,
                TranspilerVerdict::TargetError(_) => stats.target_errors += 1,
                TranspilerVerdict::OutputMismatch => stats.mismatches += 1,
                TranspilerVerdict::Timeout => stats.timeouts += 1,
            }

            results.push(verification);
        }

        (results, stats)
    }

    /// Generate and verify programs
    pub fn verify_generated(
        &self,
        count: usize,
        max_depth: usize,
    ) -> (Vec<TranspilerVerification>, VerificationStats) {
        let generator = Generator::new(self.transpiler.source_language());
        let programs = generator.generate_exhaustive(max_depth);

        let sources: Vec<_> = programs
            .into_iter()
            .take(count)
            .map(|p| (p.code, String::new())) // Empty input for generated programs
            .collect();

        self.verify_batch(&sources)
    }

    /// Get reference to the transpiler
    #[must_use]
    pub fn transpiler(&self) -> &T {
        &self.transpiler
    }
}

impl<T: Transpiler> std::fmt::Debug for TranspilerOracle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranspilerOracle")
            .field("source_language", &self.transpiler.source_language())
            .field("target_language", &self.transpiler.target_language())
            .field("timeout_ms", &self.timeout_ms)
            .field("use_sandbox", &self.use_sandbox)
            .field("diff_options", &self.diff_options)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{grammar_for, Grammar};
    use crate::{Error, Result};

    /// Mock transpiler for testing
    struct MockTranspiler {
        source: Language,
        target: Language,
    }

    impl MockTranspiler {
        fn new(source: Language, target: Language) -> Self {
            Self { source, target }
        }
    }

    impl Transpiler for MockTranspiler {
        fn source_language(&self) -> Language {
            self.source
        }

        fn target_language(&self) -> Language {
            self.target
        }

        fn transpile(&self, source: &str) -> Result<String> {
            // Simple mock: Python print -> Rust println
            if source.contains("print") {
                // Extract the argument
                if let Some(start) = source.find("print(") {
                    let rest = &source[start + 6..];
                    if let Some(end) = rest.find(')') {
                        let arg = &rest[..end];
                        // Handle string literals
                        if arg.starts_with('\'') || arg.starts_with('"') {
                            let content = &arg[1..arg.len() - 1];
                            return Ok(format!("fn main() {{\n    println!(\"{}\");\n}}", content));
                        }
                        // Handle numeric expressions
                        return Ok(format!(
                            "fn main() {{\n    println!(\"{{}}\", {});\n}}",
                            arg
                        ));
                    }
                }
            }
            // Simple pass-through for other code
            Ok(format!("fn main() {{\n    // {}\n}}", source))
        }

        fn grammar(&self) -> &dyn Grammar {
            // Return a boxed grammar and leak it to get a static reference
            // This is acceptable in tests
            Box::leak(grammar_for(self.source))
        }

        fn version(&self) -> &str {
            "0.1.0-mock"
        }
    }

    #[test]
    fn test_verification_stats() {
        let stats = VerificationStats {
            total: 100,
            passed: 80,
            transpile_errors: 5,
            source_errors: 5,
            target_errors: 5,
            mismatches: 3,
            timeouts: 2,
        };

        assert!((stats.pass_rate() - 80.0).abs() < 0.001);
        assert!((stats.transpile_rate() - 95.0).abs() < 0.001);
    }

    #[test]
    fn test_verification_stats_empty() {
        let stats = VerificationStats::default();
        assert!((stats.pass_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_transpiler_verdict_eq() {
        assert_eq!(TranspilerVerdict::Pass, TranspilerVerdict::Pass);
        assert_ne!(TranspilerVerdict::Pass, TranspilerVerdict::OutputMismatch);
    }

    #[test]
    fn test_mock_transpiler() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        assert_eq!(transpiler.source_language(), Language::Python);
        assert_eq!(transpiler.target_language(), Language::Rust);

        let result = transpiler.transpile("print('hello')");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("println!"));
        assert!(code.contains("hello"));
    }

    #[test]
    fn test_transpiler_oracle_creation() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let oracle = TranspilerOracle::new(transpiler);

        assert_eq!(oracle.transpiler().source_language(), Language::Python);
        assert_eq!(oracle.transpiler().target_language(), Language::Rust);
    }

    #[test]
    fn test_transpiler_oracle_verify_transpile_error() {
        struct FailingTranspiler;

        impl Transpiler for FailingTranspiler {
            fn source_language(&self) -> Language {
                Language::Python
            }
            fn target_language(&self) -> Language {
                Language::Rust
            }
            fn transpile(&self, _source: &str) -> Result<String> {
                Err(Error::Transpile("Unsupported syntax".to_string()))
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "0.0.0"
            }
        }

        let oracle = TranspilerOracle::new(FailingTranspiler);
        let result = oracle.verify("invalid code", "");

        assert!(matches!(
            result.verdict,
            TranspilerVerdict::TranspileError(_)
        ));
    }

    #[test]
    fn test_transpiler_oracle_with_timeout() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let oracle = TranspilerOracle::new(transpiler).with_timeout(1000);
        assert_eq!(oracle.timeout_ms, 1000);
    }

    #[test]
    fn test_transpiler_oracle_with_diff_options() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let options = DiffOptions {
            normalize_whitespace: true,
            ignore_trailing_whitespace: true,
            ignore_case: false,
            float_tolerance: Some(0.001),
            ignore_stderr: false,
            ignore_exit_code: false,
        };
        let oracle = TranspilerOracle::new(transpiler).with_diff_options(options);
        assert!(oracle.diff_options.normalize_whitespace);
    }

    #[test]
    fn test_transpiler_oracle_without_sandbox() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let oracle = TranspilerOracle::new(transpiler).without_sandbox();
        assert!(!oracle.use_sandbox);
    }

    #[test]
    fn test_transpiler_oracle_debug() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let oracle = TranspilerOracle::new(transpiler);
        let debug = format!("{:?}", oracle);
        assert!(debug.contains("TranspilerOracle"));
        assert!(debug.contains("Python"));
        assert!(debug.contains("Rust"));
    }

    #[test]
    fn test_transpiler_verification_clone() {
        let verification = TranspilerVerification {
            source_code: "print(1)".to_string(),
            target_code: Some("fn main() {}".to_string()),
            source_result: None,
            target_result: None,
            diff: None,
            verdict: TranspilerVerdict::Pass,
            input: String::new(),
        };
        let cloned = verification.clone();
        assert_eq!(cloned.source_code, verification.source_code);
        assert_eq!(cloned.verdict, verification.verdict);
    }

    #[test]
    fn test_transpiler_verdict_debug() {
        let verdict = TranspilerVerdict::TranspileError("test error".to_string());
        let debug = format!("{:?}", verdict);
        assert!(debug.contains("TranspileError"));
    }

    #[test]
    fn test_transpiler_verdict_source_error() {
        let verdict = TranspilerVerdict::SourceError("runtime error".to_string());
        assert!(matches!(verdict, TranspilerVerdict::SourceError(_)));
    }

    #[test]
    fn test_transpiler_verdict_target_error() {
        let verdict = TranspilerVerdict::TargetError("compilation error".to_string());
        assert!(matches!(verdict, TranspilerVerdict::TargetError(_)));
    }

    #[test]
    fn test_transpiler_verdict_output_mismatch() {
        let verdict = TranspilerVerdict::OutputMismatch;
        assert_eq!(verdict, TranspilerVerdict::OutputMismatch);
    }

    #[test]
    fn test_transpiler_verdict_timeout() {
        let verdict = TranspilerVerdict::Timeout;
        assert_eq!(verdict, TranspilerVerdict::Timeout);
    }

    #[test]
    fn test_verification_stats_default() {
        let stats = VerificationStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.passed, 0);
        assert_eq!(stats.transpile_errors, 0);
    }

    #[test]
    fn test_verification_stats_transpile_rate() {
        let stats = VerificationStats {
            total: 100,
            passed: 80,
            transpile_errors: 10,
            source_errors: 5,
            target_errors: 3,
            mismatches: 1,
            timeouts: 1,
        };
        assert!((stats.transpile_rate() - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_verification_stats_debug() {
        let stats = VerificationStats {
            total: 10,
            passed: 8,
            transpile_errors: 1,
            source_errors: 0,
            target_errors: 0,
            mismatches: 1,
            timeouts: 0,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("VerificationStats"));
    }

    #[test]
    fn test_verification_stats_clone() {
        let stats = VerificationStats {
            total: 50,
            passed: 40,
            transpile_errors: 5,
            source_errors: 2,
            target_errors: 1,
            mismatches: 1,
            timeouts: 1,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total, stats.total);
        assert_eq!(cloned.passed, stats.passed);
    }

    #[test]
    fn test_transpiler_oracle_verify_batch_empty() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let oracle = TranspilerOracle::new(transpiler);
        let (results, stats) = oracle.verify_batch(&[]);
        assert!(results.is_empty());
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_mock_transpiler_numeric_expression() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let result = transpiler.transpile("print(1 + 2)");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("println!"));
        assert!(code.contains("1 + 2"));
    }

    #[test]
    fn test_mock_transpiler_non_print() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let result = transpiler.transpile("x = 5");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("fn main()"));
        assert!(code.contains("x = 5"));
    }

    #[test]
    fn test_mock_transpiler_version() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        assert_eq!(transpiler.version(), "0.1.0-mock");
    }

    #[test]
    fn test_mock_transpiler_grammar() {
        let transpiler = MockTranspiler::new(Language::Python, Language::Rust);
        let grammar = transpiler.grammar();
        assert_eq!(grammar.language(), Language::Python);
    }

    #[test]
    fn test_verification_stats_transpile_rate_zero_total() {
        let stats = VerificationStats::default();
        assert!((stats.transpile_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_transpiler_oracle_verify_pass() {
        use crate::oracle::PythonExecutor;
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        // Create a mock transpiler that produces working code
        struct PassingTranspiler;

        impl Transpiler for PassingTranspiler {
            fn source_language(&self) -> Language {
                Language::Python
            }
            fn target_language(&self) -> Language {
                Language::Python
            }
            fn transpile(&self, source: &str) -> Result<String> {
                // Just return the same code - it's Python to Python
                Ok(source.to_string())
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let oracle = TranspilerOracle::new(PassingTranspiler);
        let result = oracle.verify("print('hello')", "");

        assert_eq!(result.verdict, TranspilerVerdict::Pass);
        assert!(result.source_result.is_some());
        assert!(result.target_result.is_some());
        assert!(result.diff.is_some());
    }

    #[test]
    fn test_transpiler_oracle_verify_mismatch() {
        use crate::oracle::PythonExecutor;
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        // Create a transpiler that produces different output
        struct MismatchTranspiler;

        impl Transpiler for MismatchTranspiler {
            fn source_language(&self) -> Language {
                Language::Python
            }
            fn target_language(&self) -> Language {
                Language::Python
            }
            fn transpile(&self, _source: &str) -> Result<String> {
                // Always return code that prints 'goodbye'
                Ok("print('goodbye')".to_string())
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let oracle = TranspilerOracle::new(MismatchTranspiler);
        let result = oracle.verify("print('hello')", "");

        assert_eq!(result.verdict, TranspilerVerdict::OutputMismatch);
    }

    #[test]
    fn test_transpiler_oracle_verify_target_execution_error() {
        use crate::oracle::PythonExecutor;
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        // Transpiler that returns invalid Python for target
        struct TargetErrorTranspiler;

        impl Transpiler for TargetErrorTranspiler {
            fn source_language(&self) -> Language {
                Language::Python
            }
            fn target_language(&self) -> Language {
                Language::Python
            }
            fn transpile(&self, _source: &str) -> Result<String> {
                // Return code with a syntax error
                Ok("this is not valid python syntax !!@#$".to_string())
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let oracle = TranspilerOracle::new(TargetErrorTranspiler);
        let result = oracle.verify("print('hello')", "");

        // The source runs fine, but target fails to execute
        // Due to different error handling, may result in OutputMismatch or TargetError
        assert!(matches!(
            result.verdict,
            TranspilerVerdict::TargetError(_) | TranspilerVerdict::OutputMismatch
        ));
    }

    #[test]
    fn test_transpiler_oracle_verify_batch() {
        use crate::oracle::PythonExecutor;
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        // Simple pass-through transpiler
        struct PassthroughTranspiler;

        impl Transpiler for PassthroughTranspiler {
            fn source_language(&self) -> Language {
                Language::Python
            }
            fn target_language(&self) -> Language {
                Language::Python
            }
            fn transpile(&self, source: &str) -> Result<String> {
                Ok(source.to_string())
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let oracle = TranspilerOracle::new(PassthroughTranspiler);
        let sources = vec![
            ("print(1)".to_string(), String::new()),
            ("print(2)".to_string(), String::new()),
            ("print(3)".to_string(), String::new()),
        ];
        let (results, stats) = oracle.verify_batch(&sources);

        assert_eq!(results.len(), 3);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.passed, 3);
    }

    #[test]
    fn test_transpiler_oracle_verify_batch_mixed() {
        // Transpiler that fails on specific inputs
        struct MixedTranspiler;

        impl Transpiler for MixedTranspiler {
            fn source_language(&self) -> Language {
                Language::Python
            }
            fn target_language(&self) -> Language {
                Language::Python
            }
            fn transpile(&self, source: &str) -> Result<String> {
                if source.contains("FAIL") {
                    Err(Error::Transpile("intentional failure".to_string()))
                } else {
                    Ok(source.to_string())
                }
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let oracle = TranspilerOracle::new(MixedTranspiler);
        let sources = vec![
            ("print(1)".to_string(), String::new()),
            ("FAIL".to_string(), String::new()),
        ];
        let (results, stats) = oracle.verify_batch(&sources);

        assert_eq!(results.len(), 2);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.transpile_errors, 1);
    }

    #[test]
    fn test_transpiler_oracle_verify_generated() {
        use crate::oracle::PythonExecutor;
        let executor = PythonExecutor::new();
        if !executor.is_available() {
            eprintln!("Python not available, skipping test");
            return;
        }

        struct GeneratedTranspiler;

        impl Transpiler for GeneratedTranspiler {
            fn source_language(&self) -> Language {
                Language::Python
            }
            fn target_language(&self) -> Language {
                Language::Python
            }
            fn transpile(&self, source: &str) -> Result<String> {
                Ok(source.to_string())
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let oracle = TranspilerOracle::new(GeneratedTranspiler);
        let (results, stats) = oracle.verify_generated(5, 2);

        // Should have generated some test cases
        assert!(results.len() <= 5);
        assert!(stats.total <= 5);
    }

    #[test]
    fn test_transpiler_oracle_new_with_rust_source() {
        // Test creating oracle with Rust source language (falls back to PythonExecutor)
        struct RustSourceTranspiler;

        impl Transpiler for RustSourceTranspiler {
            fn source_language(&self) -> Language {
                Language::Rust
            }
            fn target_language(&self) -> Language {
                Language::Python
            }
            fn transpile(&self, _source: &str) -> Result<String> {
                Ok("print('hi')".to_string())
            }
            fn grammar(&self) -> &dyn Grammar {
                Box::leak(grammar_for(Language::Python))
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let oracle = TranspilerOracle::new(RustSourceTranspiler);
        assert_eq!(oracle.transpiler().source_language(), Language::Rust);
    }
}
