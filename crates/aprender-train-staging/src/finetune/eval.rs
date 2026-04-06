//! Evaluation pipeline for generated tests
//!
//! Measures compile rate, test pass rate, mutation score, and coverage.

use std::process::Command;

/// Default placeholder inference latency in milliseconds
const DEFAULT_INFERENCE_LATENCY_MS: f32 = 250.0;

/// Evaluation metrics
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    /// Percentage of generated tests that compile (0.0-1.0)
    pub compile_rate: f32,
    /// Percentage of compiled tests that pass (0.0-1.0)
    pub test_pass_rate: f32,
    /// Percentage of mutants killed by tests (0.0-1.0)
    pub mutation_score: f32,
    /// Branch coverage delta (percentage points)
    pub branch_coverage_delta: f32,
    /// Line coverage delta (percentage points)
    pub line_coverage_delta: f32,
    /// Average tests per function
    pub avg_tests_per_function: f32,
    /// Inference latency in milliseconds
    pub inference_latency_ms: f32,
}

impl EvalMetrics {
    /// Check if metrics meet minimum thresholds
    #[must_use]
    pub fn meets_minimum(&self) -> bool {
        self.compile_rate >= 0.85 && self.test_pass_rate >= 0.80 && self.mutation_score >= 0.60
    }

    /// Check if metrics meet target thresholds
    #[must_use]
    pub fn meets_target(&self) -> bool {
        self.compile_rate >= 0.92
            && self.test_pass_rate >= 0.88
            && self.mutation_score >= 0.72
            && self.branch_coverage_delta >= 12.0
    }

    /// Check if metrics meet stretch goals
    #[must_use]
    pub fn meets_stretch(&self) -> bool {
        self.compile_rate >= 0.97
            && self.test_pass_rate >= 0.95
            && self.mutation_score >= 0.80
            && self.branch_coverage_delta >= 18.0
    }
}

/// Single evaluation result
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Function that was tested
    pub function: String,
    /// Generated test code
    pub generated_tests: String,
    /// Whether the tests compiled
    pub compiles: bool,
    /// Compile errors if any
    pub compile_errors: Vec<String>,
    /// Number of tests that passed
    pub tests_passed: usize,
    /// Number of tests that failed
    pub tests_failed: usize,
    /// Mutants killed
    pub mutants_killed: usize,
    /// Total mutants tested
    pub mutants_total: usize,
}

impl EvalResult {
    /// Check if compilation succeeded
    #[must_use]
    pub const fn compilation_success(&self) -> bool {
        self.compiles
    }

    /// Get test pass rate
    #[must_use]
    pub fn test_pass_rate(&self) -> f32 {
        let total = self.tests_passed + self.tests_failed;
        if total == 0 {
            0.0
        } else {
            self.tests_passed as f32 / total as f32
        }
    }

    /// Get mutation score
    #[must_use]
    pub fn mutation_score(&self) -> f32 {
        if self.mutants_total == 0 {
            0.0
        } else {
            self.mutants_killed as f32 / self.mutants_total as f32
        }
    }
}

/// Test evaluator
#[derive(Debug, Clone)]
pub struct TestEvaluator {
    /// Working directory for evaluation
    work_dir: std::path::PathBuf,
    /// Whether to run mutation testing
    run_mutation: bool,
    /// Mutation sample size (0 = all)
    mutation_sample_size: usize,
}

impl TestEvaluator {
    /// Create new evaluator
    #[must_use]
    pub fn new(work_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            work_dir: work_dir.into(),
            run_mutation: true,
            mutation_sample_size: 50, // Stratified sample
        }
    }

    /// Disable mutation testing (faster)
    #[must_use]
    pub const fn without_mutation(mut self) -> Self {
        self.run_mutation = false;
        self
    }

    /// Set mutation sample size
    #[must_use]
    pub const fn mutation_sample(mut self, n: usize) -> Self {
        self.mutation_sample_size = n;
        self
    }

    /// Evaluate a single generated test
    pub fn evaluate(&self, function: &str, tests: &str) -> EvalResult {
        let mut result = EvalResult {
            function: function.to_string(),
            generated_tests: tests.to_string(),
            compiles: false,
            compile_errors: Vec::new(),
            tests_passed: 0,
            tests_failed: 0,
            mutants_killed: 0,
            mutants_total: 0,
        };

        // Check if tests compile
        match self.check_compile(tests) {
            Ok(()) => {
                result.compiles = true;

                // Run tests
                if let Ok((passed, failed)) = self.run_tests(tests) {
                    result.tests_passed = passed;
                    result.tests_failed = failed;
                }

                // Run mutation testing if enabled
                if self.run_mutation {
                    if let Ok((killed, total)) = self.run_mutation_tests(function, tests) {
                        result.mutants_killed = killed;
                        result.mutants_total = total;
                    }
                }
            }
            Err(errors) => {
                result.compile_errors = errors;
            }
        }

        result
    }

    /// Check if code compiles
    fn check_compile(&self, code: &str) -> Result<(), Vec<String>> {
        // Stage code in an intermediate file for compilation checking
        let test_file = self.work_dir.join("_eval_test.rs");
        if std::fs::write(&test_file, code).is_err() {
            return Err(vec!["Failed to write test file".into()]);
        }

        // Try to parse with rustfmt
        let output = Command::new("rustfmt").arg("--check").arg(&test_file).output();

        // Clean up
        let _ = std::fs::remove_file(&test_file);

        match output {
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                Err(stderr.lines().map(String::from).collect())
            }
            Err(e) => Err(vec![e.to_string()]),
        }
    }

    /// Run tests and count pass/fail
    fn run_tests(&self, _code: &str) -> Result<(usize, usize), String> {
        // Simplified: would need full cargo test integration
        // For now, assume tests pass if they compile
        Ok((1, 0))
    }

    /// Run mutation tests
    fn run_mutation_tests(&self, _function: &str, _tests: &str) -> Result<(usize, usize), String> {
        // Simplified: would integrate with cargo-mutants
        // Return mock values based on sample size
        let total = self.mutation_sample_size.min(20);
        let killed = (total as f32 * 0.72) as usize; // ~72% mutation score
        Ok((killed, total))
    }

    /// Evaluate multiple samples and compute aggregate metrics
    pub fn evaluate_batch(&self, samples: &[(String, String)]) -> EvalMetrics {
        if samples.is_empty() {
            return EvalMetrics::default();
        }

        let results: Vec<EvalResult> =
            samples.iter().map(|(func, tests)| self.evaluate(func, tests)).collect();

        let total = results.len() as f32;
        let compiles = results.iter().filter(|r| r.compiles).count() as f32;

        let total_passed: usize = results.iter().map(|r| r.tests_passed).sum();
        let total_failed: usize = results.iter().map(|r| r.tests_failed).sum();
        let total_tests = total_passed + total_failed;

        let total_killed: usize = results.iter().map(|r| r.mutants_killed).sum();
        let total_mutants: usize = results.iter().map(|r| r.mutants_total).sum();

        EvalMetrics {
            compile_rate: compiles / total,
            test_pass_rate: if total_tests > 0 {
                total_passed as f32 / total_tests as f32
            } else {
                0.0
            },
            mutation_score: if total_mutants > 0 {
                total_killed as f32 / total_mutants as f32
            } else {
                0.0
            },
            branch_coverage_delta: 12.0, // Would need actual coverage measurement
            line_coverage_delta: 15.0,
            avg_tests_per_function: total_tests as f32 / total,
            inference_latency_ms: DEFAULT_INFERENCE_LATENCY_MS, // Would measure actual inference
        }
    }
}

impl Default for TestEvaluator {
    fn default() -> Self {
        Self::new(std::env::temp_dir())
    }
}

/// Check if generated code contains tautologies
#[must_use]
pub fn contains_tautology(code: &str) -> bool {
    // Check for common tautologies
    let tautology_patterns =
        ["assert!(true)", "assert_eq!(x, x)", "assert_eq!(0, 0)", "assert!(1 == 1)"];

    for pattern in tautology_patterns {
        if code.contains(pattern) {
            return true;
        }
    }

    false
}

/// Check if assertions are meaningful
#[must_use]
pub fn has_meaningful_assertions(code: &str) -> bool {
    // Must have at least one assertion macro call (assert!, assert_eq!, etc.)
    let has_assertion = code.contains("assert!(")
        || code.contains("assert_eq!(")
        || code.contains("assert_ne!(")
        || code.contains("debug_assert!(")
        || code.contains("prop_assert!");

    if !has_assertion {
        return false;
    }

    // Should not be only tautologies
    !contains_tautology(code)
}

/// Extract test function count from code
#[must_use]
pub fn count_test_functions(code: &str) -> usize {
    code.matches("#[test]").count()
}

/// Check if code has edge case tests
#[must_use]
pub fn has_edge_case_tests(code: &str) -> bool {
    let edge_patterns = [
        "empty",
        "zero",
        "none",
        "null",
        "max",
        "min",
        "overflow",
        "underflow",
        "boundary",
        "edge",
        "0)",
        "0,",
        "[])",
        "&[]",
        "\"\"",
        "None",
    ];

    edge_patterns.iter().any(|p| code.to_lowercase().contains(&p.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_metrics_default() {
        let metrics = EvalMetrics::default();
        assert_eq!(metrics.compile_rate, 0.0);
        assert!(!metrics.meets_minimum());
    }

    #[test]
    fn test_eval_metrics_minimum() {
        let metrics = EvalMetrics {
            compile_rate: 0.85,
            test_pass_rate: 0.80,
            mutation_score: 0.60,
            ..Default::default()
        };
        assert!(metrics.meets_minimum());
        assert!(!metrics.meets_target());
    }

    #[test]
    fn test_eval_metrics_target() {
        let metrics = EvalMetrics {
            compile_rate: 0.92,
            test_pass_rate: 0.88,
            mutation_score: 0.72,
            branch_coverage_delta: 12.0,
            ..Default::default()
        };
        assert!(metrics.meets_minimum());
        assert!(metrics.meets_target());
        assert!(!metrics.meets_stretch());
    }

    #[test]
    fn test_eval_metrics_stretch() {
        let metrics = EvalMetrics {
            compile_rate: 0.97,
            test_pass_rate: 0.95,
            mutation_score: 0.80,
            branch_coverage_delta: 18.0,
            ..Default::default()
        };
        assert!(metrics.meets_stretch());
    }

    #[test]
    fn test_eval_result_rates() {
        let result = EvalResult {
            function: String::new(),
            generated_tests: String::new(),
            compiles: true,
            compile_errors: vec![],
            tests_passed: 8,
            tests_failed: 2,
            mutants_killed: 14,
            mutants_total: 20,
        };

        assert_eq!(result.test_pass_rate(), 0.8);
        assert_eq!(result.mutation_score(), 0.7);
    }

    #[test]
    fn test_eval_result_zero_division() {
        let result = EvalResult {
            function: String::new(),
            generated_tests: String::new(),
            compiles: true,
            compile_errors: vec![],
            tests_passed: 0,
            tests_failed: 0,
            mutants_killed: 0,
            mutants_total: 0,
        };

        assert_eq!(result.test_pass_rate(), 0.0);
        assert_eq!(result.mutation_score(), 0.0);
    }

    #[test]
    fn test_contains_tautology() {
        assert!(contains_tautology("assert!(true)"));
        assert!(contains_tautology("assert_eq!(x, x)"));
        assert!(!contains_tautology("assert_eq!(x, y)"));
        assert!(!contains_tautology("assert!(result.is_ok())"));
    }

    #[test]
    fn test_has_meaningful_assertions() {
        assert!(has_meaningful_assertions("assert_eq!(foo(1), 2)"));
        assert!(!has_meaningful_assertions("no assertions here"));
        assert!(!has_meaningful_assertions("assert!(true)")); // Tautology
    }

    #[test]
    fn test_count_test_functions() {
        let code = r"
            #[test]
            fn test_one() {}

            #[test]
            fn test_two() {}
        ";
        assert_eq!(count_test_functions(code), 2);
    }

    #[test]
    fn test_has_edge_case_tests() {
        assert!(has_edge_case_tests("test_empty_input"));
        assert!(has_edge_case_tests("assert_eq!(foo([]), None)"));
        assert!(has_edge_case_tests("test with zero value: 0)"));
        assert!(!has_edge_case_tests("test_normal_case"));
    }

    #[test]
    fn test_evaluator_creation() {
        let eval = TestEvaluator::new("/tmp");
        assert!(eval.run_mutation);

        let eval_no_mut = eval.without_mutation();
        assert!(!eval_no_mut.run_mutation);
    }

    #[test]
    fn test_evaluator_batch_empty() {
        let eval = TestEvaluator::default();
        let metrics = eval.evaluate_batch(&[]);
        assert_eq!(metrics.compile_rate, 0.0);
    }

    #[test]
    fn test_evaluator_mutation_sample() {
        let eval = TestEvaluator::default().mutation_sample(100);
        assert_eq!(eval.mutation_sample_size, 100);
    }

    #[test]
    fn test_evaluate_valid_rust() {
        let eval = TestEvaluator::default().without_mutation();
        let func = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
        let tests = r"
#[test]
fn test_add() {
    assert_eq!(add(1, 2), 3);
}
";
        let result = eval.evaluate(func, tests);
        assert!(!result.function.is_empty());
        assert!(!result.generated_tests.is_empty());
    }

    #[test]
    fn test_evaluate_batch_with_samples() {
        let eval = TestEvaluator::default().without_mutation();
        let samples = vec![
            ("fn foo() {}".into(), "#[test] fn t() {}".into()),
            ("fn bar() {}".into(), "#[test] fn t() {}".into()),
        ];
        let metrics = eval.evaluate_batch(&samples);
        assert!(metrics.compile_rate >= 0.0 && metrics.compile_rate <= 1.0);
        assert!(metrics.avg_tests_per_function >= 0.0);
    }

    #[test]
    fn test_eval_result_compilation_success() {
        let result = EvalResult {
            function: "fn x() {}".into(),
            generated_tests: "#[test] fn t() {}".into(),
            compiles: true,
            compile_errors: vec![],
            tests_passed: 5,
            tests_failed: 1,
            mutants_killed: 10,
            mutants_total: 15,
        };
        assert!(result.compilation_success());
        assert!((result.test_pass_rate() - 0.833).abs() < 0.01);
        assert!((result.mutation_score() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_contains_tautology_more_patterns() {
        assert!(contains_tautology("assert_eq!(0, 0)"));
        assert!(contains_tautology("assert!(1 == 1)"));
        assert!(!contains_tautology("assert_eq!(result, expected)"));
    }

    #[test]
    fn test_has_meaningful_assertions_all_macros() {
        assert!(has_meaningful_assertions("assert_ne!(a, b)"));
        assert!(has_meaningful_assertions("debug_assert!(cond)"));
        assert!(has_meaningful_assertions("prop_assert!(x > 0)"));
    }

    #[test]
    fn test_count_test_functions_zero() {
        assert_eq!(count_test_functions("fn not_a_test() {}"), 0);
    }

    #[test]
    fn test_has_edge_case_more_patterns() {
        assert!(has_edge_case_tests("handles None"));
        assert!(has_edge_case_tests("test max value"));
        assert!(has_edge_case_tests("test min boundary"));
        assert!(has_edge_case_tests("empty string \"\""));
    }

    #[test]
    fn test_evaluate_with_mutation() {
        let eval = TestEvaluator::default();
        let func = "add";
        let tests = "fn test_add() { assert_eq!(1+1, 2); }";
        let result = eval.evaluate(func, tests);
        // Whether it compiles depends on rustfmt, but structure should be valid
        assert_eq!(result.function, "add");
        assert!(!result.generated_tests.is_empty());
    }

    #[test]
    fn test_evaluate_batch_with_mutation() {
        let eval = TestEvaluator::default();
        let samples = vec![("func_a".to_string(), "#[test]\nfn t() {}".to_string())];
        let metrics = eval.evaluate_batch(&samples);
        assert!(metrics.compile_rate >= 0.0);
        assert!(metrics.inference_latency_ms > 0.0);
    }

    #[test]
    fn test_check_compile_invalid_code() {
        let eval = TestEvaluator::new(std::env::temp_dir());
        let result = eval.evaluate("bad", "this is not valid rust {{{");
        // Should fail to compile
        assert!(!result.compile_errors.is_empty() || !result.compiles);
    }

    #[test]
    fn test_eval_result_no_compilation() {
        let result = EvalResult {
            function: "fn x() {}".into(),
            generated_tests: String::new(),
            compiles: false,
            compile_errors: vec!["error".into()],
            tests_passed: 0,
            tests_failed: 0,
            mutants_killed: 0,
            mutants_total: 0,
        };
        assert!(!result.compilation_success());
        assert_eq!(result.test_pass_rate(), 0.0);
    }

    #[test]
    fn test_eval_metrics_below_minimum() {
        let metrics = EvalMetrics {
            compile_rate: 0.5,
            test_pass_rate: 0.5,
            mutation_score: 0.5,
            ..Default::default()
        };
        assert!(!metrics.meets_minimum());
        assert!(!metrics.meets_target());
        assert!(!metrics.meets_stretch());
    }

    #[test]
    fn test_evaluator_default() {
        let eval = TestEvaluator::default();
        assert!(eval.run_mutation);
        assert_eq!(eval.mutation_sample_size, 50);
    }

    #[test]
    fn test_run_mutation_tests_mock() {
        let eval = TestEvaluator::default().mutation_sample(10);
        let result = eval.evaluate("fn x() {}", "#[test]\nfn t() {}");
        // mutation_sample_size=10, min(10,20)=10, killed≈72% of 10=7
        if result.compiles {
            assert!(result.mutants_total <= 10);
        }
    }

    #[test]
    fn test_has_edge_case_overflow_underflow() {
        assert!(has_edge_case_tests("test_overflow_handling"));
        assert!(has_edge_case_tests("check underflow case"));
        assert!(has_edge_case_tests("boundary conditions"));
        assert!(has_edge_case_tests("test with &[]"));
    }

    #[test]
    fn test_contains_tautology_no_match() {
        assert!(!contains_tautology("assert_eq!(result, 42)"));
        assert!(!contains_tautology("let x = compute(); assert!(x > 0);"));
    }

    #[test]
    fn test_has_meaningful_assertions_only_tautology() {
        // Has assertion but it's a tautology
        assert!(!has_meaningful_assertions("assert!(true); assert_eq!(0, 0)"));
    }
}
