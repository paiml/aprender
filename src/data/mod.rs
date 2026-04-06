//! Data pipeline for storing verified test cases
//!
//! This module handles the storage and retrieval of verified
//! (source, target, correctness) tuples in Parquet format.
//!
//! # Features
//!
//! - Large-scale parallel generation with progress tracking
//! - Automatic Parquet sharding for large datasets
//! - Support for all sampling strategies

#[cfg(feature = "parquet")]
pub mod parquet;

pub mod corpus;
pub mod pipeline;

pub use corpus::{CorpusFormat, CorpusManager, CorpusMetadata, TrainingCorpus};
pub use pipeline::{DataPipeline, PipelineConfig, PipelineStats, PipelineStrategy};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::generator::GeneratedCode;
use crate::mutator::MutationOperator;
use crate::oracle::VerificationResult;
use crate::Language;

/// Verified transpilation tuple for ML training
///
/// Represents a successful transpilation with both source and target code.
/// Used for LLM fine-tuning with entrenar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedTuple {
    /// Source language
    pub source_language: Language,
    /// Target language
    pub target_language: Language,
    /// Original source code
    pub source_code: String,
    /// Transpiled target code
    pub target_code: String,
    /// Whether the transpilation was verified correct (I/O equivalent)
    pub is_correct: bool,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

impl VerifiedTuple {
    /// Create from a test case (only if transpilation succeeded)
    #[must_use]
    pub fn from_test_case(test_case: &TestCase) -> Option<Self> {
        let target_code = test_case.target_code.as_ref()?;
        Some(Self {
            source_language: test_case.source_language,
            target_language: test_case.target_language,
            source_code: test_case.source_code.clone(),
            target_code: target_code.clone(),
            is_correct: matches!(test_case.result, TestResult::Pass),
            execution_time_ms: 0, // Not tracked in TestCase
        })
    }
}

/// Test case with full metadata
///
/// From spec Section 8.1: Generated test case schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Unique identifier
    pub id: Uuid,

    /// Source language
    pub source_language: Language,

    /// Source code
    pub source_code: String,

    /// Target language
    pub target_language: Language,

    /// Transpiled code (if successful)
    pub target_code: Option<String>,

    /// Verification result
    pub result: TestResult,

    /// Features for ML
    pub features: CodeFeatures,

    /// Generation metadata
    pub metadata: GenerationMetadata,
}

/// Test result enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestResult {
    /// I/O equivalent
    Pass,
    /// Transpilation failed
    TranspileError(String),
    /// Output mismatch
    OutputMismatch {
        /// Expected output
        expected: String,
        /// Actual output
        actual: String,
    },
    /// Timeout
    Timeout {
        /// Timeout limit in milliseconds
        limit_ms: u64,
    },
    /// Runtime error
    RuntimeError {
        /// Phase where error occurred
        phase: String,
        /// Error message
        error: String,
    },
}

/// Features extracted from source code for ML
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeFeatures {
    /// AST depth
    pub ast_depth: u32,
    /// Number of operators
    pub num_operators: u32,
    /// Number of control flow statements
    pub num_control_flow: u32,
    /// Cyclomatic complexity
    pub cyclomatic_complexity: f32,
    /// Number of type coercions
    pub num_type_coercions: u32,
    /// Uses edge values (0, -1, MAX_INT, etc.)
    pub uses_edge_values: bool,
}

/// Metadata about how the test case was generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Generation strategy used
    pub strategy: String,
    /// Mutation operators applied
    pub mutation_operators: Vec<String>,
    /// Timestamp
    pub timestamp: String,
    /// Transpiler version
    pub transpiler_version: String,
}

impl TestCase {
    /// Create a new test case from generation and verification results
    #[must_use]
    pub fn new(
        generated: &GeneratedCode,
        verification: &VerificationResult,
        transpiler_version: &str,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_language: generated.language,
            source_code: generated.code.clone(),
            target_language: verification.target_language,
            target_code: Some(verification.target_code.clone()),
            result: TestResult::from_verification(verification),
            features: CodeFeatures {
                ast_depth: generated.ast_depth as u32,
                ..Default::default()
            },
            metadata: GenerationMetadata {
                strategy: "unknown".to_string(),
                mutation_operators: vec![],
                timestamp: chrono_lite_timestamp(),
                transpiler_version: transpiler_version.to_string(),
            },
        }
    }
}

impl TestResult {
    /// Convert from verification result
    fn from_verification(verification: &VerificationResult) -> Self {
        match &verification.verdict {
            crate::oracle::Verdict::Pass => Self::Pass,
            crate::oracle::Verdict::OutputMismatch { expected, actual } => Self::OutputMismatch {
                expected: expected.clone(),
                actual: actual.clone(),
            },
            crate::oracle::Verdict::Timeout { limit_ms, .. } => Self::Timeout {
                limit_ms: *limit_ms,
            },
            crate::oracle::Verdict::RuntimeError { phase, error } => Self::RuntimeError {
                phase: phase.to_string(),
                error: error.clone(),
            },
        }
    }
}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Builder for test cases with mutations
#[derive(Debug, Default)]
pub struct TestCaseBuilder {
    source_code: Option<String>,
    source_language: Option<Language>,
    mutation_operators: Vec<MutationOperator>,
    strategy: Option<String>,
}

impl TestCaseBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source code
    #[must_use]
    pub fn source_code(mut self, code: impl Into<String>) -> Self {
        self.source_code = Some(code.into());
        self
    }

    /// Set the source language
    #[must_use]
    pub fn source_language(mut self, language: Language) -> Self {
        self.source_language = Some(language);
        self
    }

    /// Add a mutation operator
    #[must_use]
    pub fn mutation_operator(mut self, operator: MutationOperator) -> Self {
        self.mutation_operators.push(operator);
        self
    }

    /// Set the generation strategy
    #[must_use]
    pub fn strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = Some(strategy.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::Verdict;

    #[test]
    fn test_test_result_pass() {
        let result = TestResult::Pass;
        assert_eq!(result, TestResult::Pass);
    }

    #[test]
    fn test_test_result_transpile_error() {
        let result = TestResult::TranspileError("syntax error".to_string());
        assert!(matches!(result, TestResult::TranspileError(_)));
    }

    #[test]
    fn test_test_result_output_mismatch() {
        let result = TestResult::OutputMismatch {
            expected: "hello".to_string(),
            actual: "world".to_string(),
        };
        assert!(matches!(result, TestResult::OutputMismatch { .. }));
    }

    #[test]
    fn test_test_result_timeout() {
        let result = TestResult::Timeout { limit_ms: 5000 };
        if let TestResult::Timeout { limit_ms } = result {
            assert_eq!(limit_ms, 5000);
        } else {
            panic!("Expected Timeout");
        }
    }

    #[test]
    fn test_test_result_runtime_error() {
        let result = TestResult::RuntimeError {
            phase: "source".to_string(),
            error: "division by zero".to_string(),
        };
        assert!(matches!(result, TestResult::RuntimeError { .. }));
    }

    #[test]
    fn test_code_features_default() {
        let features = CodeFeatures::default();
        assert_eq!(features.ast_depth, 0);
        assert_eq!(features.num_operators, 0);
        assert_eq!(features.num_control_flow, 0);
        assert!((features.cyclomatic_complexity - 0.0).abs() < f32::EPSILON);
        assert_eq!(features.num_type_coercions, 0);
        assert!(!features.uses_edge_values);
    }

    #[test]
    fn test_code_features_custom() {
        let features = CodeFeatures {
            ast_depth: 5,
            num_operators: 10,
            num_control_flow: 3,
            cyclomatic_complexity: 4.5,
            num_type_coercions: 2,
            uses_edge_values: true,
        };
        assert_eq!(features.ast_depth, 5);
        assert!(features.uses_edge_values);
    }

    #[test]
    fn test_test_case_builder() {
        let builder = TestCaseBuilder::new()
            .source_code("x = 1")
            .source_language(Language::Python)
            .mutation_operator(MutationOperator::Aor)
            .strategy("exhaustive");

        assert_eq!(builder.source_code, Some("x = 1".to_string()));
        assert_eq!(builder.source_language, Some(Language::Python));
        assert_eq!(builder.mutation_operators.len(), 1);
        assert_eq!(builder.strategy, Some("exhaustive".to_string()));
    }

    #[test]
    fn test_test_case_builder_multiple_operators() {
        let builder = TestCaseBuilder::new()
            .mutation_operator(MutationOperator::Aor)
            .mutation_operator(MutationOperator::Ror)
            .mutation_operator(MutationOperator::Lor);

        assert_eq!(builder.mutation_operators.len(), 3);
    }

    #[test]
    fn test_chrono_lite_timestamp() {
        let ts = chrono_lite_timestamp();
        // Should be a numeric string
        assert!(!ts.is_empty());
        assert!(ts.parse::<u64>().is_ok());
    }

    #[test]
    fn test_generation_metadata_debug() {
        let metadata = GenerationMetadata {
            strategy: "exhaustive".to_string(),
            mutation_operators: vec!["AOR".to_string()],
            timestamp: "123456".to_string(),
            transpiler_version: "0.1.0".to_string(),
        };
        let debug = format!("{:?}", metadata);
        assert!(debug.contains("exhaustive"));
    }

    #[test]
    fn test_test_result_from_verdict_pass() {
        let verification = crate::oracle::VerificationResult {
            source_code: "print(1)".to_string(),
            source_language: Language::Python,
            target_code: "fn main() {}".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Pass,
            source_result: None,
            target_result: None,
        };
        let result = TestResult::from_verification(&verification);
        assert_eq!(result, TestResult::Pass);
    }

    #[test]
    fn test_test_result_from_verdict_mismatch() {
        use crate::oracle::Phase;
        let verification = crate::oracle::VerificationResult {
            source_code: "print(1)".to_string(),
            source_language: Language::Python,
            target_code: "fn main() {}".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::OutputMismatch {
                expected: "1".to_string(),
                actual: "2".to_string(),
            },
            source_result: None,
            target_result: None,
        };
        let result = TestResult::from_verification(&verification);
        assert!(matches!(result, TestResult::OutputMismatch { .. }));
    }

    #[test]
    fn test_test_result_from_verdict_timeout() {
        use crate::oracle::Phase;
        let verification = crate::oracle::VerificationResult {
            source_code: "while True: pass".to_string(),
            source_language: Language::Python,
            target_code: "loop {}".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Timeout {
                phase: Phase::Source,
                limit_ms: 5000,
            },
            source_result: None,
            target_result: None,
        };
        let result = TestResult::from_verification(&verification);
        assert!(matches!(result, TestResult::Timeout { .. }));
    }

    #[test]
    fn test_test_result_from_verdict_runtime_error() {
        use crate::oracle::Phase;
        let verification = crate::oracle::VerificationResult {
            source_code: "1/0".to_string(),
            source_language: Language::Python,
            target_code: "panic!()".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::RuntimeError {
                phase: Phase::Source,
                error: "division by zero".to_string(),
            },
            source_result: None,
            target_result: None,
        };
        let result = TestResult::from_verification(&verification);
        assert!(matches!(result, TestResult::RuntimeError { .. }));
    }

    #[test]
    fn test_test_case_new() {
        let generated = crate::generator::GeneratedCode {
            code: "print(1)".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec!["print".to_string()],
        };

        let verification = crate::oracle::VerificationResult {
            source_code: "print(1)".to_string(),
            source_language: Language::Python,
            target_code: "fn main() { println!(\"1\"); }".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Pass,
            source_result: None,
            target_result: None,
        };

        let test_case = TestCase::new(&generated, &verification, "0.1.0");

        assert_eq!(test_case.source_language, Language::Python);
        assert_eq!(test_case.target_language, Language::Rust);
        assert!(test_case.target_code.is_some());
        assert_eq!(test_case.result, TestResult::Pass);
        assert_eq!(test_case.metadata.transpiler_version, "0.1.0");
    }

    #[test]
    fn test_test_case_debug() {
        let generated = crate::generator::GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };

        let verification = crate::oracle::VerificationResult {
            source_code: "x = 1".to_string(),
            source_language: Language::Python,
            target_code: "let x = 1;".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Pass,
            source_result: None,
            target_result: None,
        };

        let test_case = TestCase::new(&generated, &verification, "0.1.0");
        let debug = format!("{:?}", test_case);
        assert!(debug.contains("TestCase"));
    }

    #[test]
    fn test_test_case_clone() {
        let generated = crate::generator::GeneratedCode {
            code: "x = 1".to_string(),
            language: Language::Python,
            ast_depth: 1,
            features: vec![],
        };

        let verification = crate::oracle::VerificationResult {
            source_code: "x = 1".to_string(),
            source_language: Language::Python,
            target_code: "let x = 1;".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Pass,
            source_result: None,
            target_result: None,
        };

        let test_case = TestCase::new(&generated, &verification, "0.1.0");
        let cloned = test_case.clone();
        assert_eq!(cloned.source_code, test_case.source_code);
    }
}
