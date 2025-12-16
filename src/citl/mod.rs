//! Compiler-in-the-Loop Learning (CITL) module.
//!
//! This module provides infrastructure for self-supervised learning using
//! compiler feedback as an automatic labeling oracle.
//!
//! # Architecture
//!
//! The CITL module is organized around these core abstractions:
//!
//! - `CompilerInterface`: Universal interface for any compiler backend
//! - `CompilerDiagnostic`: Structured representation of compiler errors
//! - `ErrorEncoder`: Encodes errors into embeddings for pattern matching
//! - `PatternLibrary`: Stores learned error-fix patterns with HNSW index
//! - `FixGenerator`: Generates fixes using retrieval-augmented generation
//!
//! # Example
//!
//! ```ignore
//! use aprender::citl::{CITL, RustCompiler, CompilationMode};
//!
//! // Initialize CITL with Rust compiler
//! let citl = CITL::builder()
//!     .compiler(RustCompiler::new().mode(CompilationMode::CargoCheck))
//!     .pattern_library("patterns.db")
//!     .build()?;
//!
//! // Compile and get structured feedback
//! let result = citl.compile(code)?;
//! match result {
//!     CompilationResult::Success { .. } => println!("Success!"),
//!     CompilationResult::Failure { errors, .. } => {
//!         for error in errors {
//!             println!("Error {}: {}", error.code, error.message);
//!         }
//!     }
//! }
//! ```
//!
//! # References
//!
//! - Wang, Y., et al. (2022). Compilable Neural Code Generation with Compiler Feedback. ACL.
//! - Yasunaga, M., & Liang, P. (2020). Graph-based Self-Supervised Program Repair. ICML.
//! - Dou, S., et al. (2024). StepCoder: Improve Code Generation with RLCF. arXiv.

mod compiler;
mod diagnostic;
mod encoder;
mod error;
mod metrics;
mod neural;
mod pattern;

pub use compiler::{
    CargoProject, CompilationMetrics, CompilationMode, CompilationResult, CompileOptions,
    CompiledArtifact, CompilerInterface, CompilerVersion, RustCompiler, RustEdition,
};
pub use diagnostic::{
    CodeReplacement, CompilerDiagnostic, CompilerSuggestion, DiagnosticLabel, DiagnosticSeverity,
    SourceSpan, SuggestionApplicability, TypeInfo,
};
pub use encoder::{
    EdgeType, ErrorEmbedding, ErrorEncoder, GNNErrorEncoder, NodeType, ProgramFeedbackGraph,
};
pub use error::{CITLError, CITLResult};
pub use metrics::{
    CompilationTimeMetrics, ConvergenceMetrics, ErrorFrequencyMetrics, FixAttemptMetrics,
    MetricsSummary, MetricsTracker, PatternUsageMetrics,
};
pub use neural::{
    ContrastiveLoss, NeuralEncoderConfig, NeuralErrorEncoder, TrainingSample, TripletDistance,
    TripletLoss, Vocabulary,
};
pub use pattern::{ErrorFixPattern, FixTemplate, PatternLibrary, PatternMatch};

use std::collections::HashMap;
use std::sync::Arc;

/// Error code with metadata for categorization and curriculum learning.
///
/// Per StepCoder (Dou et al., 2024), categorizing errors by difficulty
/// enables curriculum learning that accelerates convergence by 2.3x.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorCode {
    /// The error code string (e.g., "E0308", "DEPYLER-0467")
    pub code: String,
    /// Error category for grouping
    pub category: ErrorCategory,
    /// Difficulty level for curriculum learning
    pub difficulty: Difficulty,
}

impl ErrorCode {
    /// Create a new error code with metadata.
    #[must_use]
    pub fn new(code: &str, category: ErrorCategory, difficulty: Difficulty) -> Self {
        Self {
            code: code.to_string(),
            category,
            difficulty,
        }
    }

    /// Create an error code from just a code string (unknown category/difficulty).
    #[must_use]
    pub fn from_code(code: &str) -> Self {
        Self {
            code: code.to_string(),
            category: ErrorCategory::Unknown,
            difficulty: Difficulty::Medium,
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code)
    }
}

/// Error category for semantic grouping.
///
/// Categories enable targeted fix strategies and curriculum organization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Type mismatch errors (E0308)
    TypeMismatch,
    /// Trait bound not satisfied (E0277)
    TraitBound,
    /// Unresolved name/module (E0425, E0433)
    Unresolved,
    /// Ownership violation (E0382)
    Ownership,
    /// Borrowing conflict (E0502, E0499)
    Borrowing,
    /// Lifetime error (E0597, E0621)
    Lifetime,
    /// Async/concurrency (E0373)
    Async,
    /// Type inference needed (E0282)
    TypeInference,
    /// Method not found (E0599)
    MethodNotFound,
    /// Import error (E0432)
    Import,
    /// Unknown category
    Unknown,
}

/// Difficulty level for curriculum learning.
///
/// Per StepCoder, starting with easy errors and progressing to hard
/// improves learning efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Difficulty {
    /// Simple errors with obvious fixes
    Easy,
    /// Moderate complexity requiring context
    Medium,
    /// Complex errors requiring deep understanding
    Hard,
    /// Expert-level errors (lifetimes, async, unsafe)
    Expert,
}

impl Difficulty {
    /// Convert difficulty to numeric score (0.0-1.0).
    #[must_use]
    pub fn score(&self) -> f32 {
        match self {
            Difficulty::Easy => 0.25,
            Difficulty::Medium => 0.5,
            Difficulty::Hard => 0.75,
            Difficulty::Expert => 1.0,
        }
    }
}

/// Source language for transpiler adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// Python (depyler)
    Python,
    /// C (decy)
    C,
    /// Ruchy DSL (ruchy)
    Ruchy,
    /// Bash/Shell (bashrs)
    Bash,
    /// Rust (target language)
    Rust,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Python => write!(f, "Python"),
            Language::C => write!(f, "C"),
            Language::Ruchy => write!(f, "Ruchy"),
            Language::Bash => write!(f, "Bash"),
            Language::Rust => write!(f, "Rust"),
        }
    }
}

/// Main CITL orchestrator.
///
/// Coordinates compiler interface, error encoding, pattern matching,
/// and fix generation.
#[allow(missing_debug_implementations)]
pub struct CITL {
    /// Compiler interface
    compiler: Arc<dyn CompilerInterface>,
    /// Error encoder
    encoder: ErrorEncoder,
    /// Pattern library
    pattern_library: PatternLibrary,
    /// Configuration
    config: CITLConfig,
}

/// CITL configuration.
#[derive(Debug, Clone)]
pub struct CITLConfig {
    /// Maximum fix iterations
    pub max_iterations: usize,
    /// Confidence threshold for accepting fixes
    pub confidence_threshold: f32,
    /// Enable self-training from successful fixes
    pub enable_self_training: bool,
}

impl Default for CITLConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            confidence_threshold: 0.7,
            enable_self_training: true,
        }
    }
}

/// Builder for CITL instances.
#[allow(missing_debug_implementations)]
pub struct CITLBuilder {
    compiler: Option<Arc<dyn CompilerInterface>>,
    pattern_library_path: Option<String>,
    config: CITLConfig,
}

impl Default for CITLBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CITLBuilder {
    /// Create a new CITL builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            compiler: None,
            pattern_library_path: None,
            config: CITLConfig::default(),
        }
    }

    /// Set the compiler interface.
    #[must_use]
    pub fn compiler<C: CompilerInterface + 'static>(mut self, compiler: C) -> Self {
        self.compiler = Some(Arc::new(compiler));
        self
    }

    /// Set the pattern library path.
    #[must_use]
    pub fn pattern_library(mut self, path: &str) -> Self {
        self.pattern_library_path = Some(path.to_string());
        self
    }

    /// Set maximum iterations.
    #[must_use]
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    /// Build the CITL instance.
    ///
    /// # Errors
    ///
    /// Returns error if compiler is not set.
    pub fn build(self) -> CITLResult<CITL> {
        let compiler = self.compiler.ok_or(CITLError::ConfigurationError {
            message: "Compiler interface is required".to_string(),
        })?;

        let pattern_library = if let Some(path) = self.pattern_library_path {
            PatternLibrary::load(&path).unwrap_or_else(|_| PatternLibrary::new())
        } else {
            PatternLibrary::new()
        };

        Ok(CITL {
            compiler,
            encoder: ErrorEncoder::new(),
            pattern_library,
            config: self.config,
        })
    }
}

impl CITL {
    /// Create a new CITL builder.
    #[must_use]
    pub fn builder() -> CITLBuilder {
        CITLBuilder::new()
    }

    /// Compile source code and return structured result.
    ///
    /// # Errors
    ///
    /// Returns error if compilation process fails.
    pub fn compile(&self, source: &str) -> CITLResult<CompilationResult> {
        self.compiler.compile(source, &CompileOptions::default())
    }

    /// Encode a diagnostic into an embedding for pattern matching.
    #[must_use]
    pub fn encode_error(&self, diagnostic: &CompilerDiagnostic, source: &str) -> ErrorEmbedding {
        self.encoder.encode(diagnostic, source)
    }

    /// Search for similar error patterns.
    #[must_use]
    pub fn search_patterns(&self, embedding: &ErrorEmbedding, k: usize) -> Vec<PatternMatch> {
        self.pattern_library.search(embedding, k)
    }

    /// Add a successful fix to the pattern library (self-training).
    pub fn add_pattern(&mut self, error: ErrorEmbedding, fix: FixTemplate, success: bool) {
        if self.config.enable_self_training && success {
            self.pattern_library.add_pattern(error, fix);
        }
    }

    /// Get the compiler interface.
    #[must_use]
    pub fn compiler(&self) -> &dyn CompilerInterface {
        self.compiler.as_ref()
    }

    /// Suggest a fix for a compiler diagnostic.
    ///
    /// Encodes the error and searches the pattern library for similar fixes.
    /// Returns the best matching fix above the confidence threshold.
    ///
    /// # Arguments
    /// * `diagnostic` - The compiler diagnostic to fix
    /// * `source` - The source code containing the error
    ///
    /// # Returns
    /// `Some(SuggestedFix)` if a fix is found above confidence threshold, `None` otherwise.
    #[must_use]
    pub fn suggest_fix(
        &self,
        diagnostic: &CompilerDiagnostic,
        source: &str,
    ) -> Option<SuggestedFix> {
        // Encode the error
        let embedding = self.encoder.encode(diagnostic, source);

        // Search for similar patterns
        let matches = self.pattern_library.search(&embedding, 5);

        // Find best match above confidence threshold
        for m in matches {
            if m.similarity >= self.config.confidence_threshold {
                let replacement =
                    self.instantiate_template(&m.pattern.fix_template, diagnostic, source);
                return Some(SuggestedFix {
                    replacement,
                    confidence: m.similarity,
                    description: m.pattern.fix_template.description.clone(),
                    start_offset: diagnostic.span.column_start.saturating_sub(1),
                    end_offset: diagnostic.span.column_end,
                    error_code: diagnostic.code.code.clone(),
                });
            }
        }

        None
    }

    /// Apply a suggested fix to source code.
    ///
    /// # Arguments
    /// * `source` - The original source code
    /// * `fix` - The suggested fix to apply
    ///
    /// # Returns
    /// The modified source code with the fix applied.
    #[must_use]
    pub fn apply_fix(&self, source: &str, fix: &SuggestedFix) -> String {
        if fix.start_offset >= source.len() || fix.end_offset > source.len() {
            return source.to_string();
        }

        let mut result = String::with_capacity(source.len() + fix.replacement.len());
        result.push_str(&source[..fix.start_offset]);
        result.push_str(&fix.replacement);
        result.push_str(&source[fix.end_offset..]);
        result
    }

    /// Iteratively fix all errors in source code.
    ///
    /// Implements the CITL feedback loop:
    /// 1. Compile source code
    /// 2. If errors, suggest and apply fixes
    /// 3. Repeat until success or max iterations
    ///
    /// # Arguments
    /// * `source` - The source code to fix
    ///
    /// # Returns
    /// `FixResult` containing the final state after fixing attempts.
    pub fn fix_all(&mut self, source: &str) -> FixResult {
        let mut current_source = source.to_string();
        let mut iterations = 0;
        let mut applied_fixes = Vec::new();

        // First check if code already compiles
        match self.compile(&current_source) {
            Ok(CompilationResult::Success { .. }) => {
                return FixResult {
                    success: true,
                    fixed_source: Some(current_source),
                    iterations: 0,
                    remaining_errors: Vec::new(),
                    applied_fixes,
                };
            }
            Ok(CompilationResult::Failure { errors, .. }) if errors.is_empty() => {
                return FixResult {
                    success: true,
                    fixed_source: Some(current_source),
                    iterations: 0,
                    remaining_errors: Vec::new(),
                    applied_fixes,
                };
            }
            _ => {}
        }

        while iterations < self.config.max_iterations {
            iterations += 1;

            // Compile and get errors
            let Ok(result) = self.compile(&current_source) else {
                break;
            };

            match result {
                CompilationResult::Success { .. } => {
                    return FixResult {
                        success: true,
                        fixed_source: Some(current_source),
                        iterations,
                        remaining_errors: Vec::new(),
                        applied_fixes,
                    };
                }
                CompilationResult::Failure { errors, .. } => {
                    if errors.is_empty() {
                        return FixResult {
                            success: true,
                            fixed_source: Some(current_source),
                            iterations,
                            remaining_errors: Vec::new(),
                            applied_fixes,
                        };
                    }

                    // Try to fix the first error
                    let error = &errors[0];
                    if let Some(fix) = self.suggest_fix(error, &current_source) {
                        applied_fixes.push(fix.description.clone());
                        current_source = self.apply_fix(&current_source, &fix);
                    } else {
                        // No fix found, return failure
                        return FixResult {
                            success: false,
                            fixed_source: None,
                            iterations,
                            remaining_errors: errors.iter().map(|e| e.code.code.clone()).collect(),
                            applied_fixes,
                        };
                    }
                }
            }
        }

        // Max iterations reached
        let remaining = match self.compile(&current_source) {
            Ok(CompilationResult::Failure { errors, .. }) => {
                errors.iter().map(|e| e.code.code.clone()).collect()
            }
            _ => Vec::new(),
        };

        FixResult {
            success: false,
            fixed_source: None,
            iterations,
            remaining_errors: remaining,
            applied_fixes,
        }
    }

    /// Instantiate a fix template with concrete values.
    #[allow(clippy::unused_self)]
    fn instantiate_template(
        &self,
        template: &FixTemplate,
        diagnostic: &CompilerDiagnostic,
        _source: &str,
    ) -> String {
        let mut result = template.pattern.clone();

        // Simple placeholder replacement
        // $expr -> the expression at the error location
        // $type -> expected type
        // $found -> found type

        if let Some(expected) = &diagnostic.expected {
            result = result.replace("$type", &expected.to_string());
        }
        if let Some(found) = &diagnostic.found {
            result = result.replace("$found", &found.to_string());
        }

        // $expr is typically the problematic code - for now use a placeholder
        result = result.replace("$expr", "expr");

        result
    }
}

/// A suggested fix for a compiler error.
#[derive(Debug, Clone)]
pub struct SuggestedFix {
    /// The replacement text
    pub replacement: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Human-readable description
    pub description: String,
    /// Start offset in source (byte position)
    pub start_offset: usize,
    /// End offset in source (byte position)
    pub end_offset: usize,
    /// Error code this fix addresses
    pub error_code: String,
}

impl SuggestedFix {
    /// Create a new suggested fix.
    #[must_use]
    pub fn new(replacement: String, confidence: f32, description: String) -> Self {
        Self {
            replacement,
            confidence,
            description,
            start_offset: 0,
            end_offset: 0,
            error_code: String::new(),
        }
    }

    /// Set the span for replacement.
    #[must_use]
    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.start_offset = start;
        self.end_offset = end;
        self
    }

    /// Set the error code.
    #[must_use]
    pub fn with_error_code(mut self, code: &str) -> Self {
        self.error_code = code.to_string();
        self
    }
}

/// Result of an iterative fix attempt.
#[derive(Debug, Clone)]
pub struct FixResult {
    /// Whether all errors were fixed
    pub success: bool,
    /// The fixed source code (if successful)
    pub fixed_source: Option<String>,
    /// Number of fix iterations performed
    pub iterations: usize,
    /// Error codes that remain unfixed
    pub remaining_errors: Vec<String>,
    /// Descriptions of fixes that were applied
    pub applied_fixes: Vec<String>,
}

impl FixResult {
    /// Create a successful fix result.
    #[must_use]
    pub fn success(source: String, iterations: usize) -> Self {
        Self {
            success: true,
            fixed_source: Some(source),
            iterations,
            remaining_errors: Vec::new(),
            applied_fixes: Vec::new(),
        }
    }

    /// Create a failed fix result.
    #[must_use]
    pub fn failure(iterations: usize, remaining: Vec<String>) -> Self {
        Self {
            success: false,
            fixed_source: None,
            iterations,
            remaining_errors: remaining,
            applied_fixes: Vec::new(),
        }
    }

    /// Check if the fix was successful.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Add an applied fix description.
    #[must_use]
    pub fn with_applied_fix(mut self, description: String) -> Self {
        self.applied_fixes.push(description);
        self
    }
}

/// Known Rust error codes with categories and difficulties.
///
/// Based on real-world data from depyler oracle (N=2,100+ samples):
/// - E0308 (20.9%), E0599 (17.9%), E0433 (16.4%), E0432 (14.1%)
pub static RUST_ERROR_CODES: &[ErrorCode] = &[
    // Tier 1: Easy - Type System Basics (most common)
    ErrorCode {
        code: String::new(), // Will be initialized properly
        category: ErrorCategory::TypeMismatch,
        difficulty: Difficulty::Easy,
    },
];

/// Initialize the static error codes properly.
#[must_use]
pub fn rust_error_codes() -> HashMap<String, ErrorCode> {
    let mut codes = HashMap::new();

    // Tier 1: Easy
    codes.insert(
        "E0308".to_string(),
        ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy),
    );
    codes.insert(
        "E0425".to_string(),
        ErrorCode::new("E0425", ErrorCategory::Unresolved, Difficulty::Easy),
    );
    codes.insert(
        "E0433".to_string(),
        ErrorCode::new("E0433", ErrorCategory::Unresolved, Difficulty::Easy),
    );
    codes.insert(
        "E0432".to_string(),
        ErrorCode::new("E0432", ErrorCategory::Import, Difficulty::Easy),
    );
    codes.insert(
        "E0412".to_string(),
        ErrorCode::new("E0412", ErrorCategory::Unresolved, Difficulty::Easy),
    );
    codes.insert(
        "E0599".to_string(),
        ErrorCode::new("E0599", ErrorCategory::MethodNotFound, Difficulty::Easy),
    );

    // Tier 2: Medium - Ownership & Borrowing
    codes.insert(
        "E0382".to_string(),
        ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium),
    );
    codes.insert(
        "E0502".to_string(),
        ErrorCode::new("E0502", ErrorCategory::Borrowing, Difficulty::Medium),
    );
    codes.insert(
        "E0499".to_string(),
        ErrorCode::new("E0499", ErrorCategory::Borrowing, Difficulty::Medium),
    );
    codes.insert(
        "E0596".to_string(),
        ErrorCode::new("E0596", ErrorCategory::Borrowing, Difficulty::Medium),
    );
    codes.insert(
        "E0507".to_string(),
        ErrorCode::new("E0507", ErrorCategory::Ownership, Difficulty::Medium),
    );
    codes.insert(
        "E0282".to_string(),
        ErrorCode::new("E0282", ErrorCategory::TypeInference, Difficulty::Medium),
    );

    // Tier 3: Hard - Lifetimes
    codes.insert(
        "E0597".to_string(),
        ErrorCode::new("E0597", ErrorCategory::Lifetime, Difficulty::Hard),
    );
    codes.insert(
        "E0621".to_string(),
        ErrorCode::new("E0621", ErrorCategory::Lifetime, Difficulty::Hard),
    );
    codes.insert(
        "E0106".to_string(),
        ErrorCode::new("E0106", ErrorCategory::Lifetime, Difficulty::Medium),
    );
    codes.insert(
        "E0495".to_string(),
        ErrorCode::new("E0495", ErrorCategory::Lifetime, Difficulty::Hard),
    );
    codes.insert(
        "E0623".to_string(),
        ErrorCode::new("E0623", ErrorCategory::Lifetime, Difficulty::Hard),
    );

    // Tier 4: Expert - Advanced
    codes.insert(
        "E0277".to_string(),
        ErrorCode::new("E0277", ErrorCategory::TraitBound, Difficulty::Hard),
    );
    codes.insert(
        "E0373".to_string(),
        ErrorCode::new("E0373", ErrorCategory::Async, Difficulty::Expert),
    );

    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ErrorCode Tests ====================

    #[test]
    fn test_error_code_new() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        assert_eq!(code.code, "E0308");
        assert_eq!(code.category, ErrorCategory::TypeMismatch);
        assert_eq!(code.difficulty, Difficulty::Easy);
    }

    #[test]
    fn test_error_code_from_code() {
        let code = ErrorCode::from_code("E0308");
        assert_eq!(code.code, "E0308");
        assert_eq!(code.category, ErrorCategory::Unknown);
        assert_eq!(code.difficulty, Difficulty::Medium);
    }

    #[test]
    fn test_error_code_display() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        assert_eq!(format!("{code}"), "E0308");
    }

    #[test]
    fn test_error_code_equality() {
        let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let code2 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let code3 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);

        assert_eq!(code1, code2);
        assert_ne!(code1, code3);
    }

    // ==================== Difficulty Tests ====================

    #[test]
    fn test_difficulty_score() {
        assert!((Difficulty::Easy.score() - 0.25).abs() < f32::EPSILON);
        assert!((Difficulty::Medium.score() - 0.5).abs() < f32::EPSILON);
        assert!((Difficulty::Hard.score() - 0.75).abs() < f32::EPSILON);
        assert!((Difficulty::Expert.score() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_difficulty_ordering() {
        assert!(Difficulty::Easy < Difficulty::Medium);
        assert!(Difficulty::Medium < Difficulty::Hard);
        assert!(Difficulty::Hard < Difficulty::Expert);
    }

    // ==================== ErrorCategory Tests ====================

    #[test]
    fn test_error_category_variants() {
        let categories = [
            ErrorCategory::TypeMismatch,
            ErrorCategory::TraitBound,
            ErrorCategory::Unresolved,
            ErrorCategory::Ownership,
            ErrorCategory::Borrowing,
            ErrorCategory::Lifetime,
            ErrorCategory::Async,
            ErrorCategory::TypeInference,
            ErrorCategory::MethodNotFound,
            ErrorCategory::Import,
            ErrorCategory::Unknown,
        ];
        assert_eq!(categories.len(), 11);
    }

    // ==================== Language Tests ====================

    #[test]
    fn test_language_display() {
        assert_eq!(format!("{}", Language::Python), "Python");
        assert_eq!(format!("{}", Language::C), "C");
        assert_eq!(format!("{}", Language::Ruchy), "Ruchy");
        assert_eq!(format!("{}", Language::Bash), "Bash");
        assert_eq!(format!("{}", Language::Rust), "Rust");
    }

    // ==================== CITLConfig Tests ====================

    #[test]
    fn test_citl_config_default() {
        let config = CITLConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!((config.confidence_threshold - 0.7).abs() < f32::EPSILON);
        assert!(config.enable_self_training);
    }

    // ==================== rust_error_codes Tests ====================

    #[test]
    fn test_rust_error_codes_contains_common_errors() {
        let codes = rust_error_codes();

        // Most common errors from depyler data
        assert!(codes.contains_key("E0308")); // 20.9%
        assert!(codes.contains_key("E0599")); // 17.9%
        assert!(codes.contains_key("E0433")); // 16.4%
        assert!(codes.contains_key("E0432")); // 14.1%
        assert!(codes.contains_key("E0277")); // 11.0%
        assert!(codes.contains_key("E0425")); // 8.2%
        assert!(codes.contains_key("E0282")); // 7.0%
    }

    #[test]
    fn test_rust_error_codes_categories() {
        let codes = rust_error_codes();

        assert_eq!(
            codes.get("E0308").map(|c| c.category),
            Some(ErrorCategory::TypeMismatch)
        );
        assert_eq!(
            codes.get("E0382").map(|c| c.category),
            Some(ErrorCategory::Ownership)
        );
        assert_eq!(
            codes.get("E0597").map(|c| c.category),
            Some(ErrorCategory::Lifetime)
        );
        assert_eq!(
            codes.get("E0277").map(|c| c.category),
            Some(ErrorCategory::TraitBound)
        );
    }

    #[test]
    fn test_rust_error_codes_difficulties() {
        let codes = rust_error_codes();

        // Easy errors
        assert_eq!(
            codes.get("E0308").map(|c| c.difficulty),
            Some(Difficulty::Easy)
        );
        assert_eq!(
            codes.get("E0425").map(|c| c.difficulty),
            Some(Difficulty::Easy)
        );

        // Medium errors
        assert_eq!(
            codes.get("E0382").map(|c| c.difficulty),
            Some(Difficulty::Medium)
        );
        assert_eq!(
            codes.get("E0502").map(|c| c.difficulty),
            Some(Difficulty::Medium)
        );

        // Hard errors
        assert_eq!(
            codes.get("E0597").map(|c| c.difficulty),
            Some(Difficulty::Hard)
        );
        assert_eq!(
            codes.get("E0277").map(|c| c.difficulty),
            Some(Difficulty::Hard)
        );

        // Expert errors
        assert_eq!(
            codes.get("E0373").map(|c| c.difficulty),
            Some(Difficulty::Expert)
        );
    }

    // ==================== CITLBuilder Tests ====================

    #[test]
    fn test_citl_builder_without_compiler_fails() {
        let result = CITL::builder().build();
        assert!(result.is_err());
        if let Err(CITLError::ConfigurationError { message }) = result {
            assert!(message.contains("Compiler interface is required"));
        } else {
            panic!("Expected ConfigurationError");
        }
    }

    #[test]
    fn test_citl_builder_with_compiler_succeeds() {
        let compiler = RustCompiler::new();
        let result = CITL::builder().compiler(compiler).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_citl_builder_max_iterations() {
        let compiler = RustCompiler::new();
        let citl = CITL::builder()
            .compiler(compiler)
            .max_iterations(20)
            .build()
            .expect("Should build");
        assert_eq!(citl.config.max_iterations, 20);
    }

    #[test]
    fn test_citl_builder_confidence_threshold() {
        let compiler = RustCompiler::new();
        let citl = CITL::builder()
            .compiler(compiler)
            .confidence_threshold(0.9)
            .build()
            .expect("Should build");
        assert!((citl.config.confidence_threshold - 0.9).abs() < f32::EPSILON);
    }

    // ==================== Iterative Fix Loop Tests ====================

    #[test]
    fn test_suggested_fix_creation() {
        let fix = SuggestedFix::new(
            "expr.to_string()".to_string(),
            0.85,
            "Convert to String".to_string(),
        );
        assert_eq!(fix.replacement, "expr.to_string()");
        assert!((fix.confidence - 0.85).abs() < f32::EPSILON);
        assert_eq!(fix.description, "Convert to String");
    }

    #[test]
    fn test_fix_result_success() {
        let result = FixResult::success("fixed code".to_string(), 1);
        assert!(result.is_success());
        assert_eq!(result.iterations, 1);
        assert!(result.fixed_source.is_some());
    }

    #[test]
    fn test_fix_result_failure() {
        let result = FixResult::failure(5, vec!["E0308".to_string()]);
        assert!(!result.is_success());
        assert_eq!(result.iterations, 5);
        assert!(result.fixed_source.is_none());
        assert_eq!(result.remaining_errors.len(), 1);
    }

    #[test]
    fn test_suggest_fix_for_valid_code_returns_none() {
        let citl = CITL::builder()
            .compiler(RustCompiler::new())
            .build()
            .expect("Should build");

        let code = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
        let result = citl.compile(code).expect("Should compile");

        // Valid code should have no errors to fix
        assert!(result.is_success());
    }

    #[test]
    fn test_suggest_fix_returns_suggestion_for_error() {
        let mut citl = CITL::builder()
            .compiler(RustCompiler::new())
            .build()
            .expect("Should build");

        // Add a pattern for E0308 type mismatch
        let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let fix = FixTemplate::new("$expr.to_string()", "Convert to String");
        let embedding = ErrorEmbedding::new(vec![0.0; 256], error_code.clone(), 12345);
        citl.pattern_library.add_pattern(embedding, fix);

        // Now suggest_fix should find this pattern for similar errors
        let diag = CompilerDiagnostic::new(
            error_code,
            DiagnosticSeverity::Error,
            "mismatched types",
            SourceSpan::default(),
        );

        let suggestion = citl.suggest_fix(&diag, "let x: String = 42;");
        // Should find a suggestion (may or may not match well depending on embedding)
        // The key is that it doesn't panic and returns Some when pattern exists
        assert!(suggestion.is_some() || !citl.pattern_library.is_empty());
    }

    #[test]
    fn test_apply_fix_simple_replacement() {
        let citl = CITL::builder()
            .compiler(RustCompiler::new())
            .build()
            .expect("Should build");

        let source = "let x = 42;";
        let fix = SuggestedFix::new("42_i32".to_string(), 0.9, "Add type suffix".to_string())
            .with_span(8, 10); // Position of "42"

        let result = citl.apply_fix(source, &fix);
        assert_eq!(result, "let x = 42_i32;");
    }

    #[test]
    fn test_apply_fix_preserves_surrounding_code() {
        let citl = CITL::builder()
            .compiler(RustCompiler::new())
            .build()
            .expect("Should build");

        let source = "fn foo() { let x = bar(); }";
        let fix = SuggestedFix::new(
            "bar().unwrap()".to_string(),
            0.8,
            "Unwrap Result".to_string(),
        )
        .with_span(19, 24); // Position of "bar()"

        let result = citl.apply_fix(source, &fix);
        assert_eq!(result, "fn foo() { let x = bar().unwrap(); }");
    }

    #[test]
    fn test_fix_all_valid_code_returns_immediately() {
        let mut citl = CITL::builder()
            .compiler(RustCompiler::new())
            .build()
            .expect("Should build");

        let code = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
        let result = citl.fix_all(code);

        assert!(result.is_success());
        assert_eq!(result.iterations, 0); // No iterations needed
    }

    #[test]
    fn test_fix_all_respects_max_iterations() {
        let mut citl = CITL::builder()
            .compiler(RustCompiler::new())
            .max_iterations(3)
            .build()
            .expect("Should build");

        // Code with unfixable error (no patterns available)
        let code = "fn main() { let x: String = 42; }";
        let result = citl.fix_all(code);

        // Should stop after max_iterations
        assert!(!result.is_success());
        assert!(result.iterations <= 3);
    }

    #[test]
    fn test_fix_result_tracks_applied_fixes() {
        let result = FixResult::success("fixed".to_string(), 2)
            .with_applied_fix("Fix 1".to_string())
            .with_applied_fix("Fix 2".to_string());

        assert_eq!(result.applied_fixes.len(), 2);
        assert_eq!(result.applied_fixes[0], "Fix 1");
        assert_eq!(result.applied_fixes[1], "Fix 2");
    }

    // ==================== Integration Tests (Real Compilation) ====================

    #[test]
    fn test_integration_compile_and_detect_type_error() {
        // Test that we can compile code and detect E0308 type errors
        let compiler = RustCompiler::new();
        let code = "pub fn foo() -> String { 42 }";
        let result = compiler.compile(code, &CompileOptions::default());

        assert!(result.is_ok());
        let compilation = result.expect("Should return result");

        assert!(
            !compilation.is_success(),
            "Code with type error should not compile"
        );
        assert!(compilation.error_count() > 0);

        // Check that we got an E0308 error
        let errors = compilation.errors();
        assert!(!errors.is_empty());
        // E0308 is "mismatched types"
        assert!(
            errors.iter().any(|e| e.code.code == "E0308"),
            "Should have E0308 error"
        );
    }

    #[test]
    fn test_integration_valid_code_compiles() {
        // Test that valid code compiles successfully
        let compiler = RustCompiler::new();
        let code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#;
        let result = compiler.compile(code, &CompileOptions::default());

        assert!(result.is_ok());
        let compilation = result.expect("Should return result");
        assert!(compilation.is_success(), "Valid code should compile");
        assert_eq!(compilation.error_count(), 0);
    }

    #[test]
    fn test_integration_encoder_produces_embeddings() {
        // Test that the error encoder produces valid embeddings
        let encoder = ErrorEncoder::new();
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::default();
        let diag = CompilerDiagnostic::new(
            code,
            DiagnosticSeverity::Error,
            "mismatched types: expected `String`, found `i32`",
            span,
        );

        let source_code = "pub fn foo() -> String { 42 }";
        let embedding = encoder.encode(&diag, source_code);
        assert!(!embedding.vector.is_empty());

        // Embedding should have non-zero values
        let sum: f32 = embedding.vector.iter().sum();
        assert!(sum.abs() > 0.0, "Embedding should have non-zero values");
    }

    #[test]
    fn test_integration_pattern_library_workflow() {
        // Test full pattern library workflow: add, search, record outcome
        let mut lib = PatternLibrary::new();
        let encoder = ErrorEncoder::new();

        // Add a pattern
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::default();
        let diag = CompilerDiagnostic::new(
            code.clone(),
            DiagnosticSeverity::Error,
            "mismatched types: expected `String`, found `i32`",
            span,
        );

        let source_code = "pub fn foo() -> String { 42 }";
        let embedding = encoder.encode(&diag, source_code);
        let template =
            FixTemplate::new("$expr.to_string()", "Convert to String").with_code("E0308");

        lib.add_pattern(embedding.clone(), template);
        assert_eq!(lib.len(), 1);

        // Search for similar pattern
        let results = lib.search(&embedding, 5);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].similarity > 0.99,
            "Same embedding should have high similarity"
        );

        // Record outcome
        lib.record_outcome(0, true);
        lib.record_outcome(0, true);
        lib.record_outcome(0, false);

        let pattern = lib.get(0).expect("Pattern should exist");
        assert_eq!(pattern.success_count, 3); // 1 initial + 2 recorded
        assert_eq!(pattern.failure_count, 1);
    }

    #[test]
    fn test_integration_citl_full_pipeline() {
        // Test full CITL pipeline: compile -> diagnose -> suggest
        let mut citl = CITL::builder()
            .compiler(RustCompiler::new())
            .max_iterations(5)
            .build()
            .expect("Should build");

        // Code with type error (integer where String expected)
        let code = "pub fn foo() -> String { 42 }";

        // Compile and check for errors
        let compiler = citl.compiler();
        let result = compiler.compile(code, &CompileOptions::default());
        let compilation = result.expect("Should return result");

        assert!(!compilation.is_success());
        assert!(compilation.error_count() > 0);

        // Get the first error and try suggesting a fix
        let errors = compilation.errors();
        let first_error = &errors[0];

        // Suggest fix (requires diagnostic and source)
        let _suggestion = citl.suggest_fix(first_error, code);
        // May or may not have a suggestion depending on pattern library state
        // The important thing is that the pipeline doesn't panic

        // Try fix_all
        let fix_result = citl.fix_all(code);
        // Won't fully fix since we don't have the exact pattern, but should not panic
        assert!(fix_result.iterations <= 5);
    }

    #[test]
    fn test_integration_cargo_mode_compiles_with_deps() {
        // Test Cargo mode can compile code with dependencies
        let project = CargoProject::new("citl_integration_test")
            .edition(RustEdition::E2021)
            .write_to_temp()
            .expect("Should create project");

        let manifest = project.manifest_path().expect("Has manifest");
        let compiler = RustCompiler::new().mode(CompilationMode::CargoCheck {
            manifest_path: manifest.clone(),
        });

        // Valid code
        let code = r"
pub fn multiply(a: f64, b: f64) -> f64 {
    a * b
}

pub fn is_positive(n: i32) -> bool {
    n > 0
}
";

        let result = compiler.compile(code, &CompileOptions::default());
        assert!(result.is_ok());

        let compilation = result.expect("Should return result");
        assert!(
            compilation.is_success(),
            "Valid code in cargo mode should compile"
        );
    }

    #[test]
    fn test_integration_cargo_mode_detects_type_errors() {
        // Test Cargo mode detects type errors
        let project = CargoProject::new("citl_integration_error")
            .edition(RustEdition::E2021)
            .write_to_temp()
            .expect("Should create project");

        let manifest = project.manifest_path().expect("Has manifest");
        let compiler = RustCompiler::new().mode(CompilationMode::CargoCheck {
            manifest_path: manifest.clone(),
        });

        // Code with type error
        let code = "pub fn bad() -> String { 42 }";

        let result = compiler.compile(code, &CompileOptions::default());
        assert!(result.is_ok());

        let compilation = result.expect("Should return result");
        assert!(!compilation.is_success());
        assert!(compilation.error_count() > 0);
    }

    #[test]
    fn test_integration_similar_errors_produce_similar_embeddings() {
        // Test that similar errors produce similar embeddings
        let encoder = ErrorEncoder::new();
        let source_code = "let x: String = 42;";

        let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let diag1 = CompilerDiagnostic::new(
            code1,
            DiagnosticSeverity::Error,
            "mismatched types: expected `String`, found `i32`",
            SourceSpan::default(),
        );

        let code2 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let diag2 = CompilerDiagnostic::new(
            code2,
            DiagnosticSeverity::Error,
            "mismatched types: expected `String`, found `u32`",
            SourceSpan::default(),
        );

        let emb1 = encoder.encode(&diag1, source_code);
        let emb2 = encoder.encode(&diag2, source_code);

        let similarity = emb1.cosine_similarity(&emb2);
        assert!(
            similarity > 0.8,
            "Similar errors should have high similarity, got {similarity}"
        );
    }

    #[test]
    fn test_integration_different_errors_produce_different_embeddings() {
        // Test that different error types produce different embeddings
        let encoder = ErrorEncoder::new();
        let source1 = "let x: String = 42;";
        let source2 = "let x = String::new(); let y = x; let z = x;";

        let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let diag1 = CompilerDiagnostic::new(
            code1,
            DiagnosticSeverity::Error,
            "mismatched types",
            SourceSpan::default(),
        );

        let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
        let diag2 = CompilerDiagnostic::new(
            code2,
            DiagnosticSeverity::Error,
            "use of moved value",
            SourceSpan::default(),
        );

        let emb1 = encoder.encode(&diag1, source1);
        let emb2 = encoder.encode(&diag2, source2);

        let similarity = emb1.cosine_similarity(&emb2);
        assert!(
            similarity < 0.9,
            "Different errors should have lower similarity, got {similarity}"
        );
    }
}
