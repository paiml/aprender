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
/// Per `StepCoder` (Dou et al., 2024), categorizing errors by difficulty
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
/// Per `StepCoder`, starting with easy errors and progressing to hard
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
            PatternLibrary::load(&path).unwrap_or_default()
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

include!("citl_impl.rs");
