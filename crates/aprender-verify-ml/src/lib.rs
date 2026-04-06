//! Verificar - Synthetic Data Factory for Domain-Specific Code Intelligence
//!
//! Verificar is a unified combinatorial test generation and synthetic data factory
//! that serves multiple transpiler projects (depyler, bashrs, ruchy, decy). It generates
//! verified `(source, target, correctness)` tuples at scale, creating training data
//! for domain-specific code intelligence models.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                       VERIFICAR CORE                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Grammar    →   Generator   →   Mutator   →   Oracle       │
//! │  Definitions    Engine         Engine         Verification  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use verificar::generator::{Generator, SamplingStrategy};
//! use verificar::Language;
//!
//! // Create a generator for Python
//! let generator = Generator::new(Language::Python);
//!
//! // Generate test cases using coverage-guided sampling
//! let strategy = SamplingStrategy::CoverageGuided {
//!     coverage_map: None,
//!     max_depth: 3,
//!     seed: 42,
//! };
//! let test_cases = generator.generate(strategy, 100);
//! ```
//!
//! # Modules
//!
//! - [`grammar`] - Language grammar definitions (tree-sitter, pest PEGs)
//! - [`generator`] - Combinatorial program generation engine
//! - [`mutator`] - AST mutation operators (AOR, ROR, LOR, BSR, etc.)
//! - [`oracle`] - Verification oracle (sandbox execution, I/O diffing)
//! - [`data`] - Data pipeline (Parquet output)
//! - [`ml`] - ML model training (bug prediction, embeddings)

// Note: Lint configuration is in Cargo.toml [workspace.lints]
#![forbid(unsafe_code)]

pub mod audit;
pub mod data;
pub mod error;
pub mod generator;
pub mod grammar;
pub mod ml;
pub mod mutator;
pub mod oracle;
pub mod transpiler;

use serde::{Deserialize, Serialize};

pub use error::{Error, Result};

// Audit Trail exports (entrenar-style provenance)
pub use audit::{
    new_audit_collector, AuditCollector, AuditStats, AuditTrace, ChainVerification,
    ExecutionSummary, HashChainEntry, VerificationPath, VerificationTimer,
};

/// Supported source languages for generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// Python (depyler source → Rust)
    Python,
    /// Bash (bashrs source → Rust)
    Bash,
    /// C (decy source → Rust)
    C,
    /// TypeScript (decy target for C → TypeScript)
    TypeScript,
    /// Ruchy (standalone language)
    Ruchy,
    /// Rust (common target language)
    Rust,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Python => write!(f, "python"),
            Self::Bash => write!(f, "bash"),
            Self::C => write!(f, "c"),
            Self::TypeScript => write!(f, "typescript"),
            Self::Ruchy => write!(f, "ruchy"),
            Self::Rust => write!(f, "rust"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_display_python() {
        assert_eq!(format!("{}", Language::Python), "python");
    }

    #[test]
    fn test_language_display_bash() {
        assert_eq!(format!("{}", Language::Bash), "bash");
    }

    #[test]
    fn test_language_display_c() {
        assert_eq!(format!("{}", Language::C), "c");
    }

    #[test]
    fn test_language_display_ruchy() {
        assert_eq!(format!("{}", Language::Ruchy), "ruchy");
    }

    #[test]
    fn test_language_display_rust() {
        assert_eq!(format!("{}", Language::Rust), "rust");
    }

    #[test]
    fn test_language_display_typescript() {
        assert_eq!(format!("{}", Language::TypeScript), "typescript");
    }

    #[test]
    fn test_language_clone() {
        let lang = Language::Python;
        let cloned = lang.clone();
        assert_eq!(lang, cloned);
    }

    #[test]
    fn test_language_copy() {
        let lang = Language::Python;
        let copied = lang;
        assert_eq!(lang, copied);
    }

    #[test]
    fn test_language_debug() {
        let debug_str = format!("{:?}", Language::Python);
        assert!(debug_str.contains("Python"));
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::generator::{Generator, SamplingStrategy};
    pub use crate::mutator::{MutationOperator, Mutator};
    pub use crate::oracle::{Oracle, Verdict, VerificationResult};
    pub use crate::transpiler::{
        Transpiler, TranspilerConfig, TranspilerOracle, TranspilerVerdict, VerificationStats,
    };
    pub use crate::{Error, Language, Result};
}
