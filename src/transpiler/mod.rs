//! Transpiler integration
//!
//! This module provides the trait definition for transpilers and
//! integration with PAIML transpiler projects.
//!
//! # Supported Transpilers
//!
//! - **depyler**: Python → Rust
//! - **bashrs**: Bash → Safe Shell
//! - **ruchy**: Ruchy (standalone language)
//! - **decy**: C → Rust

mod ml_oracle;
mod oracle;

pub use ml_oracle::{
    BaselinePredictor, BugPredictor, CodeFeatures, ComplexityPredictor, TestPrioritizer,
};
pub use oracle::{TranspilerOracle, TranspilerVerdict, TranspilerVerification, VerificationStats};

use crate::grammar::Grammar;
use crate::{Language, Result};

/// Trait implemented by each transpiler
///
/// From spec Section 3.2: Contract enforcement via Rust type system.
pub trait Transpiler: Send + Sync {
    /// Source language identifier
    fn source_language(&self) -> Language;

    /// Target language identifier
    fn target_language(&self) -> Language;

    /// Transpile source to target
    ///
    /// # Errors
    ///
    /// Returns an error if transpilation fails
    fn transpile(&self, source: &str) -> Result<String>;

    /// Grammar for source language
    fn grammar(&self) -> &dyn Grammar;

    /// Get transpiler version
    fn version(&self) -> &str;
}

/// Configuration for transpiler testing
#[derive(Debug, Clone)]
pub struct TranspilerConfig {
    /// Name of the transpiler
    pub name: String,
    /// Source language
    pub source: Language,
    /// Target language
    pub target: Language,
    /// Enable strict mode
    pub strict: bool,
}

impl TranspilerConfig {
    /// Configuration for depyler (Python → Rust)
    #[must_use]
    pub fn depyler() -> Self {
        Self {
            name: "depyler".to_string(),
            source: Language::Python,
            target: Language::Rust,
            strict: true,
        }
    }

    /// Configuration for bashrs (Bash → Safe Shell)
    #[must_use]
    pub fn bashrs() -> Self {
        Self {
            name: "bashrs".to_string(),
            source: Language::Bash,
            target: Language::Rust,
            strict: true,
        }
    }

    /// Configuration for ruchy (standalone Ruchy language)
    #[must_use]
    pub fn ruchy() -> Self {
        Self {
            name: "ruchy".to_string(),
            source: Language::Ruchy,
            target: Language::Rust,
            strict: true,
        }
    }

    /// Configuration for decy (C → Rust)
    #[must_use]
    pub fn decy() -> Self {
        Self {
            name: "decy".to_string(),
            source: Language::C,
            target: Language::Rust,
            strict: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depyler_config() {
        let config = TranspilerConfig::depyler();
        assert_eq!(config.name, "depyler");
        assert_eq!(config.source, Language::Python);
        assert_eq!(config.target, Language::Rust);
        assert!(config.strict);
    }

    #[test]
    fn test_bashrs_config() {
        let config = TranspilerConfig::bashrs();
        assert_eq!(config.name, "bashrs");
        assert_eq!(config.source, Language::Bash);
        assert_eq!(config.target, Language::Rust);
        assert!(config.strict);
    }

    #[test]
    fn test_ruchy_config() {
        let config = TranspilerConfig::ruchy();
        assert_eq!(config.name, "ruchy");
        assert_eq!(config.source, Language::Ruchy);
        assert_eq!(config.target, Language::Rust);
        assert!(config.strict);
    }

    #[test]
    fn test_decy_config() {
        let config = TranspilerConfig::decy();
        assert_eq!(config.name, "decy");
        assert_eq!(config.source, Language::C);
        assert_eq!(config.target, Language::Rust);
        assert!(config.strict);
    }

    #[test]
    fn test_transpiler_config_debug() {
        let config = TranspilerConfig::depyler();
        let debug = format!("{:?}", config);
        assert!(debug.contains("TranspilerConfig"));
        assert!(debug.contains("depyler"));
    }

    #[test]
    fn test_transpiler_config_clone() {
        let config = TranspilerConfig::depyler();
        let cloned = config.clone();
        assert_eq!(cloned.name, config.name);
        assert_eq!(cloned.source, config.source);
        assert_eq!(cloned.target, config.target);
    }
}
