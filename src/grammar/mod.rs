//! Grammar definitions for source languages
//!
//! This module provides grammar definitions for each supported source language.
//! Grammars can be defined using tree-sitter or pest PEGs.
//!
//! # Supported Languages
//!
//! - Python (for depyler → Rust)
//! - Bash (for bashrs → Rust)
//! - C (for decy → Rust)
//! - TypeScript (for decy → TypeScript target)
//! - Ruchy (standalone language)

mod bash;
mod c;
mod python;
mod ruchy;
mod typescript;

pub use bash::BashGrammar;
pub use c::CGrammar;
pub use python::PythonGrammar;
pub use ruchy::RuchyGrammar;
pub use typescript::TypeScriptGrammar;

use crate::Language;

/// Trait for language grammar definitions
pub trait Grammar: Send + Sync + std::fmt::Debug {
    /// Get the language this grammar defines
    fn language(&self) -> Language;

    /// Validate that a code string conforms to the grammar
    fn validate(&self, code: &str) -> bool;

    /// Get the maximum depth for enumeration
    fn max_enumeration_depth(&self) -> usize {
        5 // Default per spec - combinatorial explosion at depth 5-6
    }
}

/// Create a grammar for the specified language
#[must_use]
pub fn grammar_for(language: Language) -> Box<dyn Grammar> {
    match language {
        Language::Python => Box::new(PythonGrammar::new()),
        Language::Bash => Box::new(BashGrammar::new()),
        Language::C => Box::new(CGrammar::new()),
        Language::TypeScript => Box::new(TypeScriptGrammar::new()),
        Language::Ruchy => Box::new(RuchyGrammar::new()),
        Language::Rust => todo!("Rust grammar not yet implemented"),
    }
}
