#![allow(dead_code, unused_imports)]
//! Content Creation Tooling Module
//!
//! Implements the Content Creation Tooling Specification v1.1.0 for generating
//! structured prompts for educational and technical content.
//!
//! # Content Types
//! - HLO: High-Level Outline
//! - DLO: Detailed Outline
//! - BCH: Book Chapter (mdBook)
//! - BLP: Blog Post
//! - PDM: Presentar Demo
//!
//! # Toyota Way Integration
//! - Jidoka: LLM-as-a-Judge validation
//! - Poka-Yoke: Structural constraints in templates
//! - Genchi Genbutsu: Source context mandate
//! - Heijunka: Token budgeting
//! - Kaizen: Dynamic template composition

// Submodules
mod budget;
pub mod emitter;
mod errors;
mod source;
#[cfg(test)]
mod tests;
mod types;
mod validation;

// Re-exports
pub use budget::{ModelContext, TokenBudget};
pub use emitter::{EmitConfig, PromptEmitter};
pub use errors::ContentError;
pub use source::{SourceContext, SourceSnippet};
pub use types::{ContentType, CourseLevel};
pub use validation::{ContentValidator, ValidationResult, ValidationSeverity, ValidationViolation};
