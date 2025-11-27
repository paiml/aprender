//! aprender-shell library
//!
//! AI-powered shell completion trained on your history.
//! This library exposes the core components for benchmarking and testing.
//!
//! # Hardening (v0.2.0)
//!
//! This version includes hardening per `aprender-shell-harden-plan.md`:
//! - **Error handling**: `ShellError` type with graceful degradation
//! - **Input validation**: `sanitize_prefix` for safe input handling
//! - **Security filtering**: `is_sensitive_command` for credential protection

pub mod config;
pub mod error;
pub mod history;
pub mod model;
pub mod paged_model;
pub mod quality;
pub mod security;
pub mod synthetic;
pub mod trie;
pub mod validation;

// Re-exports for convenience
pub use config::{suggest_with_fallback, ShellConfig};
pub use error::ShellError;
pub use history::HistoryParser;
pub use model::MarkovModel;
pub use paged_model::PagedMarkovModel;
pub use quality::{
    apply_typo_corrections, filter_quality_suggestions, suggestion_quality_score,
    validate_suggestion,
};
pub use security::{filter_sensitive_commands, filter_sensitive_suggestions, is_sensitive_command};
pub use synthetic::SyntheticPipeline;
pub use validation::{load_model_graceful, sanitize_prefix};
