//! aprender-shell library
//!
//! AI-powered shell completion trained on your history.
//! This library exposes the core components for benchmarking and testing.

pub mod history;
pub mod model;
pub mod paged_model;
pub mod synthetic;
pub mod trie;

// Re-exports for convenience
pub use history::HistoryParser;
pub use model::MarkovModel;
pub use paged_model::PagedMarkovModel;
pub use synthetic::SyntheticPipeline;
