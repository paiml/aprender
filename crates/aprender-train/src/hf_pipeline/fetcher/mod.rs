//! HuggingFace Model Fetcher
//!
//! Downloads models from HuggingFace Hub with authentication and caching.

mod hf_fetcher;
mod options;
mod types;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;

// Re-export all public types for API compatibility
pub use hf_fetcher::HfModelFetcher;
pub use options::FetchOptions;
pub use types::{Architecture, ModelArtifact, WeightFormat};
