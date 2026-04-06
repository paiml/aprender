//! Decision pattern storage with hybrid retrieval
//!
//! Uses trueno-rag for BM25 lexical search combined with dense embeddings
//! and Reciprocal Rank Fusion (RRF) for optimal fix suggestions.
//!
//! # References
//! - Lewis et al. (2020): Retrieval-Augmented Generation
//! - Cormack et al. (2009): Reciprocal Rank Fusion

mod chunk_id;
mod config;
mod data;
mod fix_pattern;
mod store;
mod suggestion;

#[cfg(test)]
mod tests;

// Re-export all public types for API compatibility
pub use chunk_id::ChunkId;
pub use config::PatternStoreConfig;
pub use data::PatternStoreData;
pub use fix_pattern::FixPattern;
pub use store::DecisionPatternStore;
pub use suggestion::FixSuggestion;
