//! Compiler-in-the-Loop (CITL) training module
//!
//! This module provides:
//! - `DecisionPatternStore`: Stores and retrieves fix patterns using hybrid retrieval (BM25 + dense)
//! - `DecisionCITL`: Correlates compiler decisions with errors for fault localization
//!
//! # References
//! - Lewis et al. (2020): Retrieval-Augmented Generation
//! - Cormack et al. (2009): Reciprocal Rank Fusion
//! - Zeller (2002): Isolating Cause-Effect Chains
//! - Jones & Harrold (2005): Tarantula Fault Localization
//! - Chilimbi et al. (2009): HOLMES Statistical Debugging

mod pattern_store;
mod trainer;

#[cfg(test)]
mod tests;

pub use pattern_store::{
    ChunkId, DecisionPatternStore, FixPattern, FixSuggestion, PatternStoreConfig,
};
pub use trainer::{
    CITLConfig, CompilationOutcome, DecisionCITL, DecisionStats, DecisionTrace, ErrorCorrelation,
    SourceSpan, SuspiciousDecision,
};
