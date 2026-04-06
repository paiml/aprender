//! Curriculum Learning for progressive training
//!
//! Implements curriculum learning strategies as described in:
//! - Bengio et al. (2009) "Curriculum Learning"
//!
//! This module provides schedulers that progressively adjust training
//! difficulty, data selection, and sample weighting.
//!
//! # CITL Support
//!
//! Designed to support Compiler-in-the-Loop (CITL) training where:
//! - Diagnostic verbosity increases with model maturity
//! - Rare error classes (long-tail) get appropriate attention
//! - Training efficiency is balanced against corpus size

mod adaptive;
mod efficiency;
mod linear;
mod scheduler;
mod tiered;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use adaptive::AdaptiveCurriculum;
pub use efficiency::{efficiency_score, select_optimal_tier};
pub use linear::LinearCurriculum;
pub use scheduler::CurriculumScheduler;
pub use tiered::TieredCurriculum;
