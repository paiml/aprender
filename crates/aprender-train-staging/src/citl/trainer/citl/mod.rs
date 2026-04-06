//! Compiler-in-the-Loop (CITL) trainer for error-fix correlation
//!
//! Correlates compiler decision traces with compilation outcomes
//! for fault localization using statistical debugging techniques.
//!
//! # References
//! - Zeller (2002): Isolating Cause-Effect Chains
//! - Jones & Harrold (2005): Tarantula Fault Localization
//! - Chilimbi et al. (2009): HOLMES Statistical Debugging

mod core;
mod helpers;
mod types;

#[cfg(test)]
mod tests;

// Re-export the main type for API compatibility
pub use types::DecisionCITL;
