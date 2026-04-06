//! Compiler-in-the-Loop (CITL) trainer for error-fix correlation
//!
//! Correlates compiler decision traces with compilation outcomes
//! for fault localization using statistical debugging techniques.
//!
//! # References
//! - Zeller (2002): Isolating Cause-Effect Chains
//! - Jones & Harrold (2005): Tarantula Fault Localization
//! - Chilimbi et al. (2009): HOLMES Statistical Debugging

mod citl;
mod config;
mod correlation;
mod outcome;
mod span;
mod stats;
mod trace;

// Re-export all public types for API compatibility
pub use citl::DecisionCITL;
pub use config::CITLConfig;
pub use correlation::{ErrorCorrelation, SuspiciousDecision};
pub use outcome::CompilationOutcome;
pub use span::SourceSpan;
pub use stats::DecisionStats;
pub use trace::DecisionTrace;
