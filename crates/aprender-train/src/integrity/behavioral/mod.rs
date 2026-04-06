//! Behavioral Integrity Metrics (ENT-013)
//!
//! Provides behavioral integrity verification for ML model promotion gates.
//! Tracks metamorphic testing results, syscall patterns, timing variance,
//! and semantic equivalence to ensure model behavior consistency.

mod assessment;
mod builder;
mod counts;
mod metrics;
mod violation;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use assessment::IntegrityAssessment;
pub use builder::BehavioralIntegrityBuilder;
pub use counts::ViolationCounts;
pub use metrics::BehavioralIntegrity;
pub use violation::{MetamorphicRelationType, MetamorphicViolation};
