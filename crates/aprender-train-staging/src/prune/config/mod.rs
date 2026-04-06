//! Pruning configuration module
//!
//! Provides configuration types for pruning methods, schedules,
//! and parameters.

mod method;
mod pattern;
mod pruning_config;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use method::PruneMethod;
pub use pattern::SparsityPatternConfig;
pub use pruning_config::PruningConfig;
