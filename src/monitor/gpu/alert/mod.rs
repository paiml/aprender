//! GPU alert types and Andon system for monitoring.

mod system;
mod thresholds;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use system::GpuAndonSystem;
pub use thresholds::AndonThresholds;
pub use types::GpuAlert;
