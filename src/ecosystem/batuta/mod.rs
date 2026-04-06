//! Batuta GPU Pricing and Queue Integration (ENT-030, ENT-031)
//!
//! Provides integration with Batuta for:
//! - Real-time GPU hourly rates
//! - Queue depth monitoring
//! - ETA adjustments based on queue state
//!
//! Falls back to static pricing when Batuta is unavailable.

mod client;
mod error;
mod pricing;
mod queue;

#[cfg(test)]
mod tests;

pub use client::BatutaClient;
pub use error::BatutaError;
pub use pricing::{FallbackPricing, GpuPricing};
pub use queue::{adjust_eta, QueueState};
