//! Auto-Retraining Module (APR-073-5)
//!
//! Implements the Andon Cord pattern for automated retraining when drift is detected.
//! Bridges drift detection to the training loop following Toyota Way principles.

mod action;
mod config;
mod policy;
mod retrainer;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use action::Action;
pub use config::RetrainConfig;
pub use policy::RetrainPolicy;
pub use retrainer::{AutoRetrainer, RetrainCallback, RetrainerStats};
