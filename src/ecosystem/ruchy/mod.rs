//! Ruchy Session Bridge (ENT-033)
//!
//! Provides session bridge for preserving training history from Ruchy
//! interactive sessions into Entrenar artifacts.
//!
//! This module is feature-gated behind `ruchy-sessions`.

mod artifact;
mod error;
mod metrics;
mod session;
mod types;

#[cfg(test)]
mod tests;

// Re-export public API
pub use artifact::session_to_artifact;
pub use error::RuchyBridgeError;
pub use metrics::SessionMetrics;
pub use session::{CodeCell, EntrenarSession, MetricsExportSummary, SessionExport};
pub use types::{RuchyCell, RuchySession, TrainingRun};
