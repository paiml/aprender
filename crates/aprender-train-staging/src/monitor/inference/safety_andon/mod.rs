//! Safety Andon for Inference (ENT-110)
//!
//! Toyota Way 自働化 (Jidoka): Automation with human touch.
//! Inference-specific Andon rules: low confidence, high latency, drift.

mod andon;
mod emergency;
mod sil;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use andon::SafetyAndon;
pub use emergency::EmergencyCondition;
pub use sil::SafetyIntegrityLevel;
