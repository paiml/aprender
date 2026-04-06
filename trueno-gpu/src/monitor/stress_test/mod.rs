//! Stress Test Mode (TRUENO-SPEC-025)
//!
//! Comprehensive stress testing for CPU, GPU, memory, and PCIe to validate
//! system behavior under load.
//!
//! # CLI Interface
//!
//! ```bash
//! trueno-monitor --stress-test
//! trueno-monitor --stress-test --target cpu
//! trueno-monitor --stress-test --target gpu:0
//! trueno-monitor --stress-test --duration 60s --intensity 0.8
//! trueno-monitor --stress-test --chaos gentle
//! ```
//!
//! # References
//!
//! - [Volkov2008] GPU stress via GEMM workloads
//! - renacer chaos engineering integration

// Re-export Duration so tests (which use `super::*`) can access it,
// matching the original single-file module's imports.
#[cfg(test)]
use std::time::Duration;

mod config;
mod results;

pub use config::{ChaosPreset, StressTarget, StressTestConfig};
pub use results::{StressMetrics, StressTestReport, StressTestState, StressTestVerdict};

#[cfg(test)]
mod tests;
