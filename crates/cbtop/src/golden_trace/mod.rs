//! Golden Trace Comparison (PMAT-029)
//!
//! Capture and compare performance traces against golden baselines for regression detection.
//!
//! # Features
//!
//! - Capture golden performance traces
//! - Compare current metrics against baseline
//! - Detect regressions (>10% deviation)
//! - Export traces for review
//!
//! # Falsification Criteria (F1211-F1220)
//!
//! See `tests/golden_trace_f1211.rs` for falsification tests.

mod manager;
mod trace;
mod types;

pub use manager::GoldenTraceManager;
pub use trace::{GoldenComparator, GoldenTrace, TraceComparison};
pub use types::{
    GoldenTraceError, GoldenTraceResult, SyscallBreakdown, SyscallBreakdownDelta, TraceMetrics,
};

#[cfg(test)]
mod tests;
