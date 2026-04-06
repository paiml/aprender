//! Tracing Escalation Framework (PMAT-021)
//!
//! Implements automatic escalation to renacer tracing per section 35.2 when
//! cbtop detects anomalies (CV > 15% or efficiency < 25%).
//!
//! # Escalation Triggers
//!
//! | Metric | Threshold | Action |
//! |--------|-----------|--------|
//! | CV | > 15% | Escalate to syscall tracing |
//! | Efficiency | < 25% | Escalate to function profiling |
//! | Memory cliff | Sudden drop | Escalate with memory focus |
//! | GPU transfer | > 50% | Escalate with PCIe focus |
//!
//! # Citations
//!
//! - [Sigelman et al. 2010] "Dapper: Distributed Systems Tracing" Google Tech Report
//! - [Mace et al. 2015] "Pivot Tracing: Dynamic Causal Monitoring" ACM SOSP

mod manager;
mod types;

pub use manager::{OtlpSpanAttributes, TracingEscalation};
pub use types::{EscalationReason, EscalationThresholds, SyscallBreakdown, TraceResult};

#[cfg(test)]
mod tests;
