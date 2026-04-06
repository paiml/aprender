//! Performance tracing for ptop `ComputeBlocks`
//!
//! This module provides lightweight performance tracing compatible with
//! renacer's `BrickTracer` format. It can be used standalone or integrated
//! with renacer for deep syscall-level analysis.
//!
//! **Specification**: SPEC-024 Section 23.5 (Presentar Headless Tracing)
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                   PerfTrace Architecture                         │
//! │                                                                  │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
//! │  │ PerfTracer  │ →  │ TraceEvent  │ →  │ renacer BrickTracer │  │
//! │  │ (in-process)│    │ (metrics)   │    │ (optional deep)     │  │
//! │  └─────────────┘    └─────────────┘    └─────────────────────┘  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use presentar_terminal::perf_trace::PerfTracer;
//!
//! let mut tracer = PerfTracer::new();
//!
//! // Trace a block of code
//! let result = tracer.trace("collect_metrics", || {
//!     app.collect_metrics();
//! });
//!
//! // Get performance summary
//! println!("{}", tracer.summary());
//! ```

mod analysis;
mod core;
mod data_structures;
mod helpers_batch;
mod helpers_infra;
mod trackers;

pub use self::core::*;
pub use analysis::*;
pub use data_structures::*;
pub use helpers_batch::*;
pub use helpers_infra::*;
pub use trackers::*;
