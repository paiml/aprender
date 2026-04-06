//! renacer-core: Zero-dependency tracing primitives
//!
//! Extracted from renacer to break the renacer→aprender→realizr→renacer
//! circular dependency (PMAT-284). This crate provides:
//!
//! - [`SpanRecord`] — Parquet-compatible span schema (W3C Trace Context)
//! - [`LazySpan`] — Deferred span construction (zero overhead when unused)
//! - [`SpanPool`] — Memory pool for span allocations
//! - [`TraceContext`] / [`LamportClock`] — W3C trace context + causal ordering
//!
//! # Design
//!
//! This crate has NO dependencies on aprender, realizr, or trueno.
//! It can be used by any crate in the stack for uniform instrumentation.
//!
//! # Usage in realizr
//!
//! ```ignore
//! use renacer_core::LazySpan;
//!
//! let span = LazySpan::new()
//!     .with_name_static("decode_step")
//!     .with_attribute_static("m", state.m.to_string())
//!     .timed();  // starts timing
//! // ... GPU work ...
//! span.finish(); // records duration
//! ```

pub mod lazy_span;
pub mod phase_timer;
pub mod span_pool;
pub mod span_record;
pub mod trace_context;

pub use lazy_span::LazySpan;
pub use phase_timer::PhaseTimer;
pub use span_pool::SpanPool;
pub use span_record::{SpanKind, SpanRecord, StatusCode};
pub use trace_context::{LamportClock, TraceContext};
