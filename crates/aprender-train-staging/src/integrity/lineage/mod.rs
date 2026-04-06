//! Lamport Timestamps and Causal Lineage (ENT-014)
//!
//! Provides logical clocks for tracking causal ordering of events
//! across distributed training runs without requiring synchronized wall clocks.
//!
//! # Example
//!
//! ```
//! use entrenar::integrity::{LamportTimestamp, LineageEventType, CausalLineage};
//!
//! let mut ts1 = LamportTimestamp::new("node-1");
//! let mut ts2 = LamportTimestamp::new("node-2");
//!
//! // Local events increment counter
//! ts1.increment();
//! ts1.increment();
//!
//! // Receiving a message merges timestamps
//! ts2.merge(&ts1);
//!
//! // Check causal ordering
//! assert!(ts1.happens_before(&ts2));
//! ```

mod causal;
mod event;
mod timestamp;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use causal::CausalLineage;
pub use event::{LineageEvent, LineageEventType};
pub use timestamp::LamportTimestamp;
