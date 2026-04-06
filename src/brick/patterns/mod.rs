//! Async and Buffer Pattern Catalog
//!
//! Utility patterns from Phase 12 (E.10) for async operations,
//! buffer management, and flow control.
//!
//! # Patterns Included
//!
//! - **LCP-12**: AsyncResult - async compute with sync fallback
//! - **AWP-11**: BoundedQueue - bounded message queue with back-pressure
//! - **AWP-13**: ReserveStrategy - buffer reservation strategies
//! - **LCP-08**: GraphReuseCounter - graph reuse tracking
//! - **AWP-03**: DualWakerState - dual-waker backpressure
//! - **AWP-04**: StreamCapacity - HTTP/2 flow control
//! - **AWP-09**: WakeSkipState - smart wake skip optimization

#[cfg(test)]
mod tests;

mod async_result;
mod bounded_queue;
mod dual_waker_state;
mod graph_reuse_counter;
mod reserve_strategy;
mod stream_capacity;
mod wake_skip_state;

pub use async_result::AsyncResult;
pub use bounded_queue::BoundedQueue;
pub use dual_waker_state::{DualWakerState, WakeDecision};
pub use graph_reuse_counter::GraphReuseCounter;
pub use reserve_strategy::{reserve_capacity, ReserveStrategy, StrategicBuffer};
pub use stream_capacity::{FlowControlError, StreamCapacity};
pub use wake_skip_state::WakeSkipState;
