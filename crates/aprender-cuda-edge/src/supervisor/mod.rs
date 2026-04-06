//! GPU worker supervision with restart strategies.
//!
//! This module implements Erlang/OTP-style supervision trees adapted for
//! GPU worker processes. Workers run in isolated child processes to
//! prevent one GPU crash from corrupting other tests.
//!
//! # Restart Strategies
//!
//! - **One-for-One**: Only restart the crashed worker
//! - **One-for-All**: Restart all workers when any crashes
//! - **Rest-for-One**: Restart the crashed worker and all workers started after it
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::supervisor::{SupervisionTree, SupervisionStrategy, SupervisorAction};
//!
//! let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);
//! let action = tree.handle_crash(2, 0);
//! assert_eq!(action, SupervisorAction::Restart(vec![2]));
//! ```

pub mod heartbeat;
pub mod strategy;
pub mod tree;
pub mod worker;

pub use heartbeat::{GpuHealthMonitor, GpuHealthMonitorBuilder};
pub use strategy::{HealthAction, SupervisionStrategy, SupervisorAction};
pub use tree::{RestartRecord, SupervisionTree};
pub use worker::{HeartbeatStatus, WorkerConfig, WorkerTaskId};

#[cfg(feature = "cuda")]
pub use worker::spawn_worker;
