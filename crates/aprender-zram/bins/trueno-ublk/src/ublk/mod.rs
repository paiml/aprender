//! Zero-dependency ublk implementation
//!
//! Direct kernel interface using only:
//! - nix::libc for ioctls
//! - io-uring for async I/O
//!
//! No libublk, no smol, no futures.

#![allow(dead_code, unused_imports)]

pub mod ctrl;
pub mod daemon;
pub mod io;
pub mod multi_queue;
pub mod shim;
pub mod sys;

#[cfg(test)]
pub use ctrl::MockUblkCtrl as UblkCtrl;
#[cfg(not(test))]
pub use ctrl::UblkCtrl;
pub use daemon::DaemonError;
#[cfg(test)]
pub use daemon::MockUblkDaemon as UblkDaemon;
#[cfg(not(test))]
pub use daemon::{run_daemon, run_daemon_batched, BatchedDaemonConfig, UblkDaemon};
