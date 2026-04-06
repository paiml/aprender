//! Alerting engine for Renacer observability platform
//!
//! Sprint 57: Alerting + Visualization
//!
//! Provides threshold-based and anomaly-based alerting with:
//! - Alert rule DSL in renacer.toml
//! - State machine: Pending → Firing → Resolved
//! - Notification routing
//!
//! # Peer-Reviewed Foundations
//! - Borgmon (Google 2003): Alert rule evaluation
//! - Prometheus Alertmanager: Alert state machine
//!
//! # Example
//! ```ignore
//! use renacer::alerting::{AlertEngine, AlertRule};
//!
//! let engine = AlertEngine::new(registry);
//! engine.add_rule(AlertRule::threshold(
//!     "high_latency",
//!     "syscall_duration_seconds_p99 > 0.1",
//!     Duration::from_secs(60),
//!     Severity::Warning,
//! ));
//! engine.run();
//! ```

mod engine;
mod rule;
mod state;

pub use engine::AlertEngine;
pub use rule::{AlertExpr, AlertRule, Severity};
pub use state::{ActiveAlert, AlertState};

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(AlertRule: Send, Sync);
static_assertions::assert_impl_all!(AlertExpr: Send, Sync);
static_assertions::assert_impl_all!(Severity: Send, Sync);
static_assertions::assert_impl_all!(ActiveAlert: Send, Sync);
static_assertions::assert_impl_all!(AlertState: Send, Sync);
