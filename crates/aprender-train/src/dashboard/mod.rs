//! Dashboard Module (Phase 2: ENT-003, ENT-004)
//!
//! Provides the `DashboardSource` trait for real-time training monitoring
//! and WASM bindings for browser-based dashboards.
//!
//! # Features
//!
//! - Real-time metric streaming via `subscribe()`
//! - Resource usage monitoring (GPU, CPU, memory)
//! - Trend analysis for metrics
//! - WASM support for browser dashboards (feature: "wasm")
//!
//! # Example
//!
//! ```
//! use std::sync::{Arc, Mutex};
//! use entrenar::storage::{InMemoryStorage, ExperimentStorage};
//! use entrenar::run::{Run, TracingConfig};
//! use entrenar::dashboard::{DashboardSource, Trend};
//!
//! let mut storage = InMemoryStorage::new();
//! let exp_id = storage.create_experiment("my-exp", None).expect("create experiment");
//! let storage = Arc::new(Mutex::new(storage));
//!
//! let mut run = Run::new(&exp_id, storage.clone(), TracingConfig::disabled()).expect("create run");
//! run.log_metric("loss", 0.5).expect("log metric");
//! run.log_metric("loss", 0.4).expect("log metric");
//! run.log_metric("loss", 0.3).expect("log metric");
//!
//! // Get dashboard data
//! let status = run.status();
//! let metrics = run.recent_metrics(10);
//! let resources = run.resource_usage();
//! ```

mod snapshot;
mod source;
mod trend;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(test)]
mod tests;

// Re-exports
pub use snapshot::{MetricSnapshot, ResourceSnapshot};
pub use source::{DashboardSource, SubscriptionCallback};
pub use trend::Trend;
