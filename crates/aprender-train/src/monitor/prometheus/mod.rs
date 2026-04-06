//! Prometheus Metrics Export Module (MLOPS-006)
//!
//! Integration with standard observability stacks.
//!
//! # Toyota Way: (Andon)
//!
//! Visual alerting through Prometheus/Grafana dashboards.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::monitor::prometheus::PrometheusExporter;
//!
//! let exporter = PrometheusExporter::new("my-experiment", "run-1");
//! exporter.record_epoch(1, 0.5, 0.001);
//! let metrics = exporter.export();
//! println!("{}", metrics);
//! ```

mod exporter;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use exporter::PrometheusExporter;
pub use types::{LabelSet, MetricDef, MetricType};
