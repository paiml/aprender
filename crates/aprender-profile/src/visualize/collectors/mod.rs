//! Data collectors for visualization (trueno-viz pattern)
//!
//! Collectors gather metrics from various sources and normalize them
//! for display in the TUI panels. Each collector implements a common
//! trait for consistent lifecycle management.
//!
//! # Architecture
//!
//! ```text
//! ptrace → SyscallCollector ─┐
//!                            ├→ VisualizeApp (ring buffers)
//! stats → AnomalyCollector ──┤
//!                            │
//! otlp  → SpanReceiver ──────┘
//! ```

pub mod anomaly;
pub mod span;
pub mod syscall;

pub use anomaly::AnomalyCollector;
pub use span::SpanReceiver;
pub use syscall::SyscallCollector;

use anyhow::Result;
use std::collections::HashMap;

/// Metric value types for collectors
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Instantaneous gauge value
    Gauge(f64),
    /// Monotonically increasing counter
    Counter(u64),
    /// Rate of change (per second)
    Rate(f64),
    /// Histogram percentile values
    Histogram { p50: f64, p95: f64, p99: f64, max: f64 },
}

/// Collection of metrics from a single collection cycle
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// Named metric values
    pub values: HashMap<String, MetricValue>,
    /// Timestamp of collection (monotonic nanoseconds)
    pub timestamp_ns: u64,
}

impl Metrics {
    /// Create new metrics collection with current timestamp
    pub fn new(values: HashMap<String, MetricValue>) -> Self {
        Self { values, timestamp_ns: std::time::Instant::now().elapsed().as_nanos() as u64 }
    }

    /// Create empty metrics
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Collector trait for gathering visualization metrics
///
/// Based on trueno-viz collector pattern for consistent
/// lifecycle management and testability.
pub trait Collector: Send {
    /// Collect current metrics
    ///
    /// Called at the visualization tick rate (default: 50ms).
    /// Should be fast (<10ms) to maintain frame rate.
    fn collect(&mut self) -> Result<Metrics>;

    /// Check if collector is available and functional
    ///
    /// Returns false if the data source is unavailable
    /// (e.g., process terminated, permission denied).
    fn is_available(&self) -> bool;

    /// Collector name for logging and debugging
    fn name(&self) -> &'static str;

    /// Reset collector state
    ///
    /// Called when visualization is restarted or data source changes.
    fn reset(&mut self) {}
}

/// Mock collector for testing
#[derive(Debug, Default)]
pub struct MockCollector {
    /// Canned metrics to return
    pub metrics: Metrics,
    /// Whether collector is available
    pub available: bool,
    /// Collection count for verification
    pub collect_count: usize,
}

impl MockCollector {
    /// Create new mock collector
    pub fn new() -> Self {
        Self { metrics: Metrics::empty(), available: true, collect_count: 0 }
    }

    /// Set canned metrics to return
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Set availability
    pub fn with_available(mut self, available: bool) -> Self {
        self.available = available;
        self
    }
}

impl Collector for MockCollector {
    fn collect(&mut self) -> Result<Metrics> {
        self.collect_count += 1;
        Ok(self.metrics.clone())
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn reset(&mut self) {
        self.collect_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_value_debug() {
        let gauge = MetricValue::Gauge(42.0);
        assert!(format!("{:?}", gauge).contains("42"));

        let counter = MetricValue::Counter(100);
        assert!(format!("{:?}", counter).contains("100"));

        let rate = MetricValue::Rate(1.5);
        assert!(format!("{:?}", rate).contains("1.5"));

        let hist = MetricValue::Histogram { p50: 10.0, p95: 50.0, p99: 90.0, max: 100.0 };
        assert!(format!("{:?}", hist).contains("p50"));
    }

    #[test]
    fn test_metrics_new() {
        let mut values = HashMap::new();
        values.insert("test".to_string(), MetricValue::Gauge(42.0));
        let metrics = Metrics::new(values);

        assert_eq!(metrics.values.len(), 1);
        assert!(matches!(
            metrics.values.get("test"),
            Some(MetricValue::Gauge(v)) if (*v - 42.0).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_mock_collector() {
        let mut collector = MockCollector::new();
        assert!(collector.is_available());
        assert_eq!(collector.name(), "mock");
        assert_eq!(collector.collect_count, 0);

        let _ = collector.collect();
        assert_eq!(collector.collect_count, 1);

        collector.reset();
        assert_eq!(collector.collect_count, 0);
    }

    #[test]
    fn test_mock_collector_unavailable() {
        let collector = MockCollector::new().with_available(false);
        assert!(!collector.is_available());
    }

    #[test]
    fn test_mock_collector_with_metrics() {
        let mut values = HashMap::new();
        values.insert("cpu".to_string(), MetricValue::Gauge(75.0));
        let metrics = Metrics::new(values);

        let mut collector = MockCollector::new().with_metrics(metrics);
        let result = collector.collect().unwrap();

        assert!(matches!(
            result.values.get("cpu"),
            Some(MetricValue::Gauge(v)) if (*v - 75.0).abs() < f64::EPSILON
        ));
    }
}
