//! Core types for the monitoring system.
//!
//! This module defines the fundamental types used throughout the monitor:
//!
//! - [`MetricValue`]: Enum representing different metric types (gauge, counter, histogram)
//! - [`Metrics`]: A timestamped collection of metric values
//! - [`Collector`]: Trait for metric collection implementations
//!
//! # Design Principles
//!
//! - **Type Safety**: Strongly typed metric values prevent mixing incompatible types
//! - **Extensibility**: Collector trait allows custom metric sources
//! - **Performance**: Designed for <5ms collection overhead

use super::error::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single metric value.
///
/// Metrics come in different types depending on their semantics:
///
/// - **Gauge**: A point-in-time value (CPU %, temperature)
/// - **Counter**: A monotonically increasing value (bytes transferred)
/// - **Histogram**: A distribution of values (latency percentiles)
/// - **Text**: A descriptive string value (model name, algorithm)
#[derive(Debug, Clone, PartialEq)]
pub enum MetricValue {
    /// A point-in-time measurement (CPU usage, temperature).
    Gauge(f64),

    /// A monotonically increasing counter (bytes, packets).
    Counter(u64),

    /// A distribution of values (latencies).
    Histogram(Vec<f64>),

    /// A text value (model name, status).
    Text(String),
}

impl MetricValue {
    /// Returns the value as a gauge, if it is one.
    #[must_use]
    pub fn as_gauge(&self) -> Option<f64> {
        match self {
            Self::Gauge(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as a counter, if it is one.
    #[must_use]
    pub fn as_counter(&self) -> Option<u64> {
        match self {
            Self::Counter(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as a histogram, if it is one.
    #[must_use]
    pub fn as_histogram(&self) -> Option<&[f64]> {
        match self {
            Self::Histogram(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the value as text, if it is one.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    /// Returns true if this is a gauge value.
    #[must_use]
    pub fn is_gauge(&self) -> bool {
        matches!(self, Self::Gauge(_))
    }

    /// Returns true if this is a counter value.
    #[must_use]
    pub fn is_counter(&self) -> bool {
        matches!(self, Self::Counter(_))
    }

    /// Returns true if this is a histogram value.
    #[must_use]
    pub fn is_histogram(&self) -> bool {
        matches!(self, Self::Histogram(_))
    }

    /// Returns true if this is a text value.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }
}

impl From<f64> for MetricValue {
    fn from(value: f64) -> Self {
        Self::Gauge(value)
    }
}

impl From<u64> for MetricValue {
    fn from(value: u64) -> Self {
        Self::Counter(value)
    }
}

impl From<String> for MetricValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for MetricValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<f64>> for MetricValue {
    fn from(value: Vec<f64>) -> Self {
        Self::Histogram(value)
    }
}

/// A collection of metrics with a timestamp.
///
/// Metrics are collected as a batch with a single timestamp to ensure
/// consistency within a collection cycle.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// When these metrics were collected.
    pub timestamp: Instant,

    /// The metric values, keyed by metric name.
    pub values: HashMap<String, MetricValue>,
}

impl Metrics {
    /// Creates a new empty metrics collection with the current timestamp.
    #[must_use]
    pub fn new() -> Self {
        Self { timestamp: Instant::now(), values: HashMap::new() }
    }

    /// Creates a metrics collection with a specific timestamp.
    #[must_use]
    pub fn with_timestamp(timestamp: Instant) -> Self {
        Self { timestamp, values: HashMap::new() }
    }

    /// Adds a metric value.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<MetricValue>) {
        self.values.insert(key.into(), value.into());
    }

    /// Gets a metric value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetricValue> {
        self.values.get(key)
    }

    /// Gets a gauge value by key.
    #[must_use]
    pub fn get_gauge(&self, key: &str) -> Option<f64> {
        self.values.get(key).and_then(MetricValue::as_gauge)
    }

    /// Gets a counter value by key.
    #[must_use]
    pub fn get_counter(&self, key: &str) -> Option<u64> {
        self.values.get(key).and_then(MetricValue::as_counter)
    }

    /// Returns the number of metrics in this collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if there are no metrics.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns an iterator over the metric keys and values.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetricValue)> {
        self.values.iter()
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for metric collectors.
///
/// Collectors are responsible for gathering metrics from a specific source
/// (CPU, memory, GPU, etc.). They must be `Send + Sync` to allow concurrent
/// collection in background threads.
///
/// # Example
///
/// ```rust,ignore
/// use trueno_viz::monitor::{Collector, Metrics, Result};
///
/// struct MyCollector;
///
/// impl Collector for MyCollector {
///     fn id(&self) -> &'static str {
///         "my_collector"
///     }
///
///     fn collect(&mut self) -> Result<Metrics> {
///         let mut metrics = Metrics::new();
///         metrics.insert("my_value", 42.0);
///         Ok(metrics)
///     }
///
///     fn is_available(&self) -> bool {
///         true
///     }
/// }
/// ```
pub trait Collector: Send + Sync {
    /// Returns the unique identifier for this collector.
    ///
    /// This is used for error messages and configuration keys.
    fn id(&self) -> &'static str;

    /// Collects metrics from this source.
    ///
    /// This method should complete within the collector's `interval_hint()`
    /// to avoid blocking the main collection loop.
    ///
    /// # Errors
    ///
    /// Returns an error if metric collection fails (e.g., /proc not readable).
    fn collect(&mut self) -> Result<Metrics>;

    /// Returns true if this collector is available on the current system.
    ///
    /// For example, the NVIDIA collector returns false on systems without
    /// NVIDIA GPUs.
    fn is_available(&self) -> bool;

    /// Suggests an appropriate collection interval for this collector.
    ///
    /// Collectors with expensive operations (process scanning) may suggest
    /// longer intervals than lightweight ones (CPU stats).
    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    /// Returns a human-readable name for this collector.
    fn display_name(&self) -> &'static str {
        self.id()
    }
}

/// A boxed collector for dynamic dispatch.
pub type BoxedCollector = Box<dyn Collector>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MetricValue tests
    // ========================================================================

    #[test]
    fn test_metric_value_gauge() {
        let v = MetricValue::Gauge(42.5);

        assert!(v.is_gauge());
        assert!(!v.is_counter());
        assert_eq!(v.as_gauge(), Some(42.5));
        assert_eq!(v.as_counter(), None);
    }

    #[test]
    fn test_metric_value_counter() {
        let v = MetricValue::Counter(1000);

        assert!(v.is_counter());
        assert!(!v.is_gauge());
        assert_eq!(v.as_counter(), Some(1000));
        assert_eq!(v.as_gauge(), None);
    }

    #[test]
    fn test_metric_value_histogram() {
        let v = MetricValue::Histogram(vec![1.0, 2.0, 3.0]);

        assert!(v.is_histogram());
        assert_eq!(v.as_histogram(), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn test_metric_value_text() {
        let v = MetricValue::Text("hello".to_string());

        assert!(v.is_text());
        assert_eq!(v.as_text(), Some("hello"));
    }

    #[test]
    fn test_metric_value_from_f64() {
        let v: MetricValue = 42.0.into();
        assert_eq!(v, MetricValue::Gauge(42.0));
    }

    #[test]
    fn test_metric_value_from_u64() {
        let v: MetricValue = 100u64.into();
        assert_eq!(v, MetricValue::Counter(100));
    }

    #[test]
    fn test_metric_value_from_string() {
        let v: MetricValue = "test".into();
        assert_eq!(v, MetricValue::Text("test".to_string()));
    }

    #[test]
    fn test_metric_value_from_vec() {
        let v: MetricValue = vec![1.0, 2.0].into();
        assert_eq!(v, MetricValue::Histogram(vec![1.0, 2.0]));
    }

    #[test]
    fn test_metric_value_clone() {
        let v = MetricValue::Gauge(42.0);
        let cloned = v.clone();
        assert_eq!(v, cloned);
    }

    // ========================================================================
    // Metrics tests
    // ========================================================================

    #[test]
    fn test_metrics_new() {
        let m = Metrics::new();

        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_metrics_insert_and_get() {
        let mut m = Metrics::new();

        m.insert("cpu", 75.5);
        m.insert("memory", 1024u64);

        assert_eq!(m.len(), 2);
        assert_eq!(m.get_gauge("cpu"), Some(75.5));
        assert_eq!(m.get_counter("memory"), Some(1024));
        assert_eq!(m.get("nonexistent"), None);
    }

    #[test]
    fn test_metrics_iter() {
        let mut m = Metrics::new();
        m.insert("a", 1.0);
        m.insert("b", 2.0);

        let keys: Vec<_> = m.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
    }

    #[test]
    fn test_metrics_default() {
        let m = Metrics::default();
        assert!(m.is_empty());
    }

    // ========================================================================
    // Collector trait tests
    // ========================================================================

    struct TestCollector {
        available: bool,
        value: f64,
    }

    impl Collector for TestCollector {
        fn id(&self) -> &'static str {
            "test"
        }

        fn collect(&mut self) -> Result<Metrics> {
            let mut m = Metrics::new();
            m.insert("value", self.value);
            Ok(m)
        }

        fn is_available(&self) -> bool {
            self.available
        }

        fn interval_hint(&self) -> Duration {
            Duration::from_millis(500)
        }

        fn display_name(&self) -> &'static str {
            "Test Collector"
        }
    }

    #[test]
    fn test_collector_trait() {
        let mut collector = TestCollector { available: true, value: 42.0 };

        assert_eq!(collector.id(), "test");
        assert!(collector.is_available());
        assert_eq!(collector.interval_hint(), Duration::from_millis(500));
        assert_eq!(collector.display_name(), "Test Collector");

        let metrics = collector.collect().expect("collect should succeed");
        assert_eq!(metrics.get_gauge("value"), Some(42.0));
    }

    #[test]
    fn test_collector_unavailable() {
        let collector = TestCollector { available: false, value: 0.0 };

        assert!(!collector.is_available());
    }

    #[test]
    fn test_boxed_collector() {
        let collector: BoxedCollector = Box::new(TestCollector { available: true, value: 100.0 });

        assert_eq!(collector.id(), "test");
        assert!(collector.is_available());
    }

    #[test]
    fn test_collector_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TestCollector>();
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[test]
    fn test_metric_value_partial_eq() {
        let v1 = MetricValue::Gauge(1.0);
        let v2 = MetricValue::Gauge(1.0);
        let v3 = MetricValue::Gauge(2.0);

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_metrics_with_timestamp() {
        let ts = Instant::now();
        let m = Metrics::with_timestamp(ts);

        assert_eq!(m.timestamp, ts);
    }

    #[test]
    fn test_metric_value_histogram_empty() {
        let v = MetricValue::Histogram(vec![]);
        assert!(v.is_histogram());
        assert_eq!(v.as_histogram(), Some(&[][..]));
    }
}
