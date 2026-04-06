//! Metrics registry - thread-safe collection of metrics
//!
//! The registry manages all metrics and provides:
//! - Thread-safe registration and lookup
//! - Duplicate detection
//! - Label validation
//! - Export interface for OTLP
//!
//! # Pattern
//! Prometheus client registry with dashmap for lock-free reads

use dashmap::DashMap;
use std::sync::Arc;
use thiserror::Error;

use super::counter::Counter;
use super::gauge::Gauge;
use super::histogram::{Histogram, DEFAULT_BUCKETS};
use super::labels::{LabelValidator, Labels};
use super::MetricType;

/// Registry errors
#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("metric already registered: {name}")]
    AlreadyRegistered { name: String },

    #[error("metric type mismatch: {name} is {existing}, not {requested}")]
    TypeMismatch { name: String, existing: MetricType, requested: MetricType },

    #[error("label validation failed: {0}")]
    LabelError(#[from] super::labels::LabelError),

    #[error("invalid metric name: {0} (must match [a-zA-Z_:][a-zA-Z0-9_:]*)")]
    InvalidName(String),

    #[error("cardinality limit exceeded: {count} series (max {max})")]
    CardinalityLimit { count: usize, max: usize },
}

/// Registered metric wrapper
#[derive(Clone, Debug)]
pub enum RegisteredMetric {
    Counter(Arc<Counter>),
    Gauge(Arc<Gauge>),
    Histogram(Arc<Histogram>),
}

impl RegisteredMetric {
    pub fn metric_type(&self) -> MetricType {
        match self {
            RegisteredMetric::Counter(_) => MetricType::Counter,
            RegisteredMetric::Gauge(_) => MetricType::Gauge,
            RegisteredMetric::Histogram(_) => MetricType::Histogram,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            RegisteredMetric::Counter(c) => c.name(),
            RegisteredMetric::Gauge(g) => g.name(),
            RegisteredMetric::Histogram(h) => h.name(),
        }
    }
}

/// Metrics registry - central collection of all metrics
///
/// Thread-safe via dashmap (lock-free concurrent hashmap)
#[derive(Debug)]
pub struct Registry {
    /// All registered metrics by unique key
    metrics: DashMap<String, RegisteredMetric>,
    /// Label validator
    validator: LabelValidator,
    /// Maximum number of unique time series
    max_series: usize,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// Create a new registry with default settings
    pub fn new() -> Self {
        Self { metrics: DashMap::new(), validator: LabelValidator::new(), max_series: 100_000 }
    }

    /// Create registry with custom label validator
    pub fn with_validator(validator: LabelValidator) -> Self {
        Self { metrics: DashMap::new(), validator, max_series: 100_000 }
    }

    /// Set maximum series limit
    pub fn max_series(mut self, max: usize) -> Self {
        self.max_series = max;
        self
    }

    /// Register or get existing counter
    pub fn counter(
        &self,
        name: &str,
        labels: &[(&str, &str)],
    ) -> Result<Arc<Counter>, RegistryError> {
        self.validate_name(name)?;
        let labels_map = self.labels_from_pairs(labels)?;
        let key = metric_key(name, &labels_map);

        // Check if already exists
        if let Some(entry) = self.metrics.get(&key) {
            match entry.value() {
                RegisteredMetric::Counter(c) => return Ok(c.clone()),
                other => {
                    return Err(RegistryError::TypeMismatch {
                        name: name.to_string(),
                        existing: other.metric_type(),
                        requested: MetricType::Counter,
                    })
                }
            }
        }

        // Check cardinality limit
        self.check_cardinality()?;

        // Create new counter
        let counter = Counter::new_arc(name, labels_map);
        self.metrics.insert(key, RegisteredMetric::Counter(counter.clone()));

        Ok(counter)
    }

    /// Register or get existing gauge
    pub fn gauge(&self, name: &str, labels: &[(&str, &str)]) -> Result<Arc<Gauge>, RegistryError> {
        self.validate_name(name)?;
        let labels_map = self.labels_from_pairs(labels)?;
        let key = metric_key(name, &labels_map);

        // Check if already exists
        if let Some(entry) = self.metrics.get(&key) {
            match entry.value() {
                RegisteredMetric::Gauge(g) => return Ok(g.clone()),
                other => {
                    return Err(RegistryError::TypeMismatch {
                        name: name.to_string(),
                        existing: other.metric_type(),
                        requested: MetricType::Gauge,
                    })
                }
            }
        }

        // Check cardinality limit
        self.check_cardinality()?;

        // Create new gauge
        let gauge = Gauge::new_arc(name, labels_map);
        self.metrics.insert(key, RegisteredMetric::Gauge(gauge.clone()));

        Ok(gauge)
    }

    /// Register or get existing histogram with default buckets
    pub fn histogram(
        &self,
        name: &str,
        labels: &[(&str, &str)],
    ) -> Result<Arc<Histogram>, RegistryError> {
        self.histogram_with_buckets(name, labels, DEFAULT_BUCKETS)
    }

    /// Register or get existing histogram with custom buckets
    pub fn histogram_with_buckets(
        &self,
        name: &str,
        labels: &[(&str, &str)],
        buckets: &[f64],
    ) -> Result<Arc<Histogram>, RegistryError> {
        self.validate_name(name)?;
        let labels_map = self.labels_from_pairs(labels)?;
        let key = metric_key(name, &labels_map);

        // Check if already exists
        if let Some(entry) = self.metrics.get(&key) {
            match entry.value() {
                RegisteredMetric::Histogram(h) => return Ok(h.clone()),
                other => {
                    return Err(RegistryError::TypeMismatch {
                        name: name.to_string(),
                        existing: other.metric_type(),
                        requested: MetricType::Histogram,
                    })
                }
            }
        }

        // Check cardinality limit
        self.check_cardinality()?;

        // Create new histogram
        let histogram = Histogram::new_arc(name, labels_map, buckets);
        self.metrics.insert(key, RegisteredMetric::Histogram(histogram.clone()));

        Ok(histogram)
    }

    /// Get number of registered metrics
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// Iterate over all registered metrics
    pub fn iter(&self) -> impl Iterator<Item = RegisteredMetric> + '_ {
        self.metrics.iter().map(|entry| entry.value().clone())
    }

    /// Get all counters
    pub fn counters(&self) -> Vec<Arc<Counter>> {
        self.metrics
            .iter()
            .filter_map(|entry| match entry.value() {
                RegisteredMetric::Counter(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    /// Get all gauges
    pub fn gauges(&self) -> Vec<Arc<Gauge>> {
        self.metrics
            .iter()
            .filter_map(|entry| match entry.value() {
                RegisteredMetric::Gauge(g) => Some(g.clone()),
                _ => None,
            })
            .collect()
    }

    /// Get all histograms
    pub fn histograms(&self) -> Vec<Arc<Histogram>> {
        self.metrics
            .iter()
            .filter_map(|entry| match entry.value() {
                RegisteredMetric::Histogram(h) => Some(h.clone()),
                _ => None,
            })
            .collect()
    }

    /// Unregister a metric by key
    pub fn unregister(&self, name: &str, labels: &[(&str, &str)]) -> bool {
        if let Ok(labels_map) = self.labels_from_pairs(labels) {
            let key = metric_key(name, &labels_map);
            self.metrics.remove(&key).is_some()
        } else {
            false
        }
    }

    /// Clear all metrics
    pub fn clear(&self) {
        self.metrics.clear();
    }

    /// Validate metric name
    fn validate_name(&self, name: &str) -> Result<(), RegistryError> {
        if name.is_empty() {
            return Err(RegistryError::InvalidName(name.to_string()));
        }

        let mut chars = name.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' || c == ':' => {}
            _ => return Err(RegistryError::InvalidName(name.to_string())),
        }
        for c in chars {
            if !c.is_ascii_alphanumeric() && c != '_' && c != ':' {
                return Err(RegistryError::InvalidName(name.to_string()));
            }
        }

        Ok(())
    }

    /// Convert label pairs to Labels map with validation
    fn labels_from_pairs(&self, pairs: &[(&str, &str)]) -> Result<Labels, RegistryError> {
        let labels: Labels =
            pairs.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect();

        self.validator.validate(&labels)?;
        Ok(labels)
    }

    /// Check cardinality limit
    fn check_cardinality(&self) -> Result<(), RegistryError> {
        if self.metrics.len() >= self.max_series {
            return Err(RegistryError::CardinalityLimit {
                count: self.metrics.len(),
                max: self.max_series,
            });
        }
        Ok(())
    }
}

/// Generate unique key for metric (name + sorted labels)
fn metric_key(name: &str, labels: &Labels) -> String {
    let mut key = name.to_string();
    let mut sorted_labels: Vec<_> = labels.iter().collect();
    sorted_labels.sort_by_key(|(k, _)| *k);
    for (k, v) in sorted_labels {
        key.push_str(&format!("|{}={}", k, v));
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Global default registry (lazy initialized)
    fn default_registry() -> &'static Registry {
        use std::sync::OnceLock;
        static REGISTRY: OnceLock<Registry> = OnceLock::new();
        REGISTRY.get_or_init(Registry::new)
    }

    #[test]
    fn test_registry_counter() {
        let reg = Registry::new();

        let c1 = reg.counter("requests_total", &[("method", "GET")]).unwrap();
        c1.inc();

        let c2 = reg.counter("requests_total", &[("method", "GET")]).unwrap();
        assert_eq!(c2.get(), 1); // Same counter

        let c3 = reg.counter("requests_total", &[("method", "POST")]).unwrap();
        c3.inc();
        assert_eq!(c3.get(), 1); // Different counter
    }

    #[test]
    fn test_registry_gauge() {
        let reg = Registry::new();

        let g = reg.gauge("temperature", &[("location", "kitchen")]).unwrap();
        g.set(22);

        let g2 = reg.gauge("temperature", &[("location", "kitchen")]).unwrap();
        assert_eq!(g2.get(), 22);
    }

    #[test]
    fn test_registry_histogram() {
        let reg = Registry::new();

        let h = reg.histogram("latency", &[]).unwrap();
        h.observe(0.1);

        let h2 = reg.histogram("latency", &[]).unwrap();
        assert_eq!(h2.get_count(), 1);
    }

    #[test]
    fn test_registry_type_mismatch() {
        let reg = Registry::new();

        reg.counter("metric", &[]).unwrap();
        let result = reg.gauge("metric", &[]);

        assert!(matches!(result, Err(RegistryError::TypeMismatch { .. })));
    }

    #[test]
    fn test_registry_invalid_name() {
        let reg = Registry::new();

        let result = reg.counter("123invalid", &[]);
        assert!(matches!(result, Err(RegistryError::InvalidName(_))));
    }

    #[test]
    fn test_registry_cardinality_limit() {
        let reg = Registry::new().max_series(2);

        reg.counter("metric1", &[]).unwrap();
        reg.counter("metric2", &[]).unwrap();
        let result = reg.counter("metric3", &[]);

        assert!(matches!(result, Err(RegistryError::CardinalityLimit { .. })));
    }

    #[test]
    fn test_registry_unregister() {
        let reg = Registry::new();

        reg.counter("test", &[("a", "b")]).unwrap();
        assert_eq!(reg.len(), 1);

        let removed = reg.unregister("test", &[("a", "b")]);
        assert!(removed);
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_iteration() {
        let reg = Registry::new();

        reg.counter("counter1", &[]).unwrap();
        reg.gauge("gauge1", &[]).unwrap();
        reg.histogram("histogram1", &[]).unwrap();

        assert_eq!(reg.len(), 3);
        assert_eq!(reg.counters().len(), 1);
        assert_eq!(reg.gauges().len(), 1);
        assert_eq!(reg.histograms().len(), 1);
    }

    #[test]
    fn test_default_registry() {
        let reg = default_registry();
        // Should not panic
        let _ = reg.len();
    }

    #[test]
    fn test_metric_key_generation() {
        let labels: Labels = [("b", "2"), ("a", "1")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let key = metric_key("test", &labels);
        // Labels should be sorted
        assert_eq!(key, "test|a=1|b=2");
    }

    #[test]
    fn test_registry_default() {
        let reg = Registry::default();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_is_empty() {
        let reg = Registry::new();
        assert!(reg.is_empty());

        reg.counter("test", &[]).unwrap();
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_clear() {
        let reg = Registry::new();

        reg.counter("counter1", &[]).unwrap();
        reg.gauge("gauge1", &[]).unwrap();
        reg.histogram("histogram1", &[]).unwrap();

        assert_eq!(reg.len(), 3);

        reg.clear();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_iter() {
        let reg = Registry::new();

        reg.counter("c", &[]).unwrap();
        reg.gauge("g", &[]).unwrap();
        reg.histogram("h", &[]).unwrap();

        let metrics: Vec<_> = reg.iter().collect();
        assert_eq!(metrics.len(), 3);
    }

    #[test]
    fn test_registered_metric_type() {
        let counter = Counter::new_arc("test", Labels::new());
        let gauge = Gauge::new_arc("test", Labels::new());
        let histogram = Histogram::new_arc("test", Labels::new(), DEFAULT_BUCKETS);

        let rm_counter = RegisteredMetric::Counter(counter);
        let rm_gauge = RegisteredMetric::Gauge(gauge);
        let rm_histogram = RegisteredMetric::Histogram(histogram);

        assert_eq!(rm_counter.metric_type(), MetricType::Counter);
        assert_eq!(rm_gauge.metric_type(), MetricType::Gauge);
        assert_eq!(rm_histogram.metric_type(), MetricType::Histogram);
    }

    #[test]
    fn test_registered_metric_name() {
        let counter = Counter::new_arc("my_counter", Labels::new());
        let gauge = Gauge::new_arc("my_gauge", Labels::new());
        let histogram = Histogram::new_arc("my_histogram", Labels::new(), DEFAULT_BUCKETS);

        let rm_counter = RegisteredMetric::Counter(counter);
        let rm_gauge = RegisteredMetric::Gauge(gauge);
        let rm_histogram = RegisteredMetric::Histogram(histogram);

        assert_eq!(rm_counter.name(), "my_counter");
        assert_eq!(rm_gauge.name(), "my_gauge");
        assert_eq!(rm_histogram.name(), "my_histogram");
    }

    #[test]
    fn test_registered_metric_clone() {
        let counter = Counter::new_arc("test", Labels::new());
        let rm = RegisteredMetric::Counter(counter);
        let cloned = rm.clone();
        assert_eq!(rm.name(), cloned.name());
    }

    #[test]
    fn test_registry_with_validator() {
        let validator = LabelValidator::new();
        let reg = Registry::with_validator(validator);

        // Should work with valid labels
        let result = reg.counter("test", &[("valid_key", "value")]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_histogram_with_custom_buckets() {
        let reg = Registry::new();
        let buckets = &[0.001, 0.01, 0.1, 1.0, 10.0];

        let h = reg.histogram_with_buckets("latency", &[], buckets).unwrap();
        h.observe(0.05);
        h.observe(5.0);

        assert_eq!(h.get_count(), 2);
    }

    #[test]
    fn test_registry_unregister_non_existent() {
        let reg = Registry::new();
        let removed = reg.unregister("nonexistent", &[]);
        assert!(!removed);
    }

    #[test]
    fn test_registry_valid_names() {
        let reg = Registry::new();

        // Valid names
        assert!(reg.counter("valid_name", &[]).is_ok());
        assert!(reg.counter("_underscore_start", &[]).is_ok());
        assert!(reg.counter(":colon_start", &[]).is_ok());
        assert!(reg.counter("name123", &[]).is_ok());
        assert!(reg.counter("name:with:colons", &[]).is_ok());
        assert!(reg.counter("name_with_underscores", &[]).is_ok());
    }

    #[test]
    fn test_registry_invalid_names() {
        let reg = Registry::new();

        // Invalid names
        assert!(matches!(reg.counter("", &[]), Err(RegistryError::InvalidName(_))));
        assert!(matches!(reg.counter("123start", &[]), Err(RegistryError::InvalidName(_))));
        assert!(matches!(reg.counter("has-dash", &[]), Err(RegistryError::InvalidName(_))));
        assert!(matches!(reg.counter("has.dot", &[]), Err(RegistryError::InvalidName(_))));
        assert!(matches!(reg.counter("has space", &[]), Err(RegistryError::InvalidName(_))));
    }

    #[test]
    fn test_registry_type_mismatch_gauge_to_counter() {
        let reg = Registry::new();
        reg.gauge("metric", &[]).unwrap();
        let result = reg.counter("metric", &[]);
        assert!(matches!(result, Err(RegistryError::TypeMismatch { .. })));
    }

    #[test]
    fn test_registry_type_mismatch_histogram_to_gauge() {
        let reg = Registry::new();
        reg.histogram("metric", &[]).unwrap();
        let result = reg.gauge("metric", &[]);
        assert!(matches!(result, Err(RegistryError::TypeMismatch { .. })));
    }

    #[test]
    fn test_registry_type_mismatch_counter_to_histogram() {
        let reg = Registry::new();
        reg.counter("metric", &[]).unwrap();
        let result = reg.histogram("metric", &[]);
        assert!(matches!(result, Err(RegistryError::TypeMismatch { .. })));
    }

    #[test]
    fn test_registry_cardinality_limit_gauge() {
        let reg = Registry::new().max_series(1);
        reg.gauge("metric1", &[]).unwrap();
        let result = reg.gauge("metric2", &[]);
        assert!(matches!(result, Err(RegistryError::CardinalityLimit { .. })));
    }

    #[test]
    fn test_registry_cardinality_limit_histogram() {
        let reg = Registry::new().max_series(1);
        reg.histogram("metric1", &[]).unwrap();
        let result = reg.histogram("metric2", &[]);
        assert!(matches!(result, Err(RegistryError::CardinalityLimit { .. })));
    }

    #[test]
    fn test_registry_error_display() {
        let err = RegistryError::AlreadyRegistered { name: "test".to_string() };
        assert!(err.to_string().contains("test"));

        let err = RegistryError::TypeMismatch {
            name: "metric".to_string(),
            existing: MetricType::Counter,
            requested: MetricType::Gauge,
        };
        assert!(err.to_string().contains("metric"));

        let err = RegistryError::InvalidName("bad!name".to_string());
        assert!(err.to_string().contains("bad!name"));

        let err = RegistryError::CardinalityLimit { count: 100, max: 50 };
        assert!(err.to_string().contains("100"));
    }

    #[test]
    fn test_metric_key_empty_labels() {
        let labels: Labels = Labels::new();
        let key = metric_key("test", &labels);
        assert_eq!(key, "test");
    }

    #[test]
    fn test_registry_debug() {
        let reg = Registry::new();
        let debug_str = format!("{:?}", reg);
        assert!(debug_str.contains("Registry"));
    }
}
