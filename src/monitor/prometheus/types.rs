//! Prometheus metric types and definitions.

/// Metric type for Prometheus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Gauge: current value that can go up or down
    Gauge,
    /// Counter: monotonically increasing value
    Counter,
    /// Histogram: distribution of values
    Histogram,
}

/// A single metric definition
#[derive(Debug, Clone)]
pub struct MetricDef {
    /// Metric name (must be valid Prometheus name)
    pub name: String,
    /// Help text describing the metric
    pub help: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Label names
    pub labels: Vec<String>,
}

impl MetricDef {
    /// Create a new gauge metric
    pub fn gauge(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            metric_type: MetricType::Gauge,
            labels: Vec::new(),
        }
    }

    /// Create a new counter metric
    pub fn counter(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            metric_type: MetricType::Counter,
            labels: Vec::new(),
        }
    }

    /// Add labels to the metric
    pub fn with_labels(mut self, labels: &[&str]) -> Self {
        self.labels = labels.iter().map(|s| (*s).to_string()).collect();
        self
    }
}

/// Label values for a specific metric series
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelSet {
    pub(crate) values: Vec<(String, String)>,
}

impl LabelSet {
    /// Create an empty label set
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Create label set from key-value pairs
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Self { values: pairs.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect() }
    }

    /// Add a label
    pub fn add(mut self, key: &str, value: &str) -> Self {
        self.values.push((key.to_string(), value.to_string()));
        self
    }

    /// Format labels for Prometheus output
    pub(crate) fn format(&self) -> String {
        if self.values.is_empty() {
            return String::new();
        }

        let parts: Vec<String> = self
            .values
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, escape_label_value(v)))
            .collect();

        format!("{{{}}}", parts.join(","))
    }
}

impl Default for LabelSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape label values for Prometheus format
pub(crate) fn escape_label_value(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// A metric value with optional labels
#[derive(Debug, Clone)]
pub(crate) struct MetricValue {
    pub(crate) labels: LabelSet,
    pub(crate) value: f64,
    pub(crate) timestamp: Option<u64>,
}
