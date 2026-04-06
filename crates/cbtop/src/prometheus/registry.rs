//! Prometheus metrics registry.

use std::collections::HashMap;

use super::types::{
    CounterValue, GaugeValue, HistogramValue, Labels, MetricDef, MetricType, DEFAULT_MAX_LABELS,
};

/// Prometheus metrics registry
#[derive(Debug)]
pub struct MetricsRegistry {
    /// Metric definitions
    definitions: HashMap<String, MetricDef>,
    /// Gauge values
    gauges: HashMap<String, Vec<GaugeValue>>,
    /// Counter values
    counters: HashMap<String, Vec<CounterValue>>,
    /// Histogram values
    histograms: HashMap<String, Vec<HistogramValue>>,
    /// Max labels per metric
    max_labels: usize,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            gauges: HashMap::new(),
            counters: HashMap::new(),
            histograms: HashMap::new(),
            max_labels: DEFAULT_MAX_LABELS,
        }
    }

    /// Set max labels
    pub fn with_max_labels(mut self, max: usize) -> Self {
        self.max_labels = max;
        self
    }

    /// Register gauge metric
    pub fn register_gauge(&mut self, name: &str, help: &str) {
        self.definitions.insert(name.to_string(), MetricDef::new(name, help, MetricType::Gauge));
        self.gauges.entry(name.to_string()).or_default();
    }

    /// Register counter metric
    pub fn register_counter(&mut self, name: &str, help: &str) {
        self.definitions.insert(name.to_string(), MetricDef::new(name, help, MetricType::Counter));
        self.counters.entry(name.to_string()).or_default();
    }

    /// Register histogram metric
    pub fn register_histogram(&mut self, name: &str, help: &str) {
        self.definitions
            .insert(name.to_string(), MetricDef::new(name, help, MetricType::Histogram));
        self.histograms.entry(name.to_string()).or_default();
    }

    /// Set gauge value
    pub fn set_gauge(&mut self, name: &str, value: GaugeValue) -> bool {
        if value.labels.len() > self.max_labels {
            return false;
        }

        if let Some(values) = self.gauges.get_mut(name) {
            // Update existing or add new
            let label_str = value.labels.format();
            if let Some(existing) = values.iter_mut().find(|v| v.labels.format() == label_str) {
                existing.value = value.value;
                existing.timestamp = value.timestamp;
            } else {
                values.push(value);
            }
            true
        } else {
            false
        }
    }

    /// Increment counter
    pub fn inc_counter(&mut self, name: &str, labels: Labels) -> bool {
        if labels.len() > self.max_labels {
            return false;
        }

        if let Some(values) = self.counters.get_mut(name) {
            let label_str = labels.format();
            if let Some(existing) = values.iter_mut().find(|v| v.labels.format() == label_str) {
                existing.value += 1;
            } else {
                values.push(CounterValue::new(1).with_labels(labels));
            }
            true
        } else {
            false
        }
    }

    /// Add to counter
    pub fn add_counter(&mut self, name: &str, amount: u64, labels: Labels) -> bool {
        if labels.len() > self.max_labels {
            return false;
        }

        if let Some(values) = self.counters.get_mut(name) {
            let label_str = labels.format();
            if let Some(existing) = values.iter_mut().find(|v| v.labels.format() == label_str) {
                existing.value += amount;
            } else {
                values.push(CounterValue::new(amount).with_labels(labels));
            }
            true
        } else {
            false
        }
    }

    /// Observe histogram value
    pub fn observe_histogram(&mut self, name: &str, value: f64, labels: Labels) -> bool {
        if labels.len() > self.max_labels {
            return false;
        }

        if let Some(values) = self.histograms.get_mut(name) {
            let label_str = labels.format();
            if let Some(existing) = values.iter_mut().find(|v| v.labels.format() == label_str) {
                existing.observe(value);
            } else {
                let mut hist = HistogramValue::new().with_labels(labels);
                hist.observe(value);
                values.push(hist);
            }
            true
        } else {
            false
        }
    }

    /// Export as Prometheus text format
    pub fn export(&self) -> String {
        let mut lines = Vec::new();

        // Export gauges
        for (name, values) in &self.gauges {
            if let Some(def) = self.definitions.get(name) {
                lines.push(def.format_help());
                lines.push(def.format_type());
                for value in values {
                    lines.push(value.format(name));
                }
            }
        }

        // Export counters
        for (name, values) in &self.counters {
            if let Some(def) = self.definitions.get(name) {
                lines.push(def.format_help());
                lines.push(def.format_type());
                for value in values {
                    lines.push(value.format(name));
                }
            }
        }

        // Export histograms
        for (name, values) in &self.histograms {
            if let Some(def) = self.definitions.get(name) {
                lines.push(def.format_help());
                lines.push(def.format_type());
                for value in values {
                    lines.push(value.format(name));
                }
            }
        }

        lines.join("\n")
    }

    /// Get metric count
    pub fn metric_count(&self) -> usize {
        self.definitions.len()
    }

    /// Clear all values (keep definitions)
    pub fn clear_values(&mut self) {
        for values in self.gauges.values_mut() {
            values.clear();
        }
        for values in self.counters.values_mut() {
            values.clear();
        }
        for values in self.histograms.values_mut() {
            values.clear();
        }
    }
}
