//! Metrics Export Module (ENT-047)
//!
//! Export training metrics to various formats for external consumption.
//! Supports Prometheus, JSON, and realizar integration.

use super::MetricsSummary;
use std::collections::HashMap;

/// Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Prometheus text format
    Prometheus,
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

/// Metrics exporter
pub struct MetricsExporter {
    /// Metric prefix for namespacing
    prefix: String,
    /// Labels to add to all metrics
    labels: HashMap<String, String>,
}

impl MetricsExporter {
    /// Create a new exporter with prefix
    pub fn new(prefix: impl Into<String>) -> Self {
        Self { prefix: prefix.into(), labels: HashMap::new() }
    }

    /// Add a label to all exported metrics
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Export summary to Prometheus format
    pub fn to_prometheus(&self, summary: &MetricsSummary) -> String {
        let mut output = String::new();

        for (metric, stats) in summary {
            let name = format!("{}_{}", self.prefix, metric.as_str());
            let labels = self.format_labels();

            // HELP and TYPE comments
            output.push_str(&format!("# HELP {} Training metric: {}\n", name, metric.as_str()));
            output.push_str(&format!("# TYPE {name} gauge\n"));

            // Main metric (mean)
            output.push_str(&format!("{}{} {}\n", name, labels, stats.mean));

            // Additional stats as separate metrics
            output.push_str(&format!("{}_min{} {}\n", name, labels, stats.min));
            output.push_str(&format!("{}_max{} {}\n", name, labels, stats.max));
            output.push_str(&format!("{}_std{} {}\n", name, labels, stats.std));
            output.push_str(&format!("{}_count{} {}\n", name, labels, stats.count));
            output.push('\n');
        }

        output
    }

    /// Export summary to JSON format
    pub fn to_json(&self, summary: &MetricsSummary) -> Result<String, serde_json::Error> {
        let mut export: HashMap<String, serde_json::Value> = HashMap::new();

        export.insert("prefix".to_string(), self.prefix.clone().into());
        export.insert("labels".to_string(), serde_json::to_value(&self.labels)?);

        let metrics: HashMap<String, serde_json::Value> = summary
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    serde_json::json!({
                        "mean": v.mean,
                        "std": v.std,
                        "min": v.min,
                        "max": v.max,
                        "count": v.count,
                        "sum": v.sum,
                        "has_nan": v.has_nan,
                        "has_inf": v.has_inf,
                    }),
                )
            })
            .collect();

        export.insert("metrics".to_string(), serde_json::to_value(metrics)?);

        serde_json::to_string_pretty(&export)
    }

    /// Export summary to CSV format
    pub fn to_csv(&self, summary: &MetricsSummary) -> String {
        let mut output = String::from("metric,mean,std,min,max,count,sum\n");

        for (metric, stats) in summary {
            output.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                metric.as_str(),
                stats.mean,
                stats.std,
                stats.min,
                stats.max,
                stats.count,
                stats.sum
            ));
        }

        output
    }

    /// Format labels for Prometheus
    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> =
                self.labels.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

impl Default for MetricsExporter {
    fn default() -> Self {
        Self::new("entrenar")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{Metric, MetricStats};

    fn sample_summary() -> MetricsSummary {
        let mut summary = HashMap::new();
        summary.insert(
            Metric::Loss,
            MetricStats {
                count: 100,
                mean: 0.25,
                std: 0.1,
                min: 0.1,
                max: 0.5,
                sum: 25.0,
                has_nan: false,
                has_inf: false,
            },
        );
        summary.insert(
            Metric::Accuracy,
            MetricStats {
                count: 100,
                mean: 0.85,
                std: 0.05,
                min: 0.7,
                max: 0.95,
                sum: 85.0,
                has_nan: false,
                has_inf: false,
            },
        );
        summary
    }

    #[test]
    fn test_exporter_new() {
        let exporter = MetricsExporter::new("test");
        assert_eq!(exporter.prefix, "test");
    }

    #[test]
    fn test_exporter_with_labels() {
        let exporter =
            MetricsExporter::new("test").with_label("model", "v1").with_label("env", "prod");
        assert_eq!(exporter.labels.len(), 2);
    }

    #[test]
    fn test_to_prometheus() {
        let exporter = MetricsExporter::new("training");
        let summary = sample_summary();
        let prom = exporter.to_prometheus(&summary);

        assert!(prom.contains("# HELP training_loss"));
        assert!(prom.contains("# TYPE training_loss gauge"));
        assert!(prom.contains("training_loss 0.25"));
        assert!(prom.contains("training_loss_min 0.1"));
    }

    #[test]
    fn test_to_prometheus_with_labels() {
        let exporter = MetricsExporter::new("training").with_label("model", "v1");
        let summary = sample_summary();
        let prom = exporter.to_prometheus(&summary);

        assert!(prom.contains("model=\"v1\""));
    }

    #[test]
    fn test_to_json() {
        let exporter = MetricsExporter::new("test");
        let summary = sample_summary();
        let json = exporter.to_json(&summary).expect("operation should succeed");

        assert!(json.contains("\"prefix\": \"test\""));
        assert!(json.contains("\"loss\""));
        assert!(json.contains("\"mean\": 0.25"));
    }

    #[test]
    fn test_to_csv() {
        let exporter = MetricsExporter::new("test");
        let summary = sample_summary();
        let csv = exporter.to_csv(&summary);

        assert!(csv.contains("metric,mean,std"));
        assert!(csv.contains("loss,0.25,0.1"));
    }
}
