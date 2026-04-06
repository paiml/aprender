//! Fluent report builder API.

use std::collections::HashMap;

use super::exporter::ReportExporter;
use super::types::{
    BenchmarkMetric, BenchmarkReport, ComparisonEntry, ComparisonReport, ExportFormat, ReportType,
};

/// Report builder for fluent API
#[derive(Debug)]
pub struct ReportBuilder {
    /// Report type
    report_type: ReportType,
    /// Export format
    format: ExportFormat,
    /// Title
    title: String,
    /// Benchmark metrics
    metrics: Vec<BenchmarkMetric>,
    /// Comparison entries
    comparisons: Vec<ComparisonEntry>,
    /// Metadata
    metadata: HashMap<String, String>,
}

impl ReportBuilder {
    /// Create new builder
    pub fn new(report_type: ReportType) -> Self {
        Self {
            report_type,
            format: ExportFormat::Json,
            title: format!("{} Report", report_type.name()),
            metrics: Vec::new(),
            comparisons: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set format
    pub fn format(mut self, format: ExportFormat) -> Self {
        self.format = format;
        self
    }

    /// Set title
    pub fn title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Add metric
    pub fn metric(mut self, name: &str, value: f64, unit: &str) -> Self {
        self.metrics.push(BenchmarkMetric::new(name, value, unit));
        self
    }

    /// Add comparison
    pub fn comparison(mut self, metric: &str, baseline: f64, current: f64, unit: &str) -> Self {
        self.comparisons
            .push(ComparisonEntry::new(metric, baseline, current, unit));
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Build and export
    pub fn build(self) -> String {
        match self.report_type {
            ReportType::Benchmark | ReportType::Summary => {
                let mut report = BenchmarkReport::new(&self.title);
                report.metrics = self.metrics;
                report.metadata = self.metadata;
                ReportExporter::export_benchmark(&report, self.format)
            }
            ReportType::Comparison | ReportType::Regression => {
                let mut report = ComparisonReport::new(&self.title);
                report.entries = self.comparisons;
                ReportExporter::export_comparison(&report, self.format)
            }
        }
    }
}
