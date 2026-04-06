//! Data types for the export reporting system.

use std::collections::HashMap;

/// Export format type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON format for API/CI integration
    Json,
    /// CSV format for spreadsheet analysis
    Csv,
    /// Markdown format for documentation
    Markdown,
    /// HTML format for interactive reports
    Html,
}

impl ExportFormat {
    /// Get format name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Markdown => "markdown",
            Self::Html => "html",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Markdown => "md",
            Self::Html => "html",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            "markdown" | "md" => Some(Self::Markdown),
            "html" => Some(Self::Html),
            _ => None,
        }
    }
}

/// Report type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportType {
    /// Benchmark results report
    Benchmark,
    /// Comparison report (baseline vs current)
    Comparison,
    /// Regression detection report
    Regression,
    /// Executive summary report
    Summary,
}

impl ReportType {
    /// Get report type name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Benchmark => "benchmark",
            Self::Comparison => "comparison",
            Self::Regression => "regression",
            Self::Summary => "summary",
        }
    }
}

/// Benchmark metric data
#[derive(Debug, Clone)]
pub struct BenchmarkMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Unit of measurement
    pub unit: String,
}

impl BenchmarkMetric {
    /// Create new metric
    pub fn new(name: &str, value: f64, unit: &str) -> Self {
        Self { name: name.to_string(), value, unit: unit.to_string() }
    }
}

/// Benchmark report data
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Report title
    pub title: String,
    /// Timestamp
    pub timestamp: String,
    /// Workload name
    pub workload: String,
    /// Backend used
    pub backend: String,
    /// Problem size
    pub size: usize,
    /// Metrics
    pub metrics: Vec<BenchmarkMetric>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self {
            title: "Benchmark Report".to_string(),
            timestamp: "".to_string(),
            workload: "".to_string(),
            backend: "".to_string(),
            size: 0,
            metrics: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl BenchmarkReport {
    /// Create new benchmark report
    pub fn new(title: &str) -> Self {
        Self { title: title.to_string(), ..Default::default() }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.timestamp = ts.to_string();
        self
    }

    /// Set workload
    pub fn with_workload(mut self, workload: &str) -> Self {
        self.workload = workload.to_string();
        self
    }

    /// Set backend
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_string();
        self
    }

    /// Set size
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Add metric
    pub fn add_metric(&mut self, name: &str, value: f64, unit: &str) {
        self.metrics.push(BenchmarkMetric::new(name, value, unit));
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// Comparison entry (baseline vs current)
#[derive(Debug, Clone)]
pub struct ComparisonEntry {
    /// Metric name
    pub metric: String,
    /// Baseline value
    pub baseline: f64,
    /// Current value
    pub current: f64,
    /// Unit
    pub unit: String,
}

impl ComparisonEntry {
    /// Create new comparison entry
    pub fn new(metric: &str, baseline: f64, current: f64, unit: &str) -> Self {
        Self { metric: metric.to_string(), baseline, current, unit: unit.to_string() }
    }

    /// Calculate percent change
    pub fn percent_change(&self) -> f64 {
        if self.baseline.abs() < 1e-10 {
            if self.current.abs() < 1e-10 {
                0.0
            } else {
                100.0
            }
        } else {
            ((self.current - self.baseline) / self.baseline) * 100.0
        }
    }

    /// Check if regressed (positive change is bad for latency, negative is bad for throughput)
    pub fn is_regression(&self, threshold_percent: f64) -> bool {
        self.percent_change().abs() > threshold_percent
    }
}

/// Comparison report data
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Report title
    pub title: String,
    /// Baseline label
    pub baseline_label: String,
    /// Current label
    pub current_label: String,
    /// Comparison entries
    pub entries: Vec<ComparisonEntry>,
    /// Regression threshold (%)
    pub regression_threshold: f64,
}

impl Default for ComparisonReport {
    fn default() -> Self {
        Self {
            title: "Comparison Report".to_string(),
            baseline_label: "baseline".to_string(),
            current_label: "current".to_string(),
            entries: Vec::new(),
            regression_threshold: 5.0,
        }
    }
}

impl ComparisonReport {
    /// Create new comparison report
    pub fn new(title: &str) -> Self {
        Self { title: title.to_string(), ..Default::default() }
    }

    /// Set labels
    pub fn with_labels(mut self, baseline: &str, current: &str) -> Self {
        self.baseline_label = baseline.to_string();
        self.current_label = current.to_string();
        self
    }

    /// Set regression threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.regression_threshold = threshold;
        self
    }

    /// Add comparison entry
    pub fn add_entry(&mut self, metric: &str, baseline: f64, current: f64, unit: &str) {
        self.entries.push(ComparisonEntry::new(metric, baseline, current, unit));
    }

    /// Get regressions
    pub fn get_regressions(&self) -> Vec<&ComparisonEntry> {
        self.entries.iter().filter(|e| e.is_regression(self.regression_threshold)).collect()
    }

    /// Check if any regressions exist
    pub fn has_regressions(&self) -> bool {
        !self.get_regressions().is_empty()
    }
}
