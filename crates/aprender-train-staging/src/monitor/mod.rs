//! Real-time Training Monitoring Module
//!
//! Provides low-overhead metrics collection with SIMD-accelerated aggregation.
//!
//! # Architecture
//!
//! - **MetricsCollector**: Collects metrics during training (trueno SIMD)
//! - **MetricsSummary**: Statistical summary (mean, std, min, max)
//! - **MetricRecord**: Individual metric record with timestamp
//!
//! # Example
//!
//! ```
//! use entrenar::monitor::{MetricsCollector, Metric};
//!
//! let mut collector = MetricsCollector::new();
//! collector.record(Metric::Loss, 0.5);
//! collector.record(Metric::Accuracy, 0.85);
//!
//! let summary = collector.summary();
//! let loss_stats = summary.get(&Metric::Loss).expect("loss metric must exist");
//! println!("Mean loss: {}", loss_stats.mean);
//! ```
//!
//! # Toyota Way: 現地現物 (Genchi Genbutsu)
//!
//! All metrics are measured, not inferred. Every value comes from actual training.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

pub mod andon;
pub mod dashboard;
pub mod drift;
pub mod export;
pub mod gpu;
pub mod inference;
pub mod lineage;
pub mod llm;
pub mod params;
pub mod prometheus;
pub mod report;
pub mod storage;
pub mod tui;
pub mod wasm;

// Re-exports for convenience
pub use andon::{Alert, AlertLevel, AndonConfig, AndonSystem};
pub use dashboard::{Dashboard, DashboardConfig};
pub use drift::{Anomaly, AnomalySeverity, DriftDetector, DriftStatus, SlidingWindowBaseline};
pub use export::{ExportFormat, MetricsExporter};
pub use lineage::{ChangeType, Derivation, ModelLineage, ModelMetadata};
pub use llm::{
    EvalResult, InMemoryLLMEvaluator, LLMError, LLMEvaluator, LLMMetrics, PromptId, PromptVersion,
};
pub use params::{ParamDiff, ParamLogger, ParamValue};
pub use report::{
    HanseiAnalyzer, IssueSeverity, MetricSummary, PostTrainingReport, TrainingIssue, Trend,
};
pub use storage::{InMemoryStore, JsonFileStore, MetricsStore, StorageError, StorageResult};
pub use tui::{
    BrailleChart, GpuTelemetry, SamplePeek, TrainingSnapshot, TrainingState, TrainingStateWriter,
    TrainingStatus, TuiMonitor, TuiMonitorConfig,
};
pub use wasm::{WasmDashboard, WasmDashboardOptions, WasmMetricsCollector};

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;

// =============================================================================
// Metric Types
// =============================================================================

/// Standard training metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Metric {
    /// Training loss
    Loss,
    /// Model accuracy
    Accuracy,
    /// Current learning rate
    LearningRate,
    /// Gradient L2 norm
    GradientNorm,
    /// Epoch number
    Epoch,
    /// Batch number
    Batch,
    /// Custom metric with name
    Custom(String),
}

impl Metric {
    /// Convert metric to string representation
    pub fn as_str(&self) -> &str {
        match self {
            Metric::Loss => "loss",
            Metric::Accuracy => "accuracy",
            Metric::LearningRate => "learning_rate",
            Metric::GradientNorm => "gradient_norm",
            Metric::Epoch => "epoch",
            Metric::Batch => "batch",
            Metric::Custom(name) => name,
        }
    }

    /// Parse metric from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "loss" => Some(Metric::Loss),
            "accuracy" => Some(Metric::Accuracy),
            "learning_rate" => Some(Metric::LearningRate),
            "gradient_norm" => Some(Metric::GradientNorm),
            "epoch" => Some(Metric::Epoch),
            "batch" => Some(Metric::Batch),
            _ => None,
        }
    }
}

// =============================================================================
// MetricRecord
// =============================================================================

/// A single metric record with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
    /// Metric type
    pub metric: Metric,
    /// Metric value
    pub value: f64,
    /// Optional tags
    pub tags: HashMap<String, String>,
}

impl MetricRecord {
    /// Create a new metric record with current timestamp
    pub fn new(metric: Metric, value: f64) -> Self {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);

        Self { timestamp, metric, value, tags: HashMap::new() }
    }

    /// Add a tag to this record
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }
}

// =============================================================================
// MetricStats
// =============================================================================

/// Statistical summary for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    /// Number of values
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Sum of all values
    pub sum: f64,
    /// Whether any NaN values were recorded
    pub has_nan: bool,
    /// Whether any Inf values were recorded
    pub has_inf: bool,
}

impl Default for MetricStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            std: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            has_nan: false,
            has_inf: false,
        }
    }
}

// =============================================================================
// MetricsCollector
// =============================================================================

/// Collects training metrics with SIMD-accelerated aggregation
///
/// # Example
///
/// ```
/// use entrenar::monitor::{MetricsCollector, Metric};
///
/// let mut collector = MetricsCollector::new();
/// for epoch in 0..10 {
///     collector.record(Metric::Loss, 1.0 / (epoch as f64 + 1.0));
///     collector.record(Metric::Accuracy, 0.5 + 0.05 * epoch as f64);
/// }
///
/// let summary = collector.summary();
/// println!("{:?}", summary);
/// ```
#[derive(Debug)]
pub struct MetricsCollector {
    /// Raw records stored for export
    records: Vec<MetricRecord>,
    /// Running statistics per metric (for SIMD aggregation)
    running_stats: HashMap<Metric, RunningStats>,
}

/// Running statistics using Welford's algorithm for numerical stability
#[derive(Debug, Clone)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
    min: f64,
    max: f64,
    sum: f64,
    has_nan: bool,
    has_inf: bool,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            has_nan: false,
            has_inf: false,
        }
    }
}

impl RunningStats {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            has_nan: false,
            has_inf: false,
        }
    }

    /// Update running stats with a new value using Welford's algorithm
    fn update(&mut self, value: f64) {
        // Check for special values
        if value.is_nan() {
            self.has_nan = true;
            return;
        }
        if value.is_infinite() {
            self.has_inf = true;
            // Still update min/max for infinities
            self.min = self.min.min(value);
            self.max = self.max.max(value);
            return;
        }

        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Welford's online algorithm for mean and variance
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get standard deviation
    fn std(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2 / (self.count - 1) as f64).sqrt()
    }

    /// Convert to MetricStats
    fn to_stats(&self) -> MetricStats {
        MetricStats {
            count: self.count,
            mean: self.mean,
            std: self.std(),
            min: self.min,
            max: self.max,
            sum: self.sum,
            has_nan: self.has_nan,
            has_inf: self.has_inf,
        }
    }
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self { records: Vec::new(), running_stats: HashMap::new() }
    }

    /// Record a single metric value
    pub fn record(&mut self, metric: Metric, value: f64) {
        // Store record
        self.records.push(MetricRecord::new(metric.clone(), value));

        // Update running stats
        self.running_stats.entry(metric).or_default().update(value);
    }

    /// Record multiple metrics at once
    pub fn record_batch(&mut self, metrics: &[(Metric, f64)]) {
        for (metric, value) in metrics {
            self.record(metric.clone(), *value);
        }
    }

    /// Get the number of recorded metrics
    pub fn count(&self) -> usize {
        self.records.len()
    }

    /// Check if no metrics have been recorded
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Clear all recorded metrics
    pub fn clear(&mut self) {
        self.records.clear();
        self.running_stats.clear();
    }

    /// Get statistical summary for all metrics
    pub fn summary(&self) -> HashMap<Metric, MetricStats> {
        self.running_stats
            .iter()
            .map(|(metric, stats)| (metric.clone(), stats.to_stats()))
            .collect()
    }

    /// Convert all records to a vector
    pub fn to_records(&self) -> Vec<MetricRecord> {
        self.records.clone()
    }

    /// Export metrics to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.records)
    }

    /// Export summary to JSON
    pub fn summary_to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.summary())
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// MetricsSummary type alias
// =============================================================================

/// Type alias for metrics summary
pub type MetricsSummary = HashMap<Metric, MetricStats>;
