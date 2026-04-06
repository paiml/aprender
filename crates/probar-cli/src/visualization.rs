//! Enhanced TUI Visualization Module (PROBAR-SPEC-006 Section H)
//!
//! Real-time TUI dashboard and report viewing for load tests.
//! NO HTML or JavaScript - all output is TUI or binary format.

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::doc_markdown)]
#![allow(unused_variables)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;

// =============================================================================
// H.4 Export Formats - NO HTML/JavaScript
// =============================================================================

/// Export formats for load test results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ExportFormat {
    /// MessagePack binary (compact, WASM-friendly)
    #[default]
    MessagePack,
    /// JSON (human-readable, tooling integration)
    Json,
    /// NDJSON stream (newline-delimited, append-friendly)
    NdJsonStream,
    /// Binary stream (for real-time TUI consumption)
    BinaryStream,
}

impl ExportFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::MessagePack => "msgpack",
            Self::Json => "json",
            Self::NdJsonStream => "ndjson",
            Self::BinaryStream => "bin",
        }
    }

    /// Parse from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "msgpack" | "mp" => Some(Self::MessagePack),
            "json" => Some(Self::Json),
            "ndjson" | "jsonl" => Some(Self::NdJsonStream),
            "bin" | "binary" => Some(Self::BinaryStream),
            _ => None,
        }
    }
}

// =============================================================================
// H.2 Real-Time Dashboard Types
// =============================================================================

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp in milliseconds since test start
    pub timestamp_ms: u64,
    /// Value at this timestamp
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(timestamp_ms: u64, value: f64) -> Self {
        Self {
            timestamp_ms,
            value,
        }
    }
}

/// Time series for streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Series name
    pub name: String,
    /// Data points (limited buffer)
    pub points: VecDeque<DataPoint>,
    /// Maximum points to retain
    pub max_points: usize,
    /// Current value
    pub current: f64,
    /// Peak value
    pub peak: f64,
    /// Peak timestamp
    pub peak_time_ms: u64,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(name: &str, max_points: usize) -> Self {
        Self {
            name: name.to_string(),
            points: VecDeque::with_capacity(max_points),
            max_points,
            current: 0.0,
            peak: 0.0,
            peak_time_ms: 0,
        }
    }

    /// Add a data point
    pub fn push(&mut self, timestamp_ms: u64, value: f64) {
        if self.points.len() >= self.max_points {
            self.points.pop_front();
        }
        self.points.push_back(DataPoint::new(timestamp_ms, value));
        self.current = value;
        if value > self.peak {
            self.peak = value;
            self.peak_time_ms = timestamp_ms;
        }
    }

    /// Get average value
    pub fn average(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(|p| p.value).sum();
        sum / self.points.len() as f64
    }

    /// Get min value
    pub fn min(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.value)
            .fold(f64::INFINITY, f64::min)
    }
}

/// Streaming histogram for latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingHistogram {
    /// Bucket counts (index = latency_ms / bucket_size)
    buckets: Vec<u64>,
    /// Bucket size in milliseconds
    bucket_size_ms: u64,
    /// Total count
    count: u64,
    /// Sum for mean calculation
    sum: u64,
    /// Min value seen
    min: u64,
    /// Max value seen
    max: u64,
}

impl StreamingHistogram {
    /// Create a new histogram
    pub fn new(bucket_size_ms: u64, num_buckets: usize) -> Self {
        Self {
            buckets: vec![0; num_buckets],
            bucket_size_ms,
            count: 0,
            sum: 0,
            min: u64::MAX,
            max: 0,
        }
    }

    /// Record a latency value
    pub fn record(&mut self, latency_ms: u64) {
        let bucket = (latency_ms / self.bucket_size_ms) as usize;
        if bucket < self.buckets.len() {
            self.buckets[bucket] += 1;
        } else {
            // Overflow to last bucket
            if let Some(last) = self.buckets.last_mut() {
                *last += 1;
            }
        }
        self.count += 1;
        self.sum += latency_ms;
        self.min = self.min.min(latency_ms);
        self.max = self.max.max(latency_ms);
    }

    /// Get percentile value
    pub fn percentile(&self, p: u8) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = ((p as f64 / 100.0) * self.count as f64) as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return (i as u64 + 1) * self.bucket_size_ms;
            }
        }
        self.max
    }

    /// Get mean latency
    pub fn mean(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.sum / self.count
        }
    }

    /// Get count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get min
    pub fn min(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.min
        }
    }

    /// Get max
    pub fn max(&self) -> u64 {
        self.max
    }

    /// Reset histogram
    pub fn reset(&mut self) {
        self.buckets.fill(0);
        self.count = 0;
        self.sum = 0;
        self.min = u64::MAX;
        self.max = 0;
    }
}

impl Default for StreamingHistogram {
    fn default() -> Self {
        Self::new(1, 10000) // 1ms buckets, up to 10 seconds
    }
}

/// Metrics stream for real-time dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsStream {
    /// Throughput time series (req/s)
    pub throughput: TimeSeries,
    /// Latency histogram
    pub latency: StreamingHistogram,
    /// Error rate time series (%)
    pub error_rate: TimeSeries,
    /// Active users time series
    pub active_users: TimeSeries,
}

impl MetricsStream {
    /// Create a new metrics stream
    pub fn new() -> Self {
        Self {
            throughput: TimeSeries::new("throughput", 300), // 5 min at 1/s
            latency: StreamingHistogram::default(),
            error_rate: TimeSeries::new("error_rate", 300),
            active_users: TimeSeries::new("active_users", 300),
        }
    }

    /// Record a request
    pub fn record_request(&mut self, timestamp_ms: u64, latency_ms: u64, success: bool) {
        self.latency.record(latency_ms);
        if !success {
            // Error rate will be computed separately
        }
    }

    /// Update throughput
    pub fn update_throughput(&mut self, timestamp_ms: u64, requests_per_sec: f64) {
        self.throughput.push(timestamp_ms, requests_per_sec);
    }

    /// Update error rate
    pub fn update_error_rate(&mut self, timestamp_ms: u64, error_percent: f64) {
        self.error_rate.push(timestamp_ms, error_percent);
    }

    /// Update active users
    pub fn update_active_users(&mut self, timestamp_ms: u64, users: u32) {
        self.active_users.push(timestamp_ms, users as f64);
    }
}

impl Default for MetricsStream {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// H.2 Dashboard State
// =============================================================================

/// Load test stage info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageInfo {
    /// Stage name
    pub name: String,
    /// Elapsed seconds in stage
    pub elapsed_secs: u64,
    /// Total duration of stage
    pub duration_secs: u64,
    /// Target users for this stage
    pub target_users: u32,
}

/// Dashboard state for TUI rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    /// Test name/scenario
    pub test_name: String,
    /// Current stage info
    pub stage: StageInfo,
    /// Metrics stream
    pub metrics: MetricsStream,
    /// Per-endpoint stats
    pub endpoints: Vec<EndpointMetrics>,
    /// Test running
    pub running: bool,
    /// Test paused
    pub paused: bool,
    /// Total elapsed milliseconds
    pub elapsed_ms: u64,
}

impl DashboardState {
    /// Create new dashboard state
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            stage: StageInfo {
                name: "init".to_string(),
                elapsed_secs: 0,
                duration_secs: 0,
                target_users: 0,
            },
            metrics: MetricsStream::new(),
            endpoints: Vec::new(),
            running: false,
            paused: false,
            elapsed_ms: 0,
        }
    }

    /// Start the test
    pub fn start(&mut self) {
        self.running = true;
        self.paused = false;
    }

    /// Pause the test
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume the test
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Stop the test
    pub fn stop(&mut self) {
        self.running = false;
    }
}

/// Per-endpoint metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointMetrics {
    /// Endpoint name
    pub name: String,
    /// Request count
    pub count: u64,
    /// p50 latency
    pub p50_ms: u64,
    /// p95 latency
    pub p95_ms: u64,
    /// p99 latency
    pub p99_ms: u64,
    /// Error count
    pub errors: u64,
}

impl EndpointMetrics {
    /// Create new endpoint metrics
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            count: 0,
            p50_ms: 0,
            p95_ms: 0,
            p99_ms: 0,
            errors: 0,
        }
    }
}

// =============================================================================
// H.3 TUI Report Viewer
// =============================================================================

/// Report viewer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportViewerConfig {
    /// Report file path
    pub path: PathBuf,
    /// Show detailed breakdown
    pub detailed: bool,
    /// Compare with baseline
    pub baseline: Option<PathBuf>,
}

/// Report comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportComparison {
    /// Current report name
    pub current_name: String,
    /// Baseline report name
    pub baseline_name: String,
    /// Throughput change (%)
    pub throughput_change: f64,
    /// p50 latency change (%)
    pub p50_change: f64,
    /// p95 latency change (%)
    pub p95_change: f64,
    /// p99 latency change (%)
    pub p99_change: f64,
    /// Error rate change (absolute)
    pub error_rate_change: f64,
    /// Overall verdict
    pub verdict: ComparisonVerdict,
}

/// Comparison verdict
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonVerdict {
    /// Performance improved
    Improved,
    /// Performance unchanged (within tolerance)
    Unchanged,
    /// Performance regressed
    Regressed,
}

impl ComparisonVerdict {
    /// Get symbol for display
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Improved => "↑",
            Self::Unchanged => "≈",
            Self::Regressed => "↓",
        }
    }

    /// Get color hint
    pub fn color(&self) -> &'static str {
        match self {
            Self::Improved => "green",
            Self::Unchanged => "yellow",
            Self::Regressed => "red",
        }
    }
}

// =============================================================================
// TUI Rendering
// =============================================================================

/// Render dashboard as TUI string
pub fn render_dashboard(state: &DashboardState) -> String {
    let mut out = String::new();

    // Header
    out.push_str("┌─────────────────────────────────────────────────────────────────────────┐\n");
    out.push_str(&format!(
        "│ LOAD TEST: {:<30} Stage: {} ({}/{}s) │\n",
        truncate(&state.test_name, 30),
        state.stage.name,
        state.stage.elapsed_secs,
        state.stage.duration_secs
    ));
    out.push_str("├─────────────────────────────────────────────────────────────────────────┤\n");

    // Metrics summary
    out.push_str(&format!(
        "│  Throughput: {:>6.0} req/s (peak: {:.0} @ t={}s)                         │\n",
        state.metrics.throughput.current,
        state.metrics.throughput.peak,
        state.metrics.throughput.peak_time_ms / 1000
    ));
    out.push_str(&format!(
        "│  Latency:    p50={:>4}ms  p95={:>4}ms  p99={:>4}ms                       │\n",
        state.metrics.latency.percentile(50),
        state.metrics.latency.percentile(95),
        state.metrics.latency.percentile(99)
    ));
    out.push_str(&format!(
        "│  Users:      {:>4}  │  Error Rate: {:>5.2}%                              │\n",
        state.metrics.active_users.current as u32, state.metrics.error_rate.current
    ));

    // Endpoints table
    if !state.endpoints.is_empty() {
        out.push_str(
            "├─────────────────────────────────────────────────────────────────────────┤\n",
        );
        out.push_str("│  Endpoint        │ Count   │ p50    │ p95    │ p99    │ Errors │\n");
        out.push_str("│──────────────────┼─────────┼────────┼────────┼────────┼────────│\n");
        for ep in &state.endpoints {
            out.push_str(&format!(
                "│  {:<16} │ {:>7} │ {:>4}ms │ {:>4}ms │ {:>4}ms │ {:>6} │\n",
                truncate(&ep.name, 16),
                ep.count,
                ep.p50_ms,
                ep.p95_ms,
                ep.p99_ms,
                ep.errors
            ));
        }
    }

    // Footer
    out.push_str("├─────────────────────────────────────────────────────────────────────────┤\n");
    let status = if state.paused {
        "PAUSED"
    } else if state.running {
        "RUNNING"
    } else {
        "STOPPED"
    };
    out.push_str(&format!(
        "│  [q] Quit  [p] Pause  [r] Reset  [e] Export      Status: {:>8}     │\n",
        status
    ));
    out.push_str("└─────────────────────────────────────────────────────────────────────────┘\n");

    out
}

/// Render report comparison
pub fn render_comparison(comp: &ReportComparison) -> String {
    let mut out = String::new();

    out.push_str("REPORT COMPARISON\n");
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    out.push_str(&format!("Baseline: {}\n", comp.baseline_name));
    out.push_str(&format!("Current:  {}\n\n", comp.current_name));

    out.push_str("┌────────────────┬────────────┬────────────┐\n");
    out.push_str("│ Metric         │ Change     │ Verdict    │\n");
    out.push_str("├────────────────┼────────────┼────────────┤\n");

    let format_change = |change: f64| -> String {
        if change > 0.0 {
            format!("+{:.1}%", change)
        } else {
            format!("{:.1}%", change)
        }
    };

    out.push_str(&format!(
        "│ Throughput     │ {:>10} │ {:>10} │\n",
        format_change(comp.throughput_change),
        if comp.throughput_change > 5.0 {
            "↑ Better"
        } else if comp.throughput_change < -5.0 {
            "↓ Worse"
        } else {
            "≈ Same"
        }
    ));
    out.push_str(&format!(
        "│ p50 Latency    │ {:>10} │ {:>10} │\n",
        format_change(comp.p50_change),
        if comp.p50_change < -5.0 {
            "↑ Better"
        } else if comp.p50_change > 5.0 {
            "↓ Worse"
        } else {
            "≈ Same"
        }
    ));
    out.push_str(&format!(
        "│ p95 Latency    │ {:>10} │ {:>10} │\n",
        format_change(comp.p95_change),
        if comp.p95_change < -5.0 {
            "↑ Better"
        } else if comp.p95_change > 5.0 {
            "↓ Worse"
        } else {
            "≈ Same"
        }
    ));
    out.push_str(&format!(
        "│ p99 Latency    │ {:>10} │ {:>10} │\n",
        format_change(comp.p99_change),
        if comp.p99_change < -5.0 {
            "↑ Better"
        } else if comp.p99_change > 5.0 {
            "↓ Worse"
        } else {
            "≈ Same"
        }
    ));
    out.push_str(&format!(
        "│ Error Rate     │ {:>10} │ {:>10} │\n",
        format_change(comp.error_rate_change),
        if comp.error_rate_change < -0.1 {
            "↑ Better"
        } else if comp.error_rate_change > 0.1 {
            "↓ Worse"
        } else {
            "≈ Same"
        }
    ));

    out.push_str("└────────────────┴────────────┴────────────┘\n\n");

    out.push_str(&format!(
        "Overall: {} {}\n",
        comp.verdict.symbol(),
        match comp.verdict {
            ComparisonVerdict::Improved => "IMPROVED",
            ComparisonVerdict::Unchanged => "UNCHANGED",
            ComparisonVerdict::Regressed => "REGRESSED",
        }
    ));

    out
}

/// Truncate string to max length
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::MessagePack.extension(), "msgpack");
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::NdJsonStream.extension(), "ndjson");
    }

    #[test]
    fn test_export_format_from_extension() {
        assert_eq!(
            ExportFormat::from_extension("msgpack"),
            Some(ExportFormat::MessagePack)
        );
        assert_eq!(
            ExportFormat::from_extension("JSON"),
            Some(ExportFormat::Json)
        );
        assert_eq!(ExportFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_time_series() {
        let mut ts = TimeSeries::new("test", 5);
        ts.push(1000, 10.0);
        ts.push(2000, 20.0);
        ts.push(3000, 30.0);

        assert_eq!(ts.current, 30.0);
        assert_eq!(ts.peak, 30.0);
        assert_eq!(ts.average(), 20.0);
    }

    #[test]
    fn test_time_series_overflow() {
        let mut ts = TimeSeries::new("test", 3);
        for i in 0..5 {
            ts.push(i * 1000, i as f64);
        }
        assert_eq!(ts.points.len(), 3);
        assert_eq!(ts.points[0].value, 2.0);
    }

    #[test]
    fn test_streaming_histogram() {
        let mut hist = StreamingHistogram::new(10, 100);
        for i in 0..100 {
            hist.record(i);
        }
        assert_eq!(hist.count(), 100);
        assert_eq!(hist.min(), 0);
        assert_eq!(hist.max(), 99);
    }

    #[test]
    fn test_streaming_histogram_percentile() {
        let mut hist = StreamingHistogram::new(1, 1000);
        for i in 1..=100 {
            hist.record(i);
        }
        // Percentiles from histogram
        assert!(hist.percentile(50) >= 45 && hist.percentile(50) <= 55);
        assert!(hist.percentile(95) >= 90);
    }

    #[test]
    fn test_metrics_stream() {
        let mut metrics = MetricsStream::new();
        metrics.record_request(1000, 50, true);
        metrics.record_request(2000, 100, true);
        metrics.update_throughput(1000, 100.0);
        metrics.update_active_users(1000, 10);

        assert_eq!(metrics.latency.count(), 2);
        assert_eq!(metrics.throughput.current, 100.0);
        assert_eq!(metrics.active_users.current, 10.0);
    }

    #[test]
    fn test_dashboard_state() {
        let mut state = DashboardState::new("Test Load");
        assert!(!state.running);

        state.start();
        assert!(state.running);
        assert!(!state.paused);

        state.pause();
        assert!(state.paused);

        state.resume();
        assert!(!state.paused);

        state.stop();
        assert!(!state.running);
    }

    #[test]
    fn test_comparison_verdict() {
        assert_eq!(ComparisonVerdict::Improved.symbol(), "↑");
        assert_eq!(ComparisonVerdict::Unchanged.symbol(), "≈");
        assert_eq!(ComparisonVerdict::Regressed.symbol(), "↓");
    }

    #[test]
    fn test_render_dashboard() {
        let state = DashboardState::new("WASM Boot Test");
        let output = render_dashboard(&state);
        assert!(output.contains("WASM Boot Test"));
        assert!(output.contains("STOPPED"));
    }

    #[test]
    fn test_render_comparison() {
        let comp = ReportComparison {
            current_name: "v2.0".to_string(),
            baseline_name: "v1.0".to_string(),
            throughput_change: 15.0,
            p50_change: -10.0,
            p95_change: -5.0,
            p99_change: 2.0,
            error_rate_change: -0.5,
            verdict: ComparisonVerdict::Improved,
        };
        let output = render_comparison(&comp);
        assert!(output.contains("v2.0"));
        assert!(output.contains("IMPROVED"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is long", 8), "this is…");
    }

    #[test]
    fn test_endpoint_metrics() {
        let ep = EndpointMetrics::new("homepage");
        assert_eq!(ep.name, "homepage");
        assert_eq!(ep.count, 0);
    }

    #[test]
    fn test_export_format_extension_all() {
        // Cover all ExportFormat::extension() branches
        assert_eq!(ExportFormat::MessagePack.extension(), "msgpack");
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::NdJsonStream.extension(), "ndjson");
        assert_eq!(ExportFormat::BinaryStream.extension(), "bin");
    }

    #[test]
    fn test_export_format_from_extension_all_variants() {
        assert_eq!(
            ExportFormat::from_extension("msgpack"),
            Some(ExportFormat::MessagePack)
        );
        assert_eq!(
            ExportFormat::from_extension("mp"),
            Some(ExportFormat::MessagePack)
        );
        assert_eq!(
            ExportFormat::from_extension("json"),
            Some(ExportFormat::Json)
        );
        assert_eq!(
            ExportFormat::from_extension("ndjson"),
            Some(ExportFormat::NdJsonStream)
        );
        assert_eq!(
            ExportFormat::from_extension("jsonl"),
            Some(ExportFormat::NdJsonStream)
        );
        assert_eq!(
            ExportFormat::from_extension("bin"),
            Some(ExportFormat::BinaryStream)
        );
        assert_eq!(
            ExportFormat::from_extension("binary"),
            Some(ExportFormat::BinaryStream)
        );
        assert_eq!(ExportFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_data_point_creation() {
        let point = DataPoint::new(1000, 42.5);
        assert_eq!(point.timestamp_ms, 1000);
        assert_eq!(point.value, 42.5);
    }

    #[test]
    fn test_streaming_histogram_mean() {
        let mut hist = StreamingHistogram::new(1, 100);
        hist.record(10);
        hist.record(20);
        hist.record(30);
        assert_eq!(hist.mean(), 20);
    }

    #[test]
    fn test_streaming_histogram_mean_empty() {
        let hist = StreamingHistogram::new(1, 100);
        assert_eq!(hist.mean(), 0);
    }

    #[test]
    fn test_streaming_histogram_min_empty() {
        let hist = StreamingHistogram::new(1, 100);
        assert_eq!(hist.min(), 0);
    }

    #[test]
    fn test_streaming_histogram_percentile_empty() {
        let hist = StreamingHistogram::new(1, 100);
        assert_eq!(hist.percentile(50), 0);
    }

    #[test]
    fn test_streaming_histogram_reset() {
        let mut hist = StreamingHistogram::new(1, 100);
        hist.record(50);
        hist.record(100);
        assert_eq!(hist.count(), 2);

        hist.reset();
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.max(), 0);
    }

    #[test]
    fn test_streaming_histogram_default() {
        let hist = StreamingHistogram::default();
        assert_eq!(hist.count(), 0);
    }

    #[test]
    fn test_streaming_histogram_overflow_bucket() {
        let mut hist = StreamingHistogram::new(10, 10); // 10 buckets of 10ms each
        hist.record(1000); // Way over the buckets
        assert_eq!(hist.count(), 1);
    }

    #[test]
    fn test_comparison_verdict_color() {
        assert_eq!(ComparisonVerdict::Improved.color(), "green");
        assert_eq!(ComparisonVerdict::Unchanged.color(), "yellow");
        assert_eq!(ComparisonVerdict::Regressed.color(), "red");
    }

    #[test]
    fn test_metrics_stream_default() {
        let metrics = MetricsStream::default();
        assert_eq!(metrics.latency.count(), 0);
    }

    #[test]
    fn test_metrics_stream_update_error_rate() {
        let mut metrics = MetricsStream::new();
        metrics.update_error_rate(1000, 5.5);
        assert_eq!(metrics.error_rate.current, 5.5);
    }

    #[test]
    fn test_dashboard_state_with_endpoints() {
        let mut state = DashboardState::new("Endpoint Test");
        state.endpoints.push(EndpointMetrics {
            name: "homepage".to_string(),
            count: 100,
            p50_ms: 50,
            p95_ms: 100,
            p99_ms: 200,
            errors: 2,
        });
        state.start();

        let output = render_dashboard(&state);
        assert!(output.contains("homepage"));
        assert!(output.contains("RUNNING"));
    }

    #[test]
    fn test_dashboard_state_paused() {
        let mut state = DashboardState::new("Pause Test");
        state.start();
        state.pause();

        let output = render_dashboard(&state);
        assert!(output.contains("PAUSED"));
    }

    #[test]
    fn test_export_format_default() {
        let format = ExportFormat::default();
        assert!(matches!(format, ExportFormat::MessagePack));
    }

    #[test]
    fn test_stage_info_creation() {
        let stage = StageInfo {
            name: "warmup".to_string(),
            elapsed_secs: 30,
            duration_secs: 60,
            target_users: 100,
        };
        assert_eq!(stage.name, "warmup");
        assert_eq!(stage.target_users, 100);
    }

    #[test]
    fn test_report_viewer_config_creation() {
        let config = ReportViewerConfig {
            path: PathBuf::from("report.json"),
            detailed: true,
            baseline: Some(PathBuf::from("baseline.json")),
        };
        assert!(config.detailed);
        assert!(config.baseline.is_some());
    }

    #[test]
    fn test_report_comparison_creation() {
        let comp = ReportComparison {
            current_name: "current".to_string(),
            baseline_name: "baseline".to_string(),
            throughput_change: 10.0,
            p50_change: -5.0,
            p95_change: -3.0,
            p99_change: -1.0,
            error_rate_change: 0.0,
            verdict: ComparisonVerdict::Unchanged,
        };
        let output = render_comparison(&comp);
        assert!(output.contains("current"));
        assert!(output.contains("baseline"));
    }

    #[test]
    fn test_render_comparison_regressed() {
        let comp = ReportComparison {
            current_name: "new".to_string(),
            baseline_name: "old".to_string(),
            throughput_change: -20.0,
            p50_change: 50.0,
            p95_change: 80.0,
            p99_change: 100.0,
            error_rate_change: 5.0,
            verdict: ComparisonVerdict::Regressed,
        };
        let output = render_comparison(&comp);
        assert!(output.contains("REGRESSED"));
    }

    #[test]
    fn test_time_series_empty_average() {
        let ts = TimeSeries::new("empty", 10);
        assert_eq!(ts.average(), 0.0);
    }
}
