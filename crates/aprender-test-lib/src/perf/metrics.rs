//! Performance Metrics
//!
//! Statistical analysis of performance data.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Statistical summary of values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean (average)
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Sample count
    pub count: usize,
}

impl Statistics {
    /// Calculate statistics from a slice of values
    #[must_use]
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::empty();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = values.len();
        let min = sorted[0];
        let max = sorted[count - 1];
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
        let p99_idx = ((count as f64 * 0.99) as usize).min(count - 1);

        Self {
            min,
            max,
            mean,
            median,
            std_dev,
            p95: sorted[p95_idx],
            p99: sorted[p99_idx],
            count,
        }
    }

    /// Create empty statistics
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            p95: 0.0,
            p99: 0.0,
            count: 0,
        }
    }

    /// Check if within acceptable range
    #[must_use]
    pub fn within_budget(&self, budget_ms: f64) -> bool {
        self.p99 <= budget_ms
    }
}

/// Frame timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetrics {
    /// Frame time in milliseconds
    pub frame_time_ms: f64,
    /// Frame number
    pub frame_number: u64,
    /// Timestamp
    pub timestamp_ms: f64,
}

impl FrameMetrics {
    /// Create new frame metrics
    #[must_use]
    pub fn new(frame_time_ms: f64) -> Self {
        Self {
            frame_time_ms,
            frame_number: 0,
            timestamp_ms: 0.0,
        }
    }

    /// Create with frame number
    #[must_use]
    pub fn with_frame_number(mut self, number: u64) -> Self {
        self.frame_number = number;
        self
    }

    /// Calculate FPS from frame time
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.frame_time_ms > 0.0 {
            1000.0 / self.frame_time_ms
        } else {
            0.0
        }
    }

    /// Check if frame meets target FPS (with small tolerance for floating-point)
    #[must_use]
    pub fn meets_target(&self, target_fps: f64) -> bool {
        const EPSILON: f64 = 1e-9;
        self.fps() >= target_fps - EPSILON
    }
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Heap bytes used
    pub heap_used: u64,
    /// Total heap size
    pub heap_total: u64,
    /// Peak memory usage
    pub peak_usage: u64,
}

impl MemoryMetrics {
    /// Create new memory metrics
    #[must_use]
    pub fn new(heap_used: u64, heap_total: u64) -> Self {
        Self {
            heap_used,
            heap_total,
            peak_usage: heap_used,
        }
    }

    /// Calculate usage percentage
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.heap_total > 0 {
            (self.heap_used as f64 / self.heap_total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Format heap used for display
    #[must_use]
    pub fn heap_used_formatted(&self) -> String {
        format_bytes(self.heap_used)
    }
}

/// Aggregate performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Frame time statistics (ms)
    pub frame_times: Statistics,
    /// Memory usage snapshots
    pub memory: Option<MemoryMetrics>,
    /// Function timing by name
    pub function_times: std::collections::HashMap<String, Statistics>,
    /// Total measurement duration
    pub duration: Duration,
}

impl PerformanceMetrics {
    /// Create from a trace
    #[must_use]
    pub fn from_trace(trace: &super::trace::Trace) -> Self {
        let mut function_times = std::collections::HashMap::new();

        // Group spans by name
        let mut by_name: std::collections::HashMap<&str, Vec<f64>> =
            std::collections::HashMap::new();
        for span in &trace.spans {
            if let Some(dur_ns) = span.duration_ns() {
                by_name
                    .entry(&span.name)
                    .or_default()
                    .push(dur_ns as f64 / 1_000_000.0);
            }
        }

        // Calculate statistics for each
        for (name, values) in by_name {
            function_times.insert(name.to_string(), Statistics::from_values(&values));
        }

        Self {
            frame_times: Statistics::empty(),
            memory: None,
            function_times,
            duration: trace.duration.unwrap_or_default(),
        }
    }

    /// Check if all metrics within budget
    #[must_use]
    pub fn within_budget(&self, frame_budget_ms: f64) -> bool {
        self.frame_times.within_budget(frame_budget_ms)
    }
}

/// Format bytes for display
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_from_values() {
        let stats = Statistics::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 5.0).abs() < f64::EPSILON);
        assert!((stats.mean - 3.0).abs() < f64::EPSILON);
        assert!((stats.median - 3.0).abs() < f64::EPSILON);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_statistics_empty() {
        let stats = Statistics::from_values(&[]);
        assert_eq!(stats.count, 0);
        assert!((stats.mean - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statistics_single_value() {
        let stats = Statistics::from_values(&[42.0]);
        assert!((stats.min - 42.0).abs() < f64::EPSILON);
        assert!((stats.max - 42.0).abs() < f64::EPSILON);
        assert!((stats.mean - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statistics_within_budget() {
        let stats = Statistics::from_values(&[10.0, 12.0, 14.0, 16.0, 18.0]);
        assert!(stats.within_budget(20.0));
        assert!(!stats.within_budget(15.0));
    }

    #[test]
    fn test_frame_metrics_fps() {
        let metrics = FrameMetrics::new(16.67);
        let fps = metrics.fps();
        assert!(fps > 59.0 && fps < 61.0);
    }

    #[test]
    fn test_frame_metrics_meets_target() {
        // 16.0ms = 62.5 FPS, clearly above 60
        let metrics = FrameMetrics::new(16.0);
        assert!(metrics.meets_target(60.0));
        assert!(!metrics.meets_target(120.0));
    }

    #[test]
    fn test_memory_metrics_usage_percent() {
        let metrics = MemoryMetrics::new(512, 1024);
        assert!((metrics.usage_percent() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_metrics_formatted() {
        let metrics = MemoryMetrics::new(1024 * 1024, 2 * 1024 * 1024);
        assert!(metrics.heap_used_formatted().contains("MB"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    // =========================================================================
    // Additional tests for 95%+ coverage
    // =========================================================================

    #[test]
    fn test_statistics_empty_method() {
        let stats = Statistics::empty();
        assert_eq!(stats.count, 0);
        assert!((stats.min - 0.0).abs() < f64::EPSILON);
        assert!((stats.max - 0.0).abs() < f64::EPSILON);
        assert!((stats.mean - 0.0).abs() < f64::EPSILON);
        assert!((stats.median - 0.0).abs() < f64::EPSILON);
        assert!((stats.std_dev - 0.0).abs() < f64::EPSILON);
        assert!((stats.p95 - 0.0).abs() < f64::EPSILON);
        assert!((stats.p99 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statistics_even_count_median() {
        // Test median calculation for even number of elements
        let stats = Statistics::from_values(&[1.0, 2.0, 3.0, 4.0]);
        // Median of [1, 2, 3, 4] = (2 + 3) / 2 = 2.5
        assert!((stats.median - 2.5).abs() < f64::EPSILON);
        assert_eq!(stats.count, 4);
    }

    #[test]
    fn test_statistics_two_values_median() {
        let stats = Statistics::from_values(&[10.0, 20.0]);
        // Median of [10, 20] = (10 + 20) / 2 = 15
        assert!((stats.median - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statistics_percentiles() {
        // Create 100 values for clear percentile testing
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let stats = Statistics::from_values(&values);

        assert_eq!(stats.count, 100);
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 100.0).abs() < f64::EPSILON);
        // p95 should be around 95
        assert!(stats.p95 >= 94.0 && stats.p95 <= 96.0);
        // p99 should be around 99
        assert!(stats.p99 >= 98.0 && stats.p99 <= 100.0);
    }

    #[test]
    fn test_statistics_std_dev() {
        // Known standard deviation case: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5, Variance = 4, Std Dev = 2
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = Statistics::from_values(&values);
        assert!((stats.mean - 5.0).abs() < f64::EPSILON);
        assert!((stats.std_dev - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statistics_with_nan_values() {
        // Test handling of NaN values in sorting
        let stats = Statistics::from_values(&[1.0, f64::NAN, 3.0]);
        // The function should handle NaN gracefully
        assert_eq!(stats.count, 3);
    }

    #[test]
    fn test_statistics_unsorted_input() {
        // Ensure sorting works correctly
        let stats = Statistics::from_values(&[5.0, 1.0, 4.0, 2.0, 3.0]);
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 5.0).abs() < f64::EPSILON);
        assert!((stats.median - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_metrics_new() {
        let metrics = FrameMetrics::new(16.67);
        assert!((metrics.frame_time_ms - 16.67).abs() < f64::EPSILON);
        assert_eq!(metrics.frame_number, 0);
        assert!((metrics.timestamp_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_metrics_with_frame_number() {
        let metrics = FrameMetrics::new(16.67).with_frame_number(42);
        assert_eq!(metrics.frame_number, 42);
        assert!((metrics.frame_time_ms - 16.67).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_metrics_fps_zero_frame_time() {
        let metrics = FrameMetrics::new(0.0);
        assert!((metrics.fps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_metrics_fps_negative_frame_time() {
        let metrics = FrameMetrics::new(-10.0);
        assert!((metrics.fps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_metrics_meets_target_edge_case() {
        // Test when FPS exactly equals target
        let metrics = FrameMetrics::new(16.666666666666668); // Exactly 60 FPS
        let fps = metrics.fps();
        assert!((fps - 60.0).abs() < 0.001);
        assert!(metrics.meets_target(60.0));
    }

    #[test]
    fn test_memory_metrics_new() {
        let metrics = MemoryMetrics::new(1024, 4096);
        assert_eq!(metrics.heap_used, 1024);
        assert_eq!(metrics.heap_total, 4096);
        assert_eq!(metrics.peak_usage, 1024); // peak starts as heap_used
    }

    #[test]
    fn test_memory_metrics_usage_percent_zero_total() {
        let metrics = MemoryMetrics::new(1024, 0);
        assert!((metrics.usage_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_metrics_usage_percent_full() {
        let metrics = MemoryMetrics::new(1024, 1024);
        assert!((metrics.usage_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_metrics_formatted_bytes() {
        let metrics = MemoryMetrics::new(500, 1000);
        assert_eq!(metrics.heap_used_formatted(), "500 B");
    }

    #[test]
    fn test_memory_metrics_formatted_kb() {
        let metrics = MemoryMetrics::new(2048, 4096);
        assert!(metrics.heap_used_formatted().contains("KB"));
    }

    #[test]
    fn test_memory_metrics_formatted_gb() {
        let metrics = MemoryMetrics::new(2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        assert!(metrics.heap_used_formatted().contains("GB"));
    }

    #[test]
    fn test_format_bytes_boundaries() {
        // Test exact boundaries
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024 - 1), "1024.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 - 1), "1024.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_large_values() {
        assert_eq!(format_bytes(10 * 1024 * 1024 * 1024), "10.00 GB");
    }

    #[test]
    fn test_performance_metrics_from_trace() {
        use super::super::trace::Tracer;

        let mut tracer = Tracer::new();
        tracer.start();

        // Create spans with known names
        for _ in 0..3 {
            let _span = tracer.span("render");
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        for _ in 0..2 {
            let _span = tracer.span("update");
            std::thread::sleep(std::time::Duration::from_micros(50));
        }

        let trace = tracer.stop();
        let metrics = PerformanceMetrics::from_trace(&trace);

        // Should have function times for both span types
        assert!(metrics.function_times.contains_key("render"));
        assert!(metrics.function_times.contains_key("update"));
        assert_eq!(metrics.function_times.get("render").unwrap().count, 3);
        assert_eq!(metrics.function_times.get("update").unwrap().count, 2);
        assert!(metrics.duration.as_nanos() > 0);
    }

    #[test]
    fn test_performance_metrics_from_empty_trace() {
        use super::super::trace::Tracer;

        let mut tracer = Tracer::new();
        tracer.start();
        let trace = tracer.stop();

        let metrics = PerformanceMetrics::from_trace(&trace);

        assert!(metrics.function_times.is_empty());
        assert!(metrics.memory.is_none());
        assert_eq!(metrics.frame_times.count, 0);
    }

    #[test]
    fn test_performance_metrics_within_budget() {
        use super::super::trace::Tracer;

        let mut tracer = Tracer::new();
        tracer.start();
        let trace = tracer.stop();

        let metrics = PerformanceMetrics::from_trace(&trace);

        // Empty frame_times should be within any budget
        assert!(metrics.within_budget(16.67));
        assert!(metrics.within_budget(0.0));
    }

    #[test]
    fn test_performance_metrics_duration_none() {
        use super::super::trace::{Trace, TraceConfig};

        // Create a trace with no duration
        let trace = Trace {
            spans: vec![],
            duration: None,
            config: TraceConfig::default(),
        };

        let metrics = PerformanceMetrics::from_trace(&trace);
        assert_eq!(metrics.duration, Duration::default());
    }

    #[test]
    fn test_performance_metrics_with_unclosed_spans() {
        use super::super::span::Span;
        use super::super::trace::{Trace, TraceConfig};

        // Create a trace with spans that have no end_ns (unclosed)
        let unclosed_span = Span::new("unclosed", 1000);
        // Note: end_ns is None, so duration_ns() returns None

        let trace = Trace {
            spans: vec![unclosed_span],
            duration: Some(Duration::from_millis(100)),
            config: TraceConfig::default(),
        };

        let metrics = PerformanceMetrics::from_trace(&trace);

        // Unclosed spans should not appear in function_times
        assert!(!metrics.function_times.contains_key("unclosed"));
    }

    #[test]
    fn test_performance_metrics_with_closed_spans() {
        use super::super::span::Span;
        use super::super::trace::{Trace, TraceConfig};

        // Create a trace with properly closed spans
        let mut span1 = Span::new("test_fn", 0);
        span1.close(1_000_000); // 1ms in ns

        let mut span2 = Span::new("test_fn", 2_000_000);
        span2.close(3_000_000); // 1ms duration

        let trace = Trace {
            spans: vec![span1, span2],
            duration: Some(Duration::from_millis(5)),
            config: TraceConfig::default(),
        };

        let metrics = PerformanceMetrics::from_trace(&trace);

        assert!(metrics.function_times.contains_key("test_fn"));
        let stats = metrics.function_times.get("test_fn").unwrap();
        assert_eq!(stats.count, 2);
        // Each span is 1ms = 1.0 in the stats
        assert!((stats.mean - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_statistics_clone_and_debug() {
        let stats = Statistics::from_values(&[1.0, 2.0, 3.0]);
        let cloned = stats.clone();
        assert_eq!(cloned.count, stats.count);

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("Statistics"));
    }

    #[test]
    fn test_frame_metrics_clone_and_debug() {
        let metrics = FrameMetrics::new(16.67).with_frame_number(10);
        let cloned = metrics.clone();
        assert_eq!(cloned.frame_number, 10);

        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("FrameMetrics"));
    }

    #[test]
    fn test_memory_metrics_clone_and_debug() {
        let metrics = MemoryMetrics::new(1024, 4096);
        let cloned = metrics.clone();
        assert_eq!(cloned.heap_used, 1024);

        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("MemoryMetrics"));
    }

    #[test]
    fn test_performance_metrics_clone_and_debug() {
        use super::super::trace::Tracer;

        let mut tracer = Tracer::new();
        tracer.start();
        let trace = tracer.stop();
        let metrics = PerformanceMetrics::from_trace(&trace);

        let cloned = metrics.clone();
        assert_eq!(cloned.frame_times.count, metrics.frame_times.count);

        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("PerformanceMetrics"));
    }

    #[test]
    fn test_statistics_serialize_deserialize() {
        let stats = Statistics::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: Statistics = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.count, stats.count);
        assert!((deserialized.mean - stats.mean).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_metrics_serialize_deserialize() {
        let metrics = FrameMetrics::new(16.67).with_frame_number(42);
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: FrameMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.frame_number, 42);
        assert!((deserialized.frame_time_ms - 16.67).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_metrics_serialize_deserialize() {
        let metrics = MemoryMetrics::new(1024, 4096);
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: MemoryMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.heap_used, 1024);
        assert_eq!(deserialized.heap_total, 4096);
    }

    #[test]
    fn test_performance_metrics_serialize_deserialize() {
        use super::super::trace::Tracer;

        let mut tracer = Tracer::new();
        tracer.start();
        {
            let _span = tracer.span("test");
        }
        let trace = tracer.stop();
        let metrics = PerformanceMetrics::from_trace(&trace);

        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.function_times.len(),
            metrics.function_times.len()
        );
    }
}
