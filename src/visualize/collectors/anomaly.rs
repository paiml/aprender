//! Anomaly collector for Z-score and Isolation Forest detection
//!
//! Detects anomalous syscall behavior using statistical methods
//! and machine learning (Isolation Forest when enabled).

use super::{Collector, MetricValue, Metrics};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Anomaly record for display
#[derive(Debug, Clone)]
pub struct AnomalyRecord {
    /// Syscall name
    pub syscall: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Z-score deviation
    pub z_score: f32,
    /// Isolation Forest anomaly score (if available)
    pub if_score: Option<f32>,
    /// Detection timestamp
    pub timestamp: Instant,
    /// Source file (from DWARF)
    pub source_file: Option<String>,
    /// Source line (from DWARF)
    pub source_line: Option<u32>,
    /// Process ID
    pub pid: i32,
}

impl AnomalyRecord {
    /// Create new anomaly record
    pub fn new(syscall: &str, duration_us: u64, z_score: f32) -> Self {
        Self {
            syscall: syscall.to_string(),
            duration_us,
            z_score,
            if_score: None,
            timestamp: Instant::now(),
            source_file: None,
            source_line: None,
            pid: 0,
        }
    }

    /// Set Isolation Forest score
    pub fn with_if_score(mut self, score: f32) -> Self {
        self.if_score = Some(score);
        self
    }

    /// Set source location
    pub fn with_source(mut self, file: &str, line: u32) -> Self {
        self.source_file = Some(file.to_string());
        self.source_line = Some(line);
        self
    }

    /// Set process ID
    pub fn with_pid(mut self, pid: i32) -> Self {
        self.pid = pid;
        self
    }

    /// Check if this is a high-severity anomaly (z > 5)
    pub fn is_high_severity(&self) -> bool {
        self.z_score > 5.0
    }
}

/// Online statistics tracker for anomaly detection
#[derive(Debug, Clone)]
pub struct OnlineStats {
    /// Running count
    count: u64,
    /// Running mean
    mean: f64,
    /// Running M2 for variance (Welford's algorithm)
    m2: f64,
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineStats {
    /// Create new online stats tracker
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    /// Update with new value using Welford's online algorithm
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get current standard deviation
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Calculate Z-score for a value
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.stddev();
        if std == 0.0 || self.count < 10 {
            0.0 // Not enough data for meaningful Z-score
        } else {
            (value - self.mean) / std
        }
    }

    /// Get sample count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Anomaly detector and collector
pub struct AnomalyCollector {
    /// Per-syscall online statistics
    stats: HashMap<String, OnlineStats>,
    /// Z-score threshold for anomaly detection
    threshold: f32,
    /// Recent anomalies (ring buffer behavior)
    anomalies: VecDeque<AnomalyRecord>,
    /// Maximum anomalies to keep
    max_anomalies: usize,
    /// Total anomaly count
    total_anomalies: u64,
    /// Current average Z-score (for display)
    avg_z_score: f64,
    /// Z-score accumulator
    z_score_sum: f64,
    /// Z-score count
    z_score_count: u64,
    /// Whether collector is available
    available: bool,
}

impl AnomalyCollector {
    /// Create new anomaly collector with default threshold (3.0σ)
    pub fn new() -> Self {
        Self::with_threshold(3.0)
    }

    /// Create anomaly collector with custom threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            stats: HashMap::new(),
            threshold,
            anomalies: VecDeque::with_capacity(100),
            max_anomalies: 100,
            total_anomalies: 0,
            avg_z_score: 0.0,
            z_score_sum: 0.0,
            z_score_count: 0,
            available: true,
        }
    }

    /// Process a syscall and check for anomaly
    ///
    /// Returns the Z-score and optional anomaly record if threshold exceeded.
    pub fn process(
        &mut self,
        syscall: &str,
        duration_us: u64,
        source_file: Option<&str>,
        source_line: Option<u32>,
        pid: i32,
    ) -> (f64, Option<AnomalyRecord>) {
        // Get or create stats for this syscall
        let stats = self.stats.entry(syscall.to_string()).or_default();

        // Calculate Z-score before updating stats
        let z_score = stats.z_score(duration_us as f64);

        // Update running statistics
        stats.update(duration_us as f64);

        // Update average Z-score
        self.z_score_sum += z_score.abs();
        self.z_score_count += 1;
        self.avg_z_score = self.z_score_sum / self.z_score_count as f64;

        // Check for anomaly
        if z_score.abs() > self.threshold as f64 {
            let mut record = AnomalyRecord::new(syscall, duration_us, z_score as f32).with_pid(pid);

            if let (Some(file), Some(line)) = (source_file, source_line) {
                record = record.with_source(file, line);
            }

            // Add to anomalies ring buffer
            if self.anomalies.len() >= self.max_anomalies {
                self.anomalies.pop_front();
            }
            self.anomalies.push_back(record.clone());
            self.total_anomalies += 1;

            (z_score, Some(record))
        } else {
            (z_score, None)
        }
    }

    /// Get recent anomalies
    pub fn anomalies(&self) -> &VecDeque<AnomalyRecord> {
        &self.anomalies
    }

    /// Get total anomaly count
    pub fn total_count(&self) -> u64 {
        self.total_anomalies
    }

    /// Get average Z-score
    pub fn avg_z_score(&self) -> f64 {
        self.avg_z_score
    }

    /// Get Z-score threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get stats for a specific syscall
    pub fn get_stats(&self, syscall: &str) -> Option<&OnlineStats> {
        self.stats.get(syscall)
    }

    /// Clear anomaly history
    pub fn clear_anomalies(&mut self) {
        self.anomalies.clear();
    }
}

impl Default for AnomalyCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for AnomalyCollector {
    fn collect(&mut self) -> Result<Metrics> {
        let mut values = HashMap::new();

        values
            .insert("anomaly.total.count".to_string(), MetricValue::Counter(self.total_anomalies));
        values.insert("anomaly.avg_z_score".to_string(), MetricValue::Gauge(self.avg_z_score));
        values.insert("anomaly.threshold".to_string(), MetricValue::Gauge(self.threshold as f64));
        values.insert(
            "anomaly.recent.count".to_string(),
            MetricValue::Gauge(self.anomalies.len() as f64),
        );

        // High severity count
        let high_severity = self.anomalies.iter().filter(|a| a.is_high_severity()).count();
        values.insert(
            "anomaly.high_severity.count".to_string(),
            MetricValue::Gauge(high_severity as f64),
        );

        Ok(Metrics::new(values))
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &'static str {
        "anomaly"
    }

    fn reset(&mut self) {
        self.stats.clear();
        self.anomalies.clear();
        self.total_anomalies = 0;
        self.avg_z_score = 0.0;
        self.z_score_sum = 0.0;
        self.z_score_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_record_new() {
        let record = AnomalyRecord::new("read", 1000, 4.5);
        assert_eq!(record.syscall, "read");
        assert_eq!(record.duration_us, 1000);
        assert!((record.z_score - 4.5).abs() < f32::EPSILON);
        assert!(!record.is_high_severity());
    }

    #[test]
    fn test_anomaly_record_high_severity() {
        let record = AnomalyRecord::new("write", 5000, 5.5);
        assert!(record.is_high_severity());
    }

    #[test]
    fn test_anomaly_record_with_source() {
        let record = AnomalyRecord::new("open", 100, 3.5).with_source("main.c", 42);
        assert_eq!(record.source_file, Some("main.c".to_string()));
        assert_eq!(record.source_line, Some(42));
    }

    #[test]
    fn test_online_stats_welford() {
        let mut stats = OnlineStats::new();

        // Add values: 10, 20, 30
        stats.update(10.0);
        stats.update(20.0);
        stats.update(30.0);

        assert_eq!(stats.count(), 3);
        assert!((stats.mean() - 20.0).abs() < f64::EPSILON);

        // Variance: ((10-20)² + (20-20)² + (30-20)²) / 2 = 200/2 = 100
        assert!((stats.variance() - 100.0).abs() < f64::EPSILON);
        assert!((stats.stddev() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_online_stats_z_score() {
        let mut stats = OnlineStats::new();

        // Add 100 values with mean=100, stddev≈10
        for i in 0..100 {
            stats.update(100.0 + (i % 20) as f64 - 10.0);
        }

        // Z-score of value at 3 stddevs
        let z = stats.z_score(stats.mean() + 3.0 * stats.stddev());
        assert!((z - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_online_stats_insufficient_data() {
        let mut stats = OnlineStats::new();
        stats.update(100.0);

        // With only 1 sample, Z-score should be 0
        assert_eq!(stats.z_score(200.0), 0.0);
    }

    #[test]
    fn test_anomaly_collector_new() {
        let collector = AnomalyCollector::new();
        assert!((collector.threshold() - 3.0).abs() < f32::EPSILON);
        assert_eq!(collector.total_count(), 0);
        assert!(collector.is_available());
    }

    #[test]
    fn test_anomaly_collector_process_normal() {
        let mut collector = AnomalyCollector::new();

        // Add baseline data
        for _ in 0..100 {
            collector.process("read", 100, None, None, 1234);
        }

        // Normal value should not trigger anomaly
        let (z, anomaly) = collector.process("read", 100, None, None, 1234);
        assert!(z.abs() < 1.0);
        assert!(anomaly.is_none());
    }

    #[test]
    fn test_anomaly_collector_process_anomaly() {
        let mut collector = AnomalyCollector::with_threshold(3.0);

        // Add baseline data (100μs ± small variation)
        for i in 0..100 {
            let duration = 100 + (i % 5); // 100-104μs
            collector.process("read", duration, None, None, 1234);
        }

        // Extreme value should trigger anomaly
        let (z, anomaly) = collector.process("read", 10000, Some("main.c"), Some(42), 1234);
        assert!(z > 3.0);
        assert!(anomaly.is_some());

        let record = anomaly.unwrap();
        assert_eq!(record.syscall, "read");
        assert_eq!(record.duration_us, 10000);
        assert_eq!(record.source_file, Some("main.c".to_string()));
        assert_eq!(record.source_line, Some(42));
    }

    #[test]
    fn test_anomaly_collector_ring_buffer() {
        let mut collector = AnomalyCollector::with_threshold(0.001); // Very low threshold
        collector.max_anomalies = 10;

        // Add baseline data first (need 10+ samples for z-score)
        for _ in 0..15 {
            collector.process("read", 100, None, None, 1234);
        }

        // Clear the initial anomalies
        collector.anomalies.clear();
        let initial_total = collector.total_anomalies;

        // Now add varying values that will trigger anomalies
        for i in 0..20 {
            collector.process("read", 100 + i * 50, None, None, 1234); // Varying durations
        }

        // Should have some anomalies (varying durations create z-scores)
        // And should be limited to max_anomalies
        assert!(collector.anomalies().len() <= 10);
        assert!(collector.total_count() > initial_total);
    }

    #[test]
    fn test_anomaly_collector_collect() {
        let mut collector = AnomalyCollector::new();
        collector.process("read", 100, None, None, 1234);

        let metrics = collector.collect().unwrap();
        assert!(metrics.values.contains_key("anomaly.total.count"));
        assert!(metrics.values.contains_key("anomaly.avg_z_score"));
    }

    #[test]
    fn test_anomaly_collector_reset() {
        let mut collector = AnomalyCollector::with_threshold(2.0);

        // Build baseline with small variation (100 ± 5)
        for i in 0..20 {
            let duration = 100 + (i % 10); // 100-109 range
            collector.process("read", duration, None, None, 1234);
        }

        // Add clearly anomalous value (100x normal)
        collector.process("read", 10000, None, None, 1234);

        assert!(
            !collector.anomalies().is_empty(),
            "Expected at least one anomaly after extreme value"
        );
        assert!(collector.total_count() > 0);

        collector.reset();
        assert!(collector.anomalies().is_empty());
        assert_eq!(collector.total_count(), 0);
        assert!(collector.stats.is_empty());
    }
}
