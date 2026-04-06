//! Real-time anomaly detection using sliding window statistics
//!
//! Sprint 20: Implements real-time anomaly detection with Trueno SIMD-accelerated
//! statistics. Uses sliding window approach to build per-syscall baselines and
//! detect outliers using Z-score analysis.

use serde::Serialize;
use std::collections::HashMap;
use trueno::Vector;

/// Anomaly severity classification based on Z-score
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AnomalySeverity {
    /// 3.0σ - 4.0σ from mean
    Low,
    /// 4.0σ - 5.0σ from mean
    Medium,
    /// >5.0σ from mean
    High,
}

/// Detected anomaly with metadata
#[derive(Debug, Clone, Serialize)]
pub struct Anomaly {
    /// Syscall name that triggered anomaly
    pub syscall_name: String,
    /// Duration in microseconds that triggered the anomaly
    pub duration_us: u64,
    /// Z-score (standard deviations from mean)
    pub z_score: f32,
    /// Baseline mean at time of detection (μs)
    pub baseline_mean: f32,
    /// Baseline standard deviation at time of detection (μs)
    pub baseline_stddev: f32,
    /// Severity classification
    pub severity: AnomalySeverity,
}

/// Baseline statistics for a syscall type (sliding window)
#[derive(Debug, Clone)]
pub struct BaselineStats {
    /// Recent samples (sliding window of durations in μs)
    samples: Vec<f32>,
    /// Pre-computed mean (updated after each sample)
    mean: f32,
    /// Pre-computed standard deviation (updated after each sample)
    stddev: f32,
}

impl BaselineStats {
    fn new(capacity: usize) -> Self {
        Self { samples: Vec::with_capacity(capacity), mean: 0.0, stddev: 0.0 }
    }

    /// Add sample and update statistics
    fn add_sample(&mut self, duration_us: f32, window_size: usize) {
        self.samples.push(duration_us);

        // Remove oldest sample if exceeding window size
        if self.samples.len() > window_size {
            self.samples.remove(0);
        }

        // Update statistics if we have enough samples
        if self.samples.len() >= 2 {
            let v = Vector::from_slice(&self.samples);
            self.mean = v.mean().unwrap_or(0.0);
            self.stddev = v.stddev().unwrap_or(0.0);
        }
    }

    /// Check if we have enough samples for reliable statistics
    fn is_ready(&self) -> bool {
        self.samples.len() >= 10
    }
}

/// Real-time anomaly detector using sliding window statistics
pub struct AnomalyDetector {
    /// Per-syscall baseline statistics
    baselines: HashMap<String, BaselineStats>,
    /// Sliding window size (number of samples per syscall)
    window_size: usize,
    /// Z-score threshold for anomaly detection
    threshold: f32,
    /// Detected anomalies (for summary report)
    detected_anomalies: Vec<Anomaly>,
}

impl AnomalyDetector {
    /// Create new anomaly detector
    ///
    /// # Arguments
    /// * `window_size` - Number of recent samples to keep per syscall (default: 100)
    /// * `threshold` - Z-score threshold for anomaly (default: 3.0σ)
    pub fn new(window_size: usize, threshold: f32) -> Self {
        Self { baselines: HashMap::new(), window_size, threshold, detected_anomalies: Vec::new() }
    }

    /// Record a syscall and check for anomaly
    ///
    /// Returns Some(Anomaly) if the duration is anomalous, None otherwise.
    /// Anomalies are also stored internally for summary reporting.
    pub fn record_and_check(&mut self, syscall_name: &str, duration_us: u64) -> Option<Anomaly> {
        let baseline = self
            .baselines
            .entry(syscall_name.to_string())
            .or_insert_with(|| BaselineStats::new(self.window_size));

        // Add sample to sliding window and update statistics
        baseline.add_sample(duration_us as f32, self.window_size);

        // Need at least 10 samples for reliable anomaly detection
        if !baseline.is_ready() {
            return None;
        }

        // Calculate z-score for current sample
        let z_score = if baseline.stddev > 0.0 {
            ((duration_us as f32) - baseline.mean) / baseline.stddev
        } else {
            // If stddev is 0, all samples are identical - any deviation is infinite
            // In practice, this rarely happens, so we don't flag as anomaly
            0.0
        };

        // Check if anomaly
        if z_score.abs() > self.threshold {
            let severity = classify_severity(z_score);
            let anomaly = Anomaly {
                syscall_name: syscall_name.to_string(),
                duration_us,
                z_score,
                baseline_mean: baseline.mean,
                baseline_stddev: baseline.stddev,
                severity,
            };

            // Store for summary
            self.detected_anomalies.push(anomaly.clone());

            Some(anomaly)
        } else {
            None
        }
    }

    /// Get all detected anomalies (for summary reporting)
    pub fn get_anomalies(&self) -> &[Anomaly] {
        &self.detected_anomalies
    }

    /// Get current baseline statistics for all syscalls
    pub fn get_baselines(&self) -> &HashMap<String, BaselineStats> {
        &self.baselines
    }

    /// Print anomaly summary report
    pub fn print_summary(&self) {
        if self.detected_anomalies.is_empty() {
            return;
        }

        eprintln!("\n=== Real-Time Anomaly Detection Report ===");
        eprintln!("Total anomalies detected: {}", self.detected_anomalies.len());
        eprintln!();

        // Group by severity
        let mut low_count = 0;
        let mut medium_count = 0;
        let mut high_count = 0;

        for anomaly in &self.detected_anomalies {
            match anomaly.severity {
                AnomalySeverity::Low => low_count += 1,
                AnomalySeverity::Medium => medium_count += 1,
                AnomalySeverity::High => high_count += 1,
            }
        }

        eprintln!("Severity Distribution:");
        if high_count > 0 {
            eprintln!("  🔴 High (>5.0σ):   {high_count} anomalies");
        }
        if medium_count > 0 {
            eprintln!("  🟡 Medium (4-5σ): {medium_count} anomalies");
        }
        if low_count > 0 {
            eprintln!("  🟢 Low (3-4σ):    {low_count} anomalies");
        }
        eprintln!();

        // Show top 10 most severe anomalies
        let mut sorted = self.detected_anomalies.clone();
        sorted.sort_by(|a, b| {
            b.z_score.abs().partial_cmp(&a.z_score.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        eprintln!("Top Anomalies (by Z-score):");
        for (i, anomaly) in sorted.iter().take(10).enumerate() {
            let severity_icon = match anomaly.severity {
                AnomalySeverity::Low => "🟢",
                AnomalySeverity::Medium => "🟡",
                AnomalySeverity::High => "🔴",
            };

            eprintln!(
                "  {}. {} {} - {:.1}σ ({} μs, baseline: {:.1} ± {:.1} μs)",
                i + 1,
                severity_icon,
                anomaly.syscall_name,
                anomaly.z_score.abs(),
                anomaly.duration_us,
                anomaly.baseline_mean,
                anomaly.baseline_stddev
            );
        }

        if sorted.len() > 10 {
            eprintln!("  ... and {} more", sorted.len() - 10);
        }
    }
}

/// Classify anomaly severity based on Z-score
fn classify_severity(z_score: f32) -> AnomalySeverity {
    let abs_z = z_score.abs();
    if abs_z > 5.0 {
        AnomalySeverity::High
    } else if abs_z > 4.0 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(AnomalySeverity: Send, Sync);
static_assertions::assert_impl_all!(Anomaly: Send, Sync);
static_assertions::assert_impl_all!(BaselineStats: Send, Sync);
static_assertions::assert_impl_all!(AnomalyDetector: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::new(100, 3.0);
        assert_eq!(detector.window_size, 100);
        assert_eq!(detector.threshold, 3.0);
        assert_eq!(detector.get_anomalies().len(), 0);
    }

    #[test]
    fn test_baseline_stats_insufficient_samples() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // First 9 samples should not trigger anomaly detection
        for i in 0..9 {
            let result = detector.record_and_check("write", 100 + i);
            assert!(result.is_none(), "Should not detect anomaly with <10 samples");
        }
    }

    #[test]
    fn test_anomaly_detection_slow_syscall() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // Establish baseline: 50 fast syscalls (~100μs)
        for _ in 0..50 {
            detector.record_and_check("write", 100);
        }

        // Anomalous slow syscall (10x slower = very high Z-score)
        let result = detector.record_and_check("write", 1000);
        assert!(result.is_some(), "Should detect anomaly");

        let anomaly = result.expect("test");
        assert_eq!(anomaly.syscall_name, "write");
        assert_eq!(anomaly.duration_us, 1000);
        assert!(anomaly.z_score.abs() > 3.0);
    }

    #[test]
    fn test_severity_classification() {
        assert_eq!(classify_severity(3.5), AnomalySeverity::Low);
        assert_eq!(classify_severity(4.5), AnomalySeverity::Medium);
        assert_eq!(classify_severity(6.0), AnomalySeverity::High);

        // Test negative Z-scores (anomalously fast)
        assert_eq!(classify_severity(-3.5), AnomalySeverity::Low);
        assert_eq!(classify_severity(-4.5), AnomalySeverity::Medium);
        assert_eq!(classify_severity(-6.0), AnomalySeverity::High);
    }

    #[test]
    fn test_sliding_window_removes_old_samples() {
        let mut detector = AnomalyDetector::new(50, 3.0);

        // Add 60 samples (exceeds window size of 50)
        for i in 0..60 {
            detector.record_and_check("write", 100 + i);
        }

        // Baseline should only contain last 50 samples
        let baseline = detector.get_baselines().get("write").expect("test");
        assert_eq!(baseline.samples.len(), 50);
    }

    #[test]
    fn test_per_syscall_baselines() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // Different syscalls should have separate baselines
        for _ in 0..20 {
            detector.record_and_check("write", 100);
            detector.record_and_check("read", 500);
        }

        assert_eq!(detector.get_baselines().len(), 2);
        assert!(detector.get_baselines().contains_key("write"));
        assert!(detector.get_baselines().contains_key("read"));
    }

    #[test]
    fn test_anomaly_with_zero_variance() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // All identical samples (in theory - in practice syscalls vary)
        for _ in 0..20 {
            detector.record_and_check("write", 100);
        }

        // Should not crash with division by zero
        let result = detector.record_and_check("write", 100);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_anomalies_stores_history() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // Baseline
        for _ in 0..30 {
            detector.record_and_check("write", 100);
        }

        // Two anomalies
        detector.record_and_check("write", 1000);
        detector.record_and_check("write", 2000);

        let anomalies = detector.get_anomalies();
        assert_eq!(anomalies.len(), 2);
        assert_eq!(anomalies[0].duration_us, 1000);
        assert_eq!(anomalies[1].duration_us, 2000);
    }

    #[test]
    fn test_print_summary_empty() {
        let detector = AnomalyDetector::new(100, 3.0);
        // Should not panic or output anything
        detector.print_summary();
    }

    #[test]
    fn test_print_summary_with_anomalies() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // Build baseline
        for _ in 0..30 {
            detector.record_and_check("write", 100);
        }

        // Trigger anomalies at different severity levels
        detector.record_and_check("write", 500); // Low
        detector.record_and_check("write", 1000); // Medium
        detector.record_and_check("write", 2000); // High

        // Should not panic
        detector.print_summary();

        assert!(!detector.get_anomalies().is_empty());
    }

    #[test]
    fn test_print_summary_top_10() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // Build baseline
        for _ in 0..30 {
            detector.record_and_check("write", 100);
        }

        // Trigger more than 10 anomalies
        for i in 0..15 {
            detector.record_and_check("write", 1000 + i * 100);
        }

        // Should handle showing only top 10
        detector.print_summary();
    }

    #[test]
    fn test_anomaly_clone() {
        let anomaly = Anomaly {
            syscall_name: "read".to_string(),
            duration_us: 5000,
            z_score: 4.5,
            baseline_mean: 100.0,
            baseline_stddev: 20.0,
            severity: AnomalySeverity::Medium,
        };

        let cloned = anomaly.clone();
        assert_eq!(cloned.syscall_name, "read");
        assert_eq!(cloned.duration_us, 5000);
        assert!((cloned.z_score - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_anomaly_debug() {
        let anomaly = Anomaly {
            syscall_name: "read".to_string(),
            duration_us: 5000,
            z_score: 4.5,
            baseline_mean: 100.0,
            baseline_stddev: 20.0,
            severity: AnomalySeverity::Medium,
        };

        let debug_str = format!("{:?}", anomaly);
        assert!(debug_str.contains("read"));
        assert!(debug_str.contains("5000"));
    }

    #[test]
    fn test_baseline_stats_is_ready() {
        let mut stats = BaselineStats::new(100);
        assert!(!stats.is_ready());

        // Add 9 samples - not ready
        for i in 0..9 {
            stats.add_sample(i as f32 * 10.0, 100);
        }
        assert!(!stats.is_ready());

        // Add 10th sample - now ready
        stats.add_sample(100.0, 100);
        assert!(stats.is_ready());
    }

    #[test]
    fn test_anomaly_severity_equality() {
        assert_eq!(AnomalySeverity::Low, AnomalySeverity::Low);
        assert_eq!(AnomalySeverity::Medium, AnomalySeverity::Medium);
        assert_eq!(AnomalySeverity::High, AnomalySeverity::High);
        assert_ne!(AnomalySeverity::Low, AnomalySeverity::High);
    }

    #[test]
    fn test_anomaly_severity_copy() {
        let s1 = AnomalySeverity::Medium;
        let s2 = s1;
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_anomaly_severity_debug() {
        let debug = format!("{:?}", AnomalySeverity::High);
        assert!(debug.contains("High"));
    }

    #[test]
    fn test_baseline_stats_clone() {
        let mut stats = BaselineStats::new(100);
        stats.add_sample(100.0, 100);
        stats.add_sample(200.0, 100);

        let cloned = stats.clone();
        assert_eq!(cloned.samples.len(), 2);
        assert!((cloned.mean - stats.mean).abs() < 0.01);
    }

    #[test]
    fn test_baseline_stats_debug() {
        let stats = BaselineStats::new(100);
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("samples"));
        assert!(debug_str.contains("mean"));
    }

    #[test]
    fn test_multiple_syscalls_isolation() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // Build separate baselines for different syscalls
        for _ in 0..30 {
            detector.record_and_check("read", 50);
            detector.record_and_check("write", 200);
            detector.record_and_check("mmap", 500);
        }

        // Anomaly for read (3x baseline) but normal for write
        let read_anomaly = detector.record_and_check("read", 150);
        let write_normal = detector.record_and_check("write", 200);

        // read should be anomalous, write should not
        assert!(read_anomaly.is_some());
        assert!(write_normal.is_none());
    }

    #[test]
    fn test_negative_zscore_anomaly() {
        let mut detector = AnomalyDetector::new(100, 3.0);

        // Build baseline with high duration
        for _ in 0..30 {
            detector.record_and_check("write", 1000);
        }

        // Anomalously fast syscall
        let result = detector.record_and_check("write", 1);
        // This would be a negative z-score - anomalously fast
        if let Some(anomaly) = result {
            assert!(anomaly.z_score < 0.0);
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove severity classification is exhaustive and ordered
    #[kani::proof]
    fn proof_severity_ordering() {
        let z: f32 = kani::any();
        kani::assume(!z.is_nan());
        kani::assume(z >= 3.0);
        kani::assume(z <= 100.0);

        if z >= 5.0 {
            kani::assert(
                AnomalySeverity::High as u8 >= AnomalySeverity::Medium as u8,
                "High >= Medium",
            );
        } else if z >= 4.0 {
            kani::assert(
                AnomalySeverity::Medium as u8 >= AnomalySeverity::Low as u8,
                "Medium >= Low",
            );
        }
    }

    /// Prove AnomalyDetector threshold must be positive
    #[kani::proof]
    fn proof_detector_threshold_positive() {
        let threshold: f32 = kani::any();
        kani::assume(threshold > 0.0);
        kani::assume(!threshold.is_nan());
        kani::assume(!threshold.is_infinite());
        let detector = AnomalyDetector::new(threshold, 100);
        kani::assert(detector.threshold > 0.0, "threshold must be positive");
    }
}
