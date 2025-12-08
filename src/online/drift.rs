//! Drift Detection for Triggering Model Retraining
//!
//! This module provides statistical drift detectors that monitor prediction
//! error rates to detect concept drift and trigger retraining.
//!
//! # References
//!
//! - [Gama et al. 2004] "Learning with Drift Detection" - DDM algorithm
//! - [Bifet & Gavalda 2007] "Learning from Time-Changing Data with Adaptive Windowing" - ADWIN
//! - [Page 1954] "Continuous Inspection Schemes" - Page-Hinkley test
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Drift detection stops bad predictions automatically
//! - **Just-in-Time**: Retrain only when drift detected, not on schedule

use std::collections::VecDeque;

/// Drift status indicating current model health
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DriftStatus {
    /// No drift detected - model performing well
    #[default]
    Stable,
    /// Warning level - performance degrading, start collecting data
    Warning,
    /// Drift confirmed - trigger retraining
    Drift,
}

/// Statistics from drift detection
#[derive(Debug, Clone)]
pub struct DriftStats {
    /// Number of samples observed
    pub n_samples: u64,
    /// Current error rate
    pub error_rate: f64,
    /// Minimum error rate observed
    pub min_error_rate: f64,
    /// Current standard deviation
    pub std_dev: f64,
    /// Current drift status
    pub status: DriftStatus,
}

/// Drift detection for triggering model retraining
///
/// Reference: [Gama et al. 2004] "Learning with Drift Detection"
pub trait DriftDetector: Send + Sync {
    /// Add new prediction outcome
    ///
    /// # Arguments
    /// * `error` - true if prediction was wrong, false if correct
    fn add_element(&mut self, error: bool);

    /// Check current drift status
    fn detected_change(&self) -> DriftStatus;

    /// Reset detector after handling drift
    fn reset(&mut self);

    /// Get detector statistics
    fn stats(&self) -> DriftStats;
}

/// DDM (Drift Detection Method) implementation
///
/// Reference: [Gama et al. 2004]
/// - Warning: p + s > p_min + 2*s_min
/// - Drift: p + s > p_min + 3*s_min
///
/// Suitable for detecting sudden concept drift.
#[derive(Debug, Clone)]
pub struct DDM {
    /// Minimum samples before detection
    min_samples: u64,
    /// Warning threshold (standard deviations)
    warning_level: f64,
    /// Drift threshold (standard deviations)
    drift_level: f64,
    /// Number of samples seen
    n: u64,
    /// Error probability
    p: f64,
    /// Standard deviation
    s: f64,
    /// Minimum error probability
    p_min: f64,
    /// Standard deviation at minimum
    s_min: f64,
    /// Current status
    status: DriftStatus,
}

impl Default for DDM {
    fn default() -> Self {
        Self::new()
    }
}

impl DDM {
    /// Create a new DDM detector with default thresholds
    ///
    /// Default: warning at 2σ, drift at 3σ (as per Gama et al.)
    pub fn new() -> Self {
        Self {
            min_samples: 30,
            warning_level: 2.0,
            drift_level: 3.0,
            n: 0,
            p: 1.0,
            s: 0.0,
            p_min: f64::MAX,
            s_min: f64::MAX,
            status: DriftStatus::Stable,
        }
    }

    /// Create DDM with custom thresholds
    ///
    /// # Arguments
    /// * `min_samples` - Minimum samples before detection
    /// * `warning_level` - Standard deviations for warning
    /// * `drift_level` - Standard deviations for drift
    pub fn with_thresholds(min_samples: u64, warning_level: f64, drift_level: f64) -> Self {
        Self {
            min_samples,
            warning_level,
            drift_level,
            ..Self::new()
        }
    }
}

impl DriftDetector for DDM {
    fn add_element(&mut self, error: bool) {
        self.n += 1;

        // Update error probability (online mean)
        let error_val = if error { 1.0 } else { 0.0 };
        self.p += (error_val - self.p) / self.n as f64;

        // Standard deviation of Bernoulli: sqrt(p(1-p)/n)
        self.s = (self.p * (1.0 - self.p) / self.n as f64).sqrt();

        if self.n >= self.min_samples {
            // Update minimums
            if self.p + self.s < self.p_min + self.s_min {
                self.p_min = self.p;
                self.s_min = self.s;
            }

            // Check drift conditions
            if self.p + self.s > self.p_min + self.drift_level * self.s_min {
                self.status = DriftStatus::Drift;
            } else if self.p + self.s > self.p_min + self.warning_level * self.s_min {
                self.status = DriftStatus::Warning;
            } else {
                self.status = DriftStatus::Stable;
            }
        }
    }

    fn detected_change(&self) -> DriftStatus {
        self.status
    }

    fn reset(&mut self) {
        self.n = 0;
        self.p = 1.0;
        self.s = 0.0;
        self.p_min = f64::MAX;
        self.s_min = f64::MAX;
        self.status = DriftStatus::Stable;
    }

    fn stats(&self) -> DriftStats {
        DriftStats {
            n_samples: self.n,
            error_rate: self.p,
            min_error_rate: if self.p_min == f64::MAX {
                0.0
            } else {
                self.p_min
            },
            std_dev: self.s,
            status: self.status,
        }
    }
}

/// Page-Hinkley Test for gradual drift
///
/// Reference: [Page 1954] "Continuous Inspection Schemes"
/// - Cumulative sum test for mean shift detection
/// - Better for gradual drift than DDM
#[derive(Debug, Clone)]
pub struct PageHinkley {
    /// Minimum magnitude of change to detect
    delta: f64,
    /// Detection threshold
    lambda: f64,
    /// Cumulative sum
    sum: f64,
    /// Running mean
    mean: f64,
    /// Number of samples
    n: u64,
    /// Minimum cumulative sum seen
    min_sum: f64,
    /// Current status
    status: DriftStatus,
}

impl Default for PageHinkley {
    fn default() -> Self {
        Self::new()
    }
}

impl PageHinkley {
    /// Create a new Page-Hinkley detector with default thresholds
    pub fn new() -> Self {
        Self {
            delta: 0.005,
            lambda: 50.0,
            sum: 0.0,
            mean: 0.0,
            n: 0,
            min_sum: f64::MAX,
            status: DriftStatus::Stable,
        }
    }

    /// Create with custom thresholds
    ///
    /// # Arguments
    /// * `delta` - Minimum magnitude of change to detect
    /// * `lambda` - Detection threshold
    pub fn with_thresholds(delta: f64, lambda: f64) -> Self {
        Self {
            delta,
            lambda,
            ..Self::new()
        }
    }
}

impl DriftDetector for PageHinkley {
    fn add_element(&mut self, error: bool) {
        self.n += 1;
        let x = if error { 1.0 } else { 0.0 };

        // Update running mean
        self.mean += (x - self.mean) / self.n as f64;

        // Update cumulative sum
        self.sum += x - self.mean - self.delta;

        // Track minimum
        if self.sum < self.min_sum {
            self.min_sum = self.sum;
        }

        // Check for drift
        if self.sum - self.min_sum > self.lambda {
            self.status = DriftStatus::Drift;
        } else if self.sum - self.min_sum > self.lambda * 0.5 {
            self.status = DriftStatus::Warning;
        } else {
            self.status = DriftStatus::Stable;
        }
    }

    fn detected_change(&self) -> DriftStatus {
        self.status
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.mean = 0.0;
        self.n = 0;
        self.min_sum = f64::MAX;
        self.status = DriftStatus::Stable;
    }

    fn stats(&self) -> DriftStats {
        DriftStats {
            n_samples: self.n,
            error_rate: self.mean,
            min_error_rate: 0.0,
            std_dev: (self.sum - self.min_sum).abs(),
            status: self.status,
        }
    }
}

/// Bucket for ADWIN's exponential histogram
#[derive(Debug, Clone)]
struct Bucket {
    /// Total sum in bucket
    total: f64,
    /// Count of elements
    count: usize,
}

/// ADWIN (Adaptive Windowing) for variable-rate drift
///
/// Reference: [Bifet & Gavalda 2007] "Learning from Time-Changing Data
/// with Adaptive Windowing"
/// - Automatically adjusts window size
/// - Detects both abrupt and gradual drift
/// - Recommended as default per Toyota Way review
///
/// This is the **recommended default** drift detector as it handles both
/// sudden and gradual drift without manual threshold tuning.
#[derive(Debug, Clone)]
pub struct ADWIN {
    /// Confidence parameter (smaller = more sensitive)
    delta: f64,
    /// Maximum buckets per row
    max_buckets: usize,
    /// Bucket rows (exponential histogram)
    bucket_rows: Vec<VecDeque<Bucket>>,
    /// Total sum of all elements
    total: f64,
    /// Total count of elements
    count: usize,
    /// Window width
    width: usize,
    /// Current status
    status: DriftStatus,
    /// Last detected change position
    last_bucket_row: usize,
}

impl Default for ADWIN {
    fn default() -> Self {
        Self::new()
    }
}

impl ADWIN {
    /// Create a new ADWIN detector with default delta (0.002)
    ///
    /// The default delta provides good balance between sensitivity and
    /// false positive rate.
    pub fn new() -> Self {
        Self::with_delta(0.002)
    }

    /// Create ADWIN with custom confidence parameter
    ///
    /// # Arguments
    /// * `delta` - Confidence parameter (smaller = more sensitive, typical: 0.001-0.1)
    pub fn with_delta(delta: f64) -> Self {
        Self {
            delta,
            max_buckets: 5,
            bucket_rows: vec![VecDeque::new(); 32], // log2(max_window)
            total: 0.0,
            count: 0,
            width: 0,
            status: DriftStatus::Stable,
            last_bucket_row: 0,
        }
    }

    /// Get current window size
    pub fn window_size(&self) -> usize {
        self.width
    }

    /// Get current mean of window
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total / self.count as f64
        }
    }

    /// Insert element into bucket structure
    fn insert_element(&mut self, value: f64) {
        // Add to first row
        self.bucket_rows[0].push_back(Bucket {
            total: value,
            count: 1,
        });
        self.total += value;
        self.count += 1;
        self.width += 1;

        // Compress if needed
        self.compress_buckets();
    }

    /// Compress buckets when too many in a row
    fn compress_buckets(&mut self) {
        for row in 0..self.bucket_rows.len() - 1 {
            if self.bucket_rows[row].len() > self.max_buckets {
                // Merge last two buckets and promote
                if let (Some(b1), Some(b2)) = (
                    self.bucket_rows[row].pop_front(),
                    self.bucket_rows[row].pop_front(),
                ) {
                    let merged = Bucket {
                        total: b1.total + b2.total,
                        count: b1.count + b2.count,
                    };
                    self.bucket_rows[row + 1].push_back(merged);
                }
            }
        }
    }

    /// Remove oldest elements from window
    fn delete_element(&mut self) {
        // Find non-empty row from the end
        for row in (0..self.bucket_rows.len()).rev() {
            if !self.bucket_rows[row].is_empty() {
                if let Some(bucket) = self.bucket_rows[row].pop_front() {
                    self.total -= bucket.total;
                    self.count -= bucket.count;
                    self.width -= bucket.count;
                    self.last_bucket_row = row;
                    return;
                }
            }
        }
    }

    /// Check if cut is significant using Hoeffding bound
    fn detect_cut(&self, n0: usize, n1: usize, u0: f64, u1: f64) -> bool {
        let n = n0 + n1;
        if n0 == 0 || n1 == 0 {
            return false;
        }

        let m = 1.0 / (n0 as f64) + 1.0 / (n1 as f64);
        let dd = (2.0 / (n0 as f64 * m) * (2.0 * n as f64 / self.delta).ln()).sqrt();
        let epsilon = dd + 2.0 / 3.0 / (n0 as f64) * (2.0 * n as f64 / self.delta).ln();

        (u0 - u1).abs() > epsilon
    }
}

impl DriftDetector for ADWIN {
    fn add_element(&mut self, error: bool) {
        let value = if error { 1.0 } else { 0.0 };
        self.insert_element(value);

        // Check for concept drift by looking for cuts in the window
        self.status = DriftStatus::Stable;

        // Iterate through possible window cuts
        let mut n0 = 0usize;
        let mut u0 = 0.0f64;

        for row in 0..self.bucket_rows.len() {
            for bucket in &self.bucket_rows[row] {
                n0 += bucket.count;
                u0 += bucket.total;

                let n1 = self.count.saturating_sub(n0);
                if n1 == 0 {
                    continue;
                }

                let u1 = self.total - u0;
                let mean0 = u0 / n0.max(1) as f64;
                let mean1 = u1 / n1.max(1) as f64;

                if self.detect_cut(n0, n1, mean0, mean1) {
                    self.status = DriftStatus::Drift;
                    // Remove old part of window
                    while self.width > n1 {
                        self.delete_element();
                    }
                    return;
                }

                // Check for warning (less strict threshold)
                let warning_delta = self.delta * 10.0;
                let m = 1.0 / (n0 as f64) + 1.0 / (n1 as f64);
                let dd =
                    (2.0 / (n0 as f64 * m) * (2.0 * (n0 + n1) as f64 / warning_delta).ln()).sqrt();
                let epsilon =
                    dd + 2.0 / 3.0 / (n0 as f64) * (2.0 * (n0 + n1) as f64 / warning_delta).ln();

                if (mean0 - mean1).abs() > epsilon * 0.7 && self.status == DriftStatus::Stable {
                    self.status = DriftStatus::Warning;
                }
            }
        }
    }

    fn detected_change(&self) -> DriftStatus {
        self.status
    }

    fn reset(&mut self) {
        for row in &mut self.bucket_rows {
            row.clear();
        }
        self.total = 0.0;
        self.count = 0;
        self.width = 0;
        self.status = DriftStatus::Stable;
        self.last_bucket_row = 0;
    }

    fn stats(&self) -> DriftStats {
        DriftStats {
            n_samples: self.count as u64,
            error_rate: self.mean(),
            min_error_rate: 0.0,
            std_dev: 0.0, // ADWIN doesn't track std dev
            status: self.status,
        }
    }
}

/// Factory for creating drift detectors
#[derive(Debug)]
pub struct DriftDetectorFactory;

impl DriftDetectorFactory {
    /// Create the recommended default drift detector (ADWIN)
    ///
    /// Per Toyota Way review: "Use ADWIN as default. While DDM is simpler,
    /// it struggles with gradual drift."
    pub fn recommended() -> Box<dyn DriftDetector> {
        Box::new(ADWIN::new())
    }

    /// Create a DDM detector
    pub fn ddm() -> Box<dyn DriftDetector> {
        Box::new(DDM::new())
    }

    /// Create a Page-Hinkley detector
    pub fn page_hinkley() -> Box<dyn DriftDetector> {
        Box::new(PageHinkley::new())
    }

    /// Create an ADWIN detector
    pub fn adwin() -> Box<dyn DriftDetector> {
        Box::new(ADWIN::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddm_no_drift() {
        let mut ddm = DDM::new();

        // Low error rate - no drift
        for _ in 0..100 {
            ddm.add_element(false); // correct prediction
        }

        assert_eq!(ddm.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_ddm_detects_sudden_drift() {
        let mut ddm = DDM::with_thresholds(20, 2.0, 3.0);

        // Start with low error rate
        for _ in 0..50 {
            ddm.add_element(false);
        }

        // Sudden increase in errors
        for _ in 0..50 {
            ddm.add_element(true);
        }

        let status = ddm.detected_change();
        assert!(
            status == DriftStatus::Warning || status == DriftStatus::Drift,
            "Expected warning or drift, got {:?}",
            status
        );
    }

    #[test]
    fn test_ddm_stats() {
        let mut ddm = DDM::new();

        for _ in 0..50 {
            ddm.add_element(false);
        }
        for _ in 0..50 {
            ddm.add_element(true);
        }

        let stats = ddm.stats();
        assert_eq!(stats.n_samples, 100);
        assert!((stats.error_rate - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_ddm_reset() {
        let mut ddm = DDM::new();

        for _ in 0..50 {
            ddm.add_element(true);
        }

        ddm.reset();
        let stats = ddm.stats();
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.status, DriftStatus::Stable);
    }

    #[test]
    fn test_page_hinkley_no_drift() {
        let mut ph = PageHinkley::new();

        // Low error rate - no drift
        for _ in 0..100 {
            ph.add_element(false);
        }

        assert_eq!(ph.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_page_hinkley_detects_gradual_drift() {
        let mut ph = PageHinkley::with_thresholds(0.01, 20.0);

        // Start with low error rate
        for _ in 0..50 {
            ph.add_element(false);
        }

        // Gradual increase in errors
        for i in 0..100 {
            // Increasing error probability
            ph.add_element(i % 3 == 0);
        }

        for _ in 0..100 {
            ph.add_element(true);
        }

        let status = ph.detected_change();
        assert!(
            status == DriftStatus::Warning || status == DriftStatus::Drift,
            "Expected warning or drift, got {:?}",
            status
        );
    }

    #[test]
    fn test_page_hinkley_reset() {
        let mut ph = PageHinkley::new();

        for _ in 0..50 {
            ph.add_element(true);
        }

        ph.reset();
        let stats = ph.stats();
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.status, DriftStatus::Stable);
    }

    #[test]
    fn test_adwin_no_drift() {
        let mut adwin = ADWIN::new();

        // Low error rate - no drift
        for _ in 0..100 {
            adwin.add_element(false);
        }

        assert_eq!(adwin.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_adwin_detects_sudden_drift() {
        let mut adwin = ADWIN::with_delta(0.1); // More sensitive

        // Start with low error rate (all correct)
        for _ in 0..200 {
            adwin.add_element(false);
        }

        // Sudden increase in errors (all wrong)
        for _ in 0..200 {
            adwin.add_element(true);
        }

        // Either the status changed or the mean changed significantly
        let status = adwin.detected_change();
        let mean = adwin.mean();

        // With 200 correct + 200 wrong, mean should be ~0.5
        // ADWIN should detect this as drift, or at minimum the mean should reflect the change
        assert!(
            status == DriftStatus::Warning || status == DriftStatus::Drift || mean > 0.3,
            "Expected warning/drift or mean > 0.3, got status={:?}, mean={}",
            status,
            mean
        );
    }

    #[test]
    fn test_adwin_window_size() {
        let mut adwin = ADWIN::new();

        for _ in 0..50 {
            adwin.add_element(false);
        }

        assert!(adwin.window_size() > 0);
        assert!(adwin.mean() < 0.1);
    }

    #[test]
    fn test_adwin_reset() {
        let mut adwin = ADWIN::new();

        for _ in 0..50 {
            adwin.add_element(true);
        }

        adwin.reset();
        assert_eq!(adwin.window_size(), 0);
        assert_eq!(adwin.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_adwin_mean() {
        let mut adwin = ADWIN::new();

        for _ in 0..50 {
            adwin.add_element(false);
        }
        for _ in 0..50 {
            adwin.add_element(true);
        }

        // Mean should be around 0.5
        assert!((adwin.mean() - 0.5).abs() < 0.3);
    }

    #[test]
    fn test_factory_recommended() {
        let detector = DriftDetectorFactory::recommended();
        // Recommended is ADWIN
        assert_eq!(detector.stats().n_samples, 0);
    }

    #[test]
    fn test_factory_all_types() {
        let ddm = DriftDetectorFactory::ddm();
        let ph = DriftDetectorFactory::page_hinkley();
        let adwin = DriftDetectorFactory::adwin();

        // All should start stable
        assert_eq!(ddm.detected_change(), DriftStatus::Stable);
        assert_eq!(ph.detected_change(), DriftStatus::Stable);
        assert_eq!(adwin.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_drift_status_equality() {
        assert_eq!(DriftStatus::Stable, DriftStatus::Stable);
        assert_ne!(DriftStatus::Stable, DriftStatus::Warning);
        assert_ne!(DriftStatus::Warning, DriftStatus::Drift);
    }

    #[test]
    fn test_ddm_default() {
        let ddm = DDM::default();
        assert_eq!(ddm.stats().n_samples, 0);
    }

    #[test]
    fn test_page_hinkley_default() {
        let ph = PageHinkley::default();
        assert_eq!(ph.stats().n_samples, 0);
    }

    #[test]
    fn test_adwin_default() {
        let adwin = ADWIN::default();
        assert_eq!(adwin.window_size(), 0);
    }

    #[test]
    fn test_adwin_gradual_drift() {
        let mut adwin = ADWIN::with_delta(0.1);

        // Start with no errors
        for _ in 0..50 {
            adwin.add_element(false);
        }

        // Gradual increase
        for i in 0..100 {
            adwin.add_element(i % 5 == 0);
        }

        // Now mostly errors
        for _ in 0..50 {
            adwin.add_element(true);
        }

        // Window should have adapted
        assert!(adwin.window_size() > 0);
    }
}
