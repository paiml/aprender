//! Quality degradation detection for synthetic data.

use std::collections::VecDeque;

/// Detects when synthetic data is hurting rather than helping model performance.
///
/// Tracks rolling performance scores and compares against a baseline trained
/// without synthetic data. If performance degrades, synthetic data should be
/// disabled or reconfigured.
///
/// # Example
///
/// ```
/// use aprender::synthetic::QualityDegradationDetector;
///
/// let mut detector = QualityDegradationDetector::new(0.85, 5);
///
/// // Record scores from training with synthetic data
/// detector.record(0.87);  // Better than baseline
/// detector.record(0.86);
/// detector.record(0.82);  // Getting worse
/// detector.record(0.80);
/// detector.record(0.78);  // Degraded below baseline
///
/// assert!(detector.should_disable_synthetic());
/// ```
#[derive(Debug, Clone)]
pub struct QualityDegradationDetector {
    /// Baseline score without synthetic data.
    baseline_score: f32,
    /// Minimum improvement required to justify synthetic data.
    min_improvement: f32,
    /// Rolling window of recent scores.
    recent_scores: VecDeque<f32>,
    /// Maximum window size.
    window_size: usize,
    /// Total evaluations recorded.
    total_evaluations: usize,
}

impl QualityDegradationDetector {
    /// Create a new degradation detector.
    ///
    /// # Arguments
    ///
    /// * `baseline_score` - Performance score without synthetic data
    /// * `window_size` - Number of recent scores to track
    #[must_use]
    pub fn new(baseline_score: f32, window_size: usize) -> Self {
        Self {
            baseline_score,
            min_improvement: 0.0,
            recent_scores: VecDeque::with_capacity(window_size.max(1)),
            window_size: window_size.max(1),
            total_evaluations: 0,
        }
    }

    /// Set minimum improvement threshold.
    ///
    /// Synthetic data is considered harmful if it doesn't improve
    /// performance by at least this amount.
    #[must_use]
    pub fn with_min_improvement(mut self, improvement: f32) -> Self {
        self.min_improvement = improvement.max(0.0);
        self
    }

    /// Get the baseline score.
    #[must_use]
    pub fn baseline_score(&self) -> f32 {
        self.baseline_score
    }

    /// Update the baseline score.
    pub fn set_baseline(&mut self, score: f32) {
        self.baseline_score = score;
    }

    /// Record a new evaluation score.
    pub fn record(&mut self, score: f32) {
        if self.recent_scores.len() >= self.window_size {
            self.recent_scores.pop_front();
        }
        self.recent_scores.push_back(score);
        self.total_evaluations += 1;
    }

    /// Get the number of recorded scores in the window.
    #[must_use]
    pub fn score_count(&self) -> usize {
        self.recent_scores.len()
    }

    /// Get total number of evaluations recorded.
    #[must_use]
    pub fn total_evaluations(&self) -> usize {
        self.total_evaluations
    }

    /// Get the mean of recent scores.
    #[must_use]
    pub fn mean_score(&self) -> f32 {
        if self.recent_scores.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.recent_scores.iter().sum();
        sum / self.recent_scores.len() as f32
    }

    /// Get the most recent score.
    #[must_use]
    pub fn latest_score(&self) -> Option<f32> {
        self.recent_scores.back().copied()
    }

    /// Get the best score in the window.
    #[must_use]
    pub fn best_score(&self) -> f32 {
        self.recent_scores
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Get the worst score in the window.
    #[must_use]
    pub fn worst_score(&self) -> f32 {
        self.recent_scores
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b))
    }

    /// Check if synthetic data should be disabled.
    ///
    /// Returns true if recent mean score is below baseline - `min_improvement`.
    #[must_use]
    pub fn should_disable_synthetic(&self) -> bool {
        if self.recent_scores.is_empty() {
            return false;
        }
        self.mean_score() < self.baseline_score - self.min_improvement
    }

    /// Check if synthetic data is providing improvement.
    ///
    /// Returns true if recent mean score is above baseline + `min_improvement`.
    #[must_use]
    pub fn is_improving(&self) -> bool {
        if self.recent_scores.is_empty() {
            return false;
        }
        self.mean_score() > self.baseline_score + self.min_improvement
    }

    /// Get the improvement over baseline.
    ///
    /// Positive values indicate synthetic data is helping.
    #[must_use]
    pub fn improvement(&self) -> f32 {
        self.mean_score() - self.baseline_score
    }

    /// Check if scores are trending downward.
    #[must_use]
    pub fn is_trending_down(&self) -> bool {
        if self.recent_scores.len() < 4 {
            return false;
        }

        let mid = self.recent_scores.len() / 2;
        let first_half: f32 = self.recent_scores.iter().take(mid).sum();
        let second_half: f32 = self.recent_scores.iter().skip(mid).sum();

        let first_avg = first_half / mid as f32;
        let second_avg = second_half / (self.recent_scores.len() - mid) as f32;

        second_avg < first_avg * 0.95 // 5% decline threshold
    }

    /// Get score variance in the window.
    #[must_use]
    pub fn variance(&self) -> f32 {
        if self.recent_scores.len() < 2 {
            return 0.0;
        }

        let mean = self.mean_score();
        let sum_sq: f32 = self.recent_scores.iter().map(|s| (s - mean).powi(2)).sum();

        sum_sq / (self.recent_scores.len() - 1) as f32
    }

    /// Get score standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Clear all recorded scores (keeps baseline).
    pub fn reset(&mut self) {
        self.recent_scores.clear();
        self.total_evaluations = 0;
    }

    /// Get a summary of the current state.
    #[must_use]
    pub fn summary(&self) -> QualitySummary {
        QualitySummary {
            baseline: self.baseline_score,
            mean: self.mean_score(),
            improvement: self.improvement(),
            should_disable: self.should_disable_synthetic(),
            is_trending_down: self.is_trending_down(),
            sample_count: self.recent_scores.len(),
        }
    }
}

/// Summary of quality detection state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualitySummary {
    /// Baseline score without synthetic data.
    pub baseline: f32,
    /// Mean of recent scores.
    pub mean: f32,
    /// Improvement over baseline.
    pub improvement: f32,
    /// Whether synthetic data should be disabled.
    pub should_disable: bool,
    /// Whether scores are trending down.
    pub is_trending_down: bool,
    /// Number of scores in window.
    pub sample_count: usize,
}

impl Default for QualityDegradationDetector {
    fn default() -> Self {
        Self::new(0.0, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_detector() {
        let detector = QualityDegradationDetector::new(0.85, 5);

        assert!((detector.baseline_score() - 0.85).abs() < f32::EPSILON);
        assert_eq!(detector.score_count(), 0);
        assert_eq!(detector.total_evaluations(), 0);
    }

    #[test]
    fn test_window_size_minimum() {
        let detector = QualityDegradationDetector::new(0.5, 0);
        assert_eq!(detector.window_size, 1);
    }

    #[test]
    fn test_with_min_improvement() {
        let detector = QualityDegradationDetector::new(0.85, 5).with_min_improvement(0.02);

        // Score at baseline should not trigger disable
        let mut d = detector.clone();
        d.record(0.85);
        assert!(!d.should_disable_synthetic());

        // Score below baseline - min_improvement should trigger
        let mut d = detector.clone();
        d.record(0.82);
        assert!(d.should_disable_synthetic());
    }

    #[test]
    fn test_negative_min_improvement_clamped() {
        let detector = QualityDegradationDetector::new(0.85, 5).with_min_improvement(-0.5);
        assert!((detector.min_improvement - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_record_and_count() {
        let mut detector = QualityDegradationDetector::new(0.85, 3);

        detector.record(0.86);
        assert_eq!(detector.score_count(), 1);
        assert_eq!(detector.total_evaluations(), 1);

        detector.record(0.87);
        detector.record(0.88);
        assert_eq!(detector.score_count(), 3);
        assert_eq!(detector.total_evaluations(), 3);

        // Window overflow
        detector.record(0.89);
        assert_eq!(detector.score_count(), 3);
        assert_eq!(detector.total_evaluations(), 4);
    }

    #[test]
    fn test_mean_score() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);

        assert!((detector.mean_score() - 0.0).abs() < f32::EPSILON);

        detector.record(0.80);
        detector.record(0.90);
        assert!((detector.mean_score() - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_latest_score() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);

        assert!(detector.latest_score().is_none());

        detector.record(0.86);
        assert!((detector.latest_score().expect("should have score") - 0.86).abs() < f32::EPSILON);

        detector.record(0.88);
        assert!((detector.latest_score().expect("should have score") - 0.88).abs() < f32::EPSILON);
    }

    #[test]
    fn test_best_worst_score() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);

        detector.record(0.82);
        detector.record(0.88);
        detector.record(0.85);

        assert!((detector.best_score() - 0.88).abs() < f32::EPSILON);
        assert!((detector.worst_score() - 0.82).abs() < f32::EPSILON);
    }

    #[test]
    fn test_should_disable_synthetic() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);

        // No data = don't disable
        assert!(!detector.should_disable_synthetic());

        // Above baseline = don't disable
        detector.record(0.87);
        assert!(!detector.should_disable_synthetic());

        // Reset and add degraded scores
        detector.reset();
        detector.record(0.80);
        detector.record(0.78);
        detector.record(0.76);
        assert!(detector.should_disable_synthetic());
    }

    #[test]
    fn test_is_improving() {
        let mut detector = QualityDegradationDetector::new(0.85, 5).with_min_improvement(0.02);

        // No data
        assert!(!detector.is_improving());

        // Below threshold
        detector.record(0.86);
        assert!(!detector.is_improving());

        // Above threshold
        detector.reset();
        detector.record(0.90);
        assert!(detector.is_improving());
    }

    #[test]
    fn test_improvement() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);

        detector.record(0.90);
        assert!((detector.improvement() - 0.05).abs() < f32::EPSILON);

        detector.reset();
        detector.record(0.80);
        assert!((detector.improvement() - (-0.05)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_trending_down() {
        let mut detector = QualityDegradationDetector::new(0.85, 10);

        // Not enough data
        detector.record(0.90);
        detector.record(0.80);
        assert!(!detector.is_trending_down());

        // Add declining trend
        detector.record(0.85);
        detector.record(0.85);
        detector.record(0.75);
        detector.record(0.70);

        assert!(detector.is_trending_down());
    }

    #[test]
    fn test_variance_and_std_dev() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);

        // Not enough data
        assert!((detector.variance() - 0.0).abs() < f32::EPSILON);

        detector.record(0.80);
        assert!((detector.variance() - 0.0).abs() < f32::EPSILON);

        // Same values = zero variance
        detector.record(0.80);
        assert!((detector.variance() - 0.0).abs() < f32::EPSILON);

        // Different values
        detector.record(0.90);
        assert!(detector.variance() > 0.0);
        assert!(detector.std_dev() > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);
        detector.record(0.86);
        detector.record(0.87);

        detector.reset();

        assert_eq!(detector.score_count(), 0);
        assert_eq!(detector.total_evaluations(), 0);
        assert!((detector.baseline_score() - 0.85).abs() < f32::EPSILON); // Baseline preserved
    }

    #[test]
    fn test_set_baseline() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);
        detector.set_baseline(0.90);
        assert!((detector.baseline_score() - 0.90).abs() < f32::EPSILON);
    }

    #[test]
    fn test_summary() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);
        detector.record(0.80);
        detector.record(0.80);

        let summary = detector.summary();

        assert!((summary.baseline - 0.85).abs() < f32::EPSILON);
        assert!((summary.mean - 0.80).abs() < f32::EPSILON);
        assert!((summary.improvement - (-0.05)).abs() < f32::EPSILON);
        assert!(summary.should_disable);
        assert!(!summary.is_trending_down); // Not enough data for trend
        assert_eq!(summary.sample_count, 2);
    }

    #[test]
    fn test_default() {
        let detector = QualityDegradationDetector::default();
        assert!((detector.baseline_score() - 0.0).abs() < f32::EPSILON);
        assert_eq!(detector.window_size, 10);
    }

    #[test]
    fn test_clone() {
        let mut detector = QualityDegradationDetector::new(0.85, 5);
        detector.record(0.86);

        let cloned = detector.clone();
        assert_eq!(cloned.score_count(), 1);
        assert!((cloned.baseline_score() - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_summary_debug() {
        let summary = QualitySummary {
            baseline: 0.85,
            mean: 0.80,
            improvement: -0.05,
            should_disable: true,
            is_trending_down: false,
            sample_count: 5,
        };

        let debug = format!("{summary:?}");
        assert!(debug.contains("QualitySummary"));
        assert!(debug.contains("baseline"));
    }
}
