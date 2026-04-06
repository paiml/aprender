//! Diversity monitoring for synthetic data generation.

use std::collections::VecDeque;

/// Diversity metrics for a batch of generated samples.
///
/// Tracks multiple diversity indicators to detect mode collapse
/// or distribution shift in synthetic data generation.
///
/// # Example
///
/// ```
/// use aprender::synthetic::DiversityScore;
///
/// let score = DiversityScore::new(0.5, 0.2, 0.8);
/// // indicates_collapse returns true if mean_distance < threshold
/// assert!(!score.indicates_collapse(0.3));  // 0.5 >= 0.3, no collapse
/// assert!(score.indicates_collapse(0.6));   // 0.5 < 0.6, collapse detected
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiversityScore {
    /// Mean pairwise distance between samples.
    pub mean_distance: f32,
    /// Minimum pairwise distance (closest pair).
    pub min_distance: f32,
    /// Coverage estimate (fraction of space covered).
    pub coverage: f32,
}

impl DiversityScore {
    /// Create a new diversity score.
    #[must_use]
    pub fn new(mean_distance: f32, min_distance: f32, coverage: f32) -> Self {
        Self {
            mean_distance,
            min_distance,
            coverage,
        }
    }

    /// Create a zero score (no diversity).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            mean_distance: 0.0,
            min_distance: 0.0,
            coverage: 0.0,
        }
    }

    /// Check if diversity indicates potential mode collapse.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum acceptable mean distance
    #[must_use]
    pub fn indicates_collapse(&self, threshold: f32) -> bool {
        self.mean_distance < threshold
    }

    /// Get a combined diversity metric (average of normalized components).
    #[must_use]
    pub fn combined(&self) -> f32 {
        (self.mean_distance + self.min_distance + self.coverage) / 3.0
    }
}

impl Default for DiversityScore {
    fn default() -> Self {
        Self::zero()
    }
}

/// Monitors diversity of generated synthetic samples over time.
///
/// Tracks rolling statistics to detect trends like mode collapse
/// or decreasing diversity during generation.
///
/// # Example
///
/// ```
/// use aprender::synthetic::{DiversityMonitor, DiversityScore};
///
/// let mut monitor = DiversityMonitor::new(5);
///
/// monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
/// monitor.record(DiversityScore::new(0.4, 0.2, 0.6));
///
/// assert_eq!(monitor.sample_count(), 2);
/// assert!(monitor.mean_diversity() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct DiversityMonitor {
    /// Rolling window of diversity scores.
    scores: VecDeque<DiversityScore>,
    /// Maximum window size.
    window_size: usize,
    /// Collapse threshold for alerts.
    collapse_threshold: f32,
    /// Total samples processed.
    total_samples: usize,
}

impl DiversityMonitor {
    /// Create a new diversity monitor.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of recent scores to track
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            scores: VecDeque::with_capacity(window_size.max(1)),
            window_size: window_size.max(1),
            collapse_threshold: 0.1,
            total_samples: 0,
        }
    }

    /// Set the collapse detection threshold.
    #[must_use]
    pub fn with_collapse_threshold(mut self, threshold: f32) -> Self {
        self.collapse_threshold = threshold.max(0.0);
        self
    }

    /// Record a new diversity score.
    pub fn record(&mut self, score: DiversityScore) {
        if self.scores.len() >= self.window_size {
            self.scores.pop_front();
        }
        self.scores.push_back(score);
        self.total_samples += 1;
    }

    /// Get the number of samples in the window.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.scores.len()
    }

    /// Get total samples processed.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get the most recent diversity score.
    #[must_use]
    pub fn latest(&self) -> Option<DiversityScore> {
        self.scores.back().copied()
    }

    /// Get mean diversity over the window.
    #[must_use]
    pub fn mean_diversity(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.scores.iter().map(|s| s.mean_distance).sum();
        sum / self.scores.len() as f32
    }

    /// Get minimum diversity in the window.
    #[must_use]
    pub fn min_diversity(&self) -> f32 {
        self.scores
            .iter()
            .map(|s| s.mean_distance)
            .fold(f32::INFINITY, f32::min)
    }

    /// Get maximum diversity in the window.
    #[must_use]
    pub fn max_diversity(&self) -> f32 {
        self.scores
            .iter()
            .map(|s| s.mean_distance)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Check if current diversity indicates collapse.
    #[must_use]
    pub fn is_collapsing(&self) -> bool {
        match self.latest() {
            Some(score) => score.indicates_collapse(self.collapse_threshold),
            None => false,
        }
    }

    /// Check if diversity is trending downward.
    ///
    /// Compares first half of window to second half.
    #[must_use]
    pub fn is_trending_down(&self) -> bool {
        if self.scores.len() < 4 {
            return false;
        }

        let mid = self.scores.len() / 2;
        let first_half: f32 = self.scores.iter().take(mid).map(|s| s.mean_distance).sum();
        let second_half: f32 = self.scores.iter().skip(mid).map(|s| s.mean_distance).sum();

        let first_avg = first_half / mid as f32;
        let second_avg = second_half / (self.scores.len() - mid) as f32;

        second_avg < first_avg * 0.9 // 10% decline threshold
    }

    /// Get diversity variance in the window.
    #[must_use]
    pub fn variance(&self) -> f32 {
        if self.scores.len() < 2 {
            return 0.0;
        }

        let mean = self.mean_diversity();
        let sum_sq: f32 = self
            .scores
            .iter()
            .map(|s| (s.mean_distance - mean).powi(2))
            .sum();

        sum_sq / (self.scores.len() - 1) as f32
    }

    /// Clear all recorded scores.
    pub fn reset(&mut self) {
        self.scores.clear();
        self.total_samples = 0;
    }

    /// Compute diversity score from pairwise distances.
    ///
    /// # Arguments
    ///
    /// * `distances` - Pairwise distances between samples
    ///
    /// # Returns
    ///
    /// Computed `DiversityScore` from the distances.
    #[must_use]
    pub fn compute_from_distances(distances: &[f32]) -> DiversityScore {
        if distances.is_empty() {
            return DiversityScore::zero();
        }

        let sum: f32 = distances.iter().sum();
        let mean_distance = sum / distances.len() as f32;

        let min_distance = distances.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Estimate coverage as normalized mean distance
        let max_dist = distances.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let coverage = if max_dist > 0.0 {
            mean_distance / max_dist
        } else {
            0.0
        };

        DiversityScore::new(mean_distance, min_distance, coverage)
    }
}

impl Default for DiversityMonitor {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diversity_score_new() {
        let score = DiversityScore::new(0.5, 0.2, 0.8);
        assert!((score.mean_distance - 0.5).abs() < f32::EPSILON);
        assert!((score.min_distance - 0.2).abs() < f32::EPSILON);
        assert!((score.coverage - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diversity_score_zero() {
        let score = DiversityScore::zero();
        assert!((score.mean_distance - 0.0).abs() < f32::EPSILON);
        assert!((score.min_distance - 0.0).abs() < f32::EPSILON);
        assert!((score.coverage - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diversity_score_default() {
        let score = DiversityScore::default();
        assert_eq!(score, DiversityScore::zero());
    }

    #[test]
    fn test_indicates_collapse() {
        let score = DiversityScore::new(0.05, 0.01, 0.1);
        assert!(score.indicates_collapse(0.1));
        assert!(!score.indicates_collapse(0.01));
    }

    #[test]
    fn test_combined_score() {
        let score = DiversityScore::new(0.3, 0.3, 0.3);
        assert!((score.combined() - 0.3).abs() < f32::EPSILON);

        let score = DiversityScore::new(0.6, 0.3, 0.0);
        assert!((score.combined() - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_new() {
        let monitor = DiversityMonitor::new(5);
        assert_eq!(monitor.sample_count(), 0);
        assert_eq!(monitor.total_samples(), 0);
    }

    #[test]
    fn test_monitor_window_size_minimum() {
        let monitor = DiversityMonitor::new(0);
        assert_eq!(monitor.window_size, 1);
    }

    #[test]
    fn test_monitor_record() {
        let mut monitor = DiversityMonitor::new(3);

        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        assert_eq!(monitor.sample_count(), 1);

        monitor.record(DiversityScore::new(0.4, 0.2, 0.6));
        assert_eq!(monitor.sample_count(), 2);
    }

    #[test]
    fn test_monitor_window_overflow() {
        let mut monitor = DiversityMonitor::new(2);

        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        monitor.record(DiversityScore::new(0.4, 0.2, 0.6));
        monitor.record(DiversityScore::new(0.3, 0.1, 0.5));

        assert_eq!(monitor.sample_count(), 2);
        assert_eq!(monitor.total_samples(), 3);
    }

    #[test]
    fn test_monitor_latest() {
        let mut monitor = DiversityMonitor::new(5);
        assert!(monitor.latest().is_none());

        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        let latest = monitor.latest().expect("should have latest score");
        assert!((latest.mean_distance - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_mean_diversity() {
        let mut monitor = DiversityMonitor::new(5);
        assert!((monitor.mean_diversity() - 0.0).abs() < f32::EPSILON);

        monitor.record(DiversityScore::new(0.4, 0.2, 0.6));
        monitor.record(DiversityScore::new(0.6, 0.3, 0.8));

        assert!((monitor.mean_diversity() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_min_max_diversity() {
        let mut monitor = DiversityMonitor::new(5);

        monitor.record(DiversityScore::new(0.3, 0.1, 0.5));
        monitor.record(DiversityScore::new(0.5, 0.2, 0.7));
        monitor.record(DiversityScore::new(0.7, 0.3, 0.9));

        assert!((monitor.min_diversity() - 0.3).abs() < f32::EPSILON);
        assert!((monitor.max_diversity() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_is_collapsing() {
        let mut monitor = DiversityMonitor::new(5).with_collapse_threshold(0.2);

        assert!(!monitor.is_collapsing()); // No data

        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        assert!(!monitor.is_collapsing());

        monitor.record(DiversityScore::new(0.1, 0.05, 0.2));
        assert!(monitor.is_collapsing());
    }

    #[test]
    fn test_monitor_is_trending_down() {
        let mut monitor = DiversityMonitor::new(10);

        // Not enough data
        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        monitor.record(DiversityScore::new(0.4, 0.2, 0.6));
        assert!(!monitor.is_trending_down());

        // Add more data with downward trend
        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        monitor.record(DiversityScore::new(0.2, 0.1, 0.3));
        monitor.record(DiversityScore::new(0.2, 0.1, 0.3));

        assert!(monitor.is_trending_down());
    }

    #[test]
    fn test_monitor_variance() {
        let mut monitor = DiversityMonitor::new(5);

        // Not enough data
        assert!((monitor.variance() - 0.0).abs() < f32::EPSILON);

        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        assert!((monitor.variance() - 0.0).abs() < f32::EPSILON);

        // Same values = zero variance
        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        assert!((monitor.variance() - 0.0).abs() < f32::EPSILON);

        // Different values = non-zero variance
        monitor.record(DiversityScore::new(0.3, 0.1, 0.5));
        assert!(monitor.variance() > 0.0);
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = DiversityMonitor::new(5);
        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));
        monitor.record(DiversityScore::new(0.4, 0.2, 0.6));

        monitor.reset();

        assert_eq!(monitor.sample_count(), 0);
        assert_eq!(monitor.total_samples(), 0);
    }

    #[test]
    fn test_compute_from_distances_empty() {
        let score = DiversityMonitor::compute_from_distances(&[]);
        assert_eq!(score, DiversityScore::zero());
    }

    #[test]
    fn test_compute_from_distances() {
        let distances = vec![0.2, 0.4, 0.6, 0.8];
        let score = DiversityMonitor::compute_from_distances(&distances);

        assert!((score.mean_distance - 0.5).abs() < f32::EPSILON);
        assert!((score.min_distance - 0.2).abs() < f32::EPSILON);
        assert!(score.coverage > 0.0);
    }

    #[test]
    fn test_compute_from_distances_uniform() {
        let distances = vec![0.5, 0.5, 0.5];
        let score = DiversityMonitor::compute_from_distances(&distances);

        assert!((score.mean_distance - 0.5).abs() < f32::EPSILON);
        assert!((score.min_distance - 0.5).abs() < f32::EPSILON);
        assert!((score.coverage - 1.0).abs() < f32::EPSILON); // mean == max
    }

    #[test]
    fn test_monitor_default() {
        let monitor = DiversityMonitor::default();
        assert_eq!(monitor.window_size, 10);
    }

    #[test]
    fn test_monitor_clone() {
        let mut monitor = DiversityMonitor::new(5);
        monitor.record(DiversityScore::new(0.5, 0.3, 0.7));

        let cloned = monitor.clone();
        assert_eq!(cloned.sample_count(), 1);
    }

    #[test]
    fn test_diversity_score_copy() {
        let s1 = DiversityScore::new(0.5, 0.3, 0.7);
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
    }
}
