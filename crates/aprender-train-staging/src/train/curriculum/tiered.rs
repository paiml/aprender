//! Tiered curriculum scheduler for CITL

use super::CurriculumScheduler;

/// Tiered curriculum for diagnostic verbosity levels
///
/// Designed for CITL training with four diagnostic tiers:
/// - Tier 1: JSON diagnostics + clippy (baseline)
/// - Tier 2: + verbose build output
/// - Tier 3: + RUSTC_LOG traces
/// - Tier 4: + full debug output
///
/// Tier advancement based on accuracy thresholds.
///
/// # Example
///
/// ```
/// use entrenar::train::{TieredCurriculum, CurriculumScheduler};
///
/// let mut curriculum = TieredCurriculum::new(vec![0.6, 0.7, 0.8], 3);
///
/// assert_eq!(curriculum.tier(), 1);
///
/// // Advance to tier 2 after achieving 60% accuracy for 3 epochs
/// for _ in 0..3 {
///     curriculum.step(0, 0.65);
/// }
/// assert_eq!(curriculum.tier(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct TieredCurriculum {
    /// Accuracy thresholds to advance to next tier
    tier_thresholds: Vec<f32>,
    /// Consecutive epochs at threshold before advancing
    patience: usize,
    /// Current tier (1-4)
    current_tier: usize,
    /// Epochs at current tier meeting threshold
    epochs_at_threshold: usize,
}

impl TieredCurriculum {
    /// Create new tiered curriculum
    ///
    /// # Arguments
    ///
    /// * `tier_thresholds` - Accuracy thresholds for each tier advancement
    /// * `patience` - Epochs at threshold before advancing
    pub fn new(tier_thresholds: Vec<f32>, patience: usize) -> Self {
        Self { tier_thresholds, patience: patience.max(1), current_tier: 1, epochs_at_threshold: 0 }
    }

    /// Create with default CITL thresholds
    ///
    /// - Tier 1 -> 2: 60% accuracy
    /// - Tier 2 -> 3: 70% accuracy
    /// - Tier 3 -> 4: 80% accuracy
    pub fn citl_default() -> Self {
        Self::new(vec![0.6, 0.7, 0.8], 3)
    }

    /// Get the threshold for current tier advancement
    pub fn current_threshold(&self) -> Option<f32> {
        if self.current_tier <= self.tier_thresholds.len() {
            Some(self.tier_thresholds[self.current_tier - 1])
        } else {
            None
        }
    }
}

impl CurriculumScheduler for TieredCurriculum {
    fn difficulty(&self) -> f32 {
        (self.current_tier as f32 - 1.0) / 3.0
    }

    fn tier(&self) -> usize {
        self.current_tier
    }

    fn step(&mut self, _epoch: usize, accuracy: f32) {
        if let Some(threshold) = self.current_threshold() {
            if accuracy >= threshold {
                self.epochs_at_threshold += 1;
                if self.epochs_at_threshold >= self.patience {
                    // Advance to next tier
                    self.current_tier = (self.current_tier + 1).min(4);
                    self.epochs_at_threshold = 0;
                }
            } else {
                // Reset counter if below threshold
                self.epochs_at_threshold = 0;
            }
        }
    }

    fn reset(&mut self) {
        self.current_tier = 1;
        self.epochs_at_threshold = 0;
    }

    fn name(&self) -> &'static str {
        "TieredCurriculum"
    }
}
