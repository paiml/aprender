//! Configuration for synthetic data generation.

use super::andon::AndonConfig;

/// Configuration for synthetic data generation.
///
/// Controls the ratio of synthetic to original data, quality thresholds,
/// and diversity weighting for sample selection.
///
/// # Example
///
/// ```
/// use aprender::synthetic::SyntheticConfig;
///
/// let config = SyntheticConfig::default()
///     .with_augmentation_ratio(1.0)  // 1x synthetic data
///     .with_quality_threshold(0.8)   // 80% minimum quality
///     .with_diversity_weight(0.3);   // 30% weight on diversity
///
/// assert_eq!(config.augmentation_ratio, 1.0);
/// assert_eq!(config.quality_threshold, 0.8);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticConfig {
    /// Ratio of synthetic to original data (0.0 = none, 2.0 = 2x original).
    pub augmentation_ratio: f32,

    /// Minimum quality threshold for accepting generated samples [0.0, 1.0].
    pub quality_threshold: f32,

    /// Weight given to diversity vs quality in sample selection [0.0, 1.0].
    pub diversity_weight: f32,

    /// Maximum generation attempts per sample before giving up.
    pub max_attempts: usize,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Andon configuration for quality monitoring (Toyota Jidoka).
    pub andon: AndonConfig,
}

impl Default for SyntheticConfig {
    /// Creates a default configuration with conservative settings.
    ///
    /// - `augmentation_ratio`: 0.5 (50% synthetic data)
    /// - `quality_threshold`: 0.7 (70% minimum quality)
    /// - `diversity_weight`: 0.3 (30% diversity weight)
    /// - `max_attempts`: 10 attempts per sample
    /// - `seed`: 42 for reproducibility
    /// - `andon`: Enabled with 90% rejection threshold (Toyota Jidoka)
    fn default() -> Self {
        Self {
            augmentation_ratio: 0.5,
            quality_threshold: 0.7,
            diversity_weight: 0.3,
            max_attempts: 10,
            seed: 42,
            andon: AndonConfig::default(),
        }
    }
}

impl SyntheticConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the augmentation ratio.
    ///
    /// # Arguments
    ///
    /// * `ratio` - Ratio of synthetic to original data (clamped to [0.0, 10.0])
    #[must_use]
    pub fn with_augmentation_ratio(mut self, ratio: f32) -> Self {
        self.augmentation_ratio = ratio.clamp(0.0, 10.0);
        self
    }

    /// Set the quality threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum quality score for acceptance (clamped to [0.0, 1.0])
    #[must_use]
    pub fn with_quality_threshold(mut self, threshold: f32) -> Self {
        self.quality_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the diversity weight.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight for diversity vs quality (clamped to [0.0, 1.0])
    #[must_use]
    pub fn with_diversity_weight(mut self, weight: f32) -> Self {
        self.diversity_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum generation attempts.
    ///
    /// # Arguments
    ///
    /// * `attempts` - Maximum attempts per sample (minimum 1)
    #[must_use]
    pub fn with_max_attempts(mut self, attempts: usize) -> Self {
        self.max_attempts = attempts.max(1);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the Andon configuration for quality monitoring.
    #[must_use]
    pub fn with_andon(mut self, andon: AndonConfig) -> Self {
        self.andon = andon;
        self
    }

    /// Enable or disable Andon monitoring.
    #[must_use]
    pub fn with_andon_enabled(mut self, enabled: bool) -> Self {
        self.andon.enabled = enabled;
        self
    }

    /// Set the Andon rejection threshold.
    #[must_use]
    pub fn with_andon_rejection_threshold(mut self, threshold: f32) -> Self {
        self.andon.rejection_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Calculate target synthetic sample count from seed count.
    ///
    /// # Arguments
    ///
    /// * `seed_count` - Number of original seed samples
    ///
    /// # Returns
    ///
    /// Target number of synthetic samples to generate.
    #[must_use]
    pub fn target_count(&self, seed_count: usize) -> usize {
        (seed_count as f32 * self.augmentation_ratio) as usize
    }

    /// Check if a quality score meets the threshold.
    #[must_use]
    pub fn meets_quality(&self, score: f32) -> bool {
        score >= self.quality_threshold
    }

    /// Calculate combined score from quality and diversity.
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality score [0.0, 1.0]
    /// * `diversity` - Diversity score [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// Weighted combination: `(1 - diversity_weight) * quality + diversity_weight * diversity`
    #[must_use]
    pub fn combined_score(&self, quality: f32, diversity: f32) -> f32 {
        let quality_weight = 1.0 - self.diversity_weight;
        quality_weight * quality + self.diversity_weight * diversity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SyntheticConfig::default();

        assert!((config.augmentation_ratio - 0.5).abs() < f32::EPSILON);
        assert!((config.quality_threshold - 0.7).abs() < f32::EPSILON);
        assert!((config.diversity_weight - 0.3).abs() < f32::EPSILON);
        assert_eq!(config.max_attempts, 10);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_new_equals_default() {
        let config1 = SyntheticConfig::new();
        let config2 = SyntheticConfig::default();
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SyntheticConfig::new()
            .with_augmentation_ratio(1.5)
            .with_quality_threshold(0.8)
            .with_diversity_weight(0.4)
            .with_max_attempts(20)
            .with_seed(123);

        assert!((config.augmentation_ratio - 1.5).abs() < f32::EPSILON);
        assert!((config.quality_threshold - 0.8).abs() < f32::EPSILON);
        assert!((config.diversity_weight - 0.4).abs() < f32::EPSILON);
        assert_eq!(config.max_attempts, 20);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_augmentation_ratio_clamping() {
        let config = SyntheticConfig::new().with_augmentation_ratio(-1.0);
        assert!((config.augmentation_ratio - 0.0).abs() < f32::EPSILON);

        let config = SyntheticConfig::new().with_augmentation_ratio(15.0);
        assert!((config.augmentation_ratio - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_threshold_clamping() {
        let config = SyntheticConfig::new().with_quality_threshold(-0.5);
        assert!((config.quality_threshold - 0.0).abs() < f32::EPSILON);

        let config = SyntheticConfig::new().with_quality_threshold(1.5);
        assert!((config.quality_threshold - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diversity_weight_clamping() {
        let config = SyntheticConfig::new().with_diversity_weight(-0.2);
        assert!((config.diversity_weight - 0.0).abs() < f32::EPSILON);

        let config = SyntheticConfig::new().with_diversity_weight(1.2);
        assert!((config.diversity_weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_max_attempts_minimum() {
        let config = SyntheticConfig::new().with_max_attempts(0);
        assert_eq!(config.max_attempts, 1);
    }

    #[test]
    fn test_target_count() {
        let config = SyntheticConfig::new().with_augmentation_ratio(0.5);
        assert_eq!(config.target_count(100), 50);
        assert_eq!(config.target_count(0), 0);

        let config = SyntheticConfig::new().with_augmentation_ratio(2.0);
        assert_eq!(config.target_count(100), 200);
    }

    #[test]
    fn test_meets_quality() {
        let config = SyntheticConfig::new().with_quality_threshold(0.7);

        assert!(config.meets_quality(0.7));
        assert!(config.meets_quality(0.9));
        assert!(!config.meets_quality(0.69));
        assert!(!config.meets_quality(0.0));
    }

    #[test]
    fn test_combined_score() {
        let config = SyntheticConfig::new().with_diversity_weight(0.3);

        // quality=1.0, diversity=0.0 → 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        let score = config.combined_score(1.0, 0.0);
        assert!((score - 0.7).abs() < f32::EPSILON);

        // quality=0.0, diversity=1.0 → 0.7 * 0.0 + 0.3 * 1.0 = 0.3
        let score = config.combined_score(0.0, 1.0);
        assert!((score - 0.3).abs() < f32::EPSILON);

        // quality=0.5, diversity=0.5 → 0.7 * 0.5 + 0.3 * 0.5 = 0.5
        let score = config.combined_score(0.5, 0.5);
        assert!((score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_combined_score_extreme_weights() {
        // All quality, no diversity
        let config = SyntheticConfig::new().with_diversity_weight(0.0);
        let score = config.combined_score(0.8, 0.2);
        assert!((score - 0.8).abs() < f32::EPSILON);

        // All diversity, no quality
        let config = SyntheticConfig::new().with_diversity_weight(1.0);
        let score = config.combined_score(0.8, 0.2);
        assert!((score - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_clone() {
        let config1 = SyntheticConfig::new()
            .with_augmentation_ratio(1.0)
            .with_seed(999);

        let config2 = config1.clone();
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_config_debug() {
        let config = SyntheticConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("SyntheticConfig"));
        assert!(debug.contains("augmentation_ratio"));
    }

    // ============================================================================
    // EXTREME TDD: Andon Integration Tests
    // ============================================================================

    #[test]
    fn test_default_andon_config() {
        let config = SyntheticConfig::default();
        assert!(config.andon.enabled);
        assert!((config.andon.rejection_threshold - 0.90).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_andon() {
        use crate::synthetic::AndonConfig;

        let andon = AndonConfig::new()
            .with_enabled(false)
            .with_rejection_threshold(0.85);

        let config = SyntheticConfig::new().with_andon(andon);

        assert!(!config.andon.enabled);
        assert!((config.andon.rejection_threshold - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_andon_enabled() {
        let config = SyntheticConfig::new().with_andon_enabled(false);
        assert!(!config.andon.enabled);

        let config = SyntheticConfig::new().with_andon_enabled(true);
        assert!(config.andon.enabled);
    }

    #[test]
    fn test_with_andon_rejection_threshold() {
        let config = SyntheticConfig::new().with_andon_rejection_threshold(0.80);
        assert!((config.andon.rejection_threshold - 0.80).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_andon_rejection_threshold_clamping() {
        let config = SyntheticConfig::new().with_andon_rejection_threshold(1.5);
        assert!((config.andon.rejection_threshold - 1.0).abs() < f32::EPSILON);

        let config = SyntheticConfig::new().with_andon_rejection_threshold(-0.5);
        assert!((config.andon.rejection_threshold - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_andon_config_in_clone() {
        let config1 = SyntheticConfig::new()
            .with_andon_enabled(false)
            .with_andon_rejection_threshold(0.75);

        let config2 = config1.clone();
        assert!(!config2.andon.enabled);
        assert!((config2.andon.rejection_threshold - 0.75).abs() < f32::EPSILON);
    }
}
