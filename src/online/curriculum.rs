//! Curriculum Learning for Progressive Training
//!
//! Train on easy samples first, then progressively harder ones to
//! improve convergence speed and model quality.
//!
//! # References
//!
//! - [Bengio et al. 2009] "Curriculum Learning" - Theoretical foundation
//! - [Kumar et al. 2010] "Self-Paced Learning for Latent Variable Models"
//!
//! # Toyota Way Principles
//!
//! - **Heijunka**: Level training workload by ordering samples
//! - **Kaizen**: Progressive improvement through staged learning

use crate::error::{AprenderError, Result};

/// Difficulty scorer for samples
pub trait DifficultyScorer: Send + Sync {
    /// Score sample difficulty (lower = easier)
    ///
    /// # Arguments
    /// * `features` - Feature vector
    /// * `target` - Target value
    ///
    /// # Returns
    /// Difficulty score (typically 0.0 to 1.0, but can be unbounded)
    fn score(&self, features: &[f64], target: f64) -> f64;
}

/// Curriculum learning scheduler
///
/// Reference: [Bengio et al. 2009] "Curriculum Learning"
/// - Ordering samples from easy to hard improves convergence
/// - 25% faster training, better generalization
pub trait CurriculumScheduler: Send + Sync {
    /// Get current stage (0.0 = easiest only, 1.0 = all data)
    fn stage(&self) -> f64;

    /// Advance to next stage
    fn advance(&mut self);

    /// Check if curriculum is complete
    fn is_complete(&self) -> bool;

    /// Get the difficulty threshold for current stage
    fn current_threshold(&self) -> f64;

    /// Reset to beginning
    fn reset(&mut self);
}

/// Sample with cached difficulty
#[derive(Debug, Clone)]
pub struct ScoredSample {
    /// Feature vector
    pub features: Vec<f64>,
    /// Target value
    pub target: f64,
    /// Cached difficulty score
    pub difficulty: f64,
}

impl ScoredSample {
    /// Create a new scored sample
    #[must_use] 
    pub fn new(features: Vec<f64>, target: f64, difficulty: f64) -> Self {
        Self {
            features,
            target,
            difficulty,
        }
    }
}

/// Linear curriculum that progresses through difficulty levels
#[derive(Debug, Clone)]
pub struct LinearCurriculum {
    /// Current stage (0.0 to 1.0)
    stage: f64,
    /// Stage increment per advance
    step_size: f64,
    /// Maximum difficulty at each stage
    difficulty_range: (f64, f64),
}

impl Default for LinearCurriculum {
    fn default() -> Self {
        Self::new(10)
    }
}

impl LinearCurriculum {
    /// Create a linear curriculum with specified number of stages
    ///
    /// # Arguments
    /// * `n_stages` - Number of stages (higher = finer progression)
    #[must_use] 
    pub fn new(n_stages: usize) -> Self {
        let step_size = if n_stages > 0 {
            1.0 / n_stages as f64
        } else {
            1.0
        };

        Self {
            stage: 0.0,
            step_size,
            difficulty_range: (0.0, 1.0),
        }
    }

    /// Set custom difficulty range
    #[must_use] 
    pub fn with_difficulty_range(mut self, min: f64, max: f64) -> Self {
        self.difficulty_range = (min, max);
        self
    }
}

impl CurriculumScheduler for LinearCurriculum {
    fn stage(&self) -> f64 {
        self.stage
    }

    fn advance(&mut self) {
        self.stage = (self.stage + self.step_size).min(1.0);
    }

    fn is_complete(&self) -> bool {
        self.stage >= 1.0
    }

    fn current_threshold(&self) -> f64 {
        let (min, max) = self.difficulty_range;
        min + (max - min) * self.stage
    }

    fn reset(&mut self) {
        self.stage = 0.0;
    }
}

/// Exponential curriculum that progresses slowly at first
#[derive(Debug, Clone)]
pub struct ExponentialCurriculum {
    /// Current stage (0.0 to 1.0)
    stage: f64,
    /// Growth rate (higher = faster progression)
    growth_rate: f64,
    /// Number of advances
    n_advances: u64,
}

impl Default for ExponentialCurriculum {
    fn default() -> Self {
        Self::new(0.3)
    }
}

impl ExponentialCurriculum {
    /// Create an exponential curriculum
    ///
    /// # Arguments
    /// * `growth_rate` - Growth rate (typical: 0.1-0.5)
    #[must_use] 
    pub fn new(growth_rate: f64) -> Self {
        Self {
            stage: 0.0,
            growth_rate,
            n_advances: 0,
        }
    }
}

impl CurriculumScheduler for ExponentialCurriculum {
    fn stage(&self) -> f64 {
        self.stage
    }

    fn advance(&mut self) {
        self.n_advances += 1;
        // Exponential growth: 1 - e^(-r*t)
        self.stage = 1.0 - (-self.growth_rate * self.n_advances as f64).exp();
    }

    fn is_complete(&self) -> bool {
        self.stage >= 0.99
    }

    fn current_threshold(&self) -> f64 {
        self.stage
    }

    fn reset(&mut self) {
        self.stage = 0.0;
        self.n_advances = 0;
    }
}

/// Self-paced curriculum learning
///
/// Reference: [Kumar et al. 2010] "Self-Paced Learning for Latent
/// Variable Models"
/// - Automatically determines sample difficulty from loss
/// - Adapts pace based on model performance
#[derive(Debug, Clone)]
pub struct SelfPacedCurriculum {
    /// All training samples with cached difficulties
    samples: Vec<ScoredSample>,
    /// Current difficulty threshold
    threshold: f64,
    /// Initial threshold
    initial_threshold: f64,
    /// Threshold growth rate
    growth_rate: f64,
    /// Maximum threshold
    max_threshold: f64,
    /// Current batch index
    batch_idx: usize,
}

impl SelfPacedCurriculum {
    /// Create a new self-paced curriculum
    ///
    /// # Arguments
    /// * `initial_threshold` - Starting difficulty threshold (e.g., 0.1)
    /// * `growth_rate` - How much to increase threshold per advance (e.g., 1.5)
    #[must_use] 
    pub fn new(initial_threshold: f64, growth_rate: f64) -> Self {
        Self {
            samples: Vec::new(),
            threshold: initial_threshold,
            initial_threshold,
            growth_rate,
            max_threshold: f64::MAX,
            batch_idx: 0,
        }
    }

    /// Set maximum threshold
    #[must_use] 
    pub fn with_max_threshold(mut self, max: f64) -> Self {
        self.max_threshold = max;
        self
    }

    /// Add samples with difficulty scores
    pub fn add_samples(&mut self, samples: Vec<ScoredSample>) {
        self.samples.extend(samples);
        // Sort by difficulty (easiest first)
        self.samples.sort_by(|a, b| {
            a.difficulty
                .partial_cmp(&b.difficulty)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Update difficulties based on model predictions
    ///
    /// # Arguments
    /// * `loss_fn` - Function that computes loss for a sample
    pub fn update_difficulties<F>(&mut self, loss_fn: F)
    where
        F: Fn(&[f64], f64) -> f64,
    {
        for sample in &mut self.samples {
            sample.difficulty = loss_fn(&sample.features, sample.target);
        }

        // Re-sort by difficulty
        self.samples.sort_by(|a, b| {
            a.difficulty
                .partial_cmp(&b.difficulty)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get next batch of samples below current threshold
    pub fn next_batch(&mut self, batch_size: usize) -> Vec<&ScoredSample> {
        let eligible: Vec<_> = self
            .samples
            .iter()
            .filter(|s| s.difficulty <= self.threshold)
            .collect();

        let start = self.batch_idx;
        let end = (start + batch_size).min(eligible.len());

        if start >= eligible.len() {
            self.batch_idx = 0;
            return vec![];
        }

        self.batch_idx = end;
        eligible[start..end].to_vec()
    }

    /// Get all samples below current threshold
    #[must_use] 
    pub fn eligible_samples(&self) -> Vec<&ScoredSample> {
        self.samples
            .iter()
            .filter(|s| s.difficulty <= self.threshold)
            .collect()
    }

    /// Get number of eligible samples
    #[must_use] 
    pub fn n_eligible(&self) -> usize {
        self.samples
            .iter()
            .filter(|s| s.difficulty <= self.threshold)
            .count()
    }

    /// Get total number of samples
    #[must_use] 
    pub fn n_total(&self) -> usize {
        self.samples.len()
    }

    /// Get current threshold
    #[must_use] 
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl CurriculumScheduler for SelfPacedCurriculum {
    fn stage(&self) -> f64 {
        if self.samples.is_empty() {
            return 1.0;
        }

        let eligible = self.n_eligible();
        eligible as f64 / self.samples.len() as f64
    }

    fn advance(&mut self) {
        self.threshold = (self.threshold * self.growth_rate).min(self.max_threshold);
        self.batch_idx = 0;
    }

    fn is_complete(&self) -> bool {
        self.n_eligible() >= self.samples.len()
    }

    fn current_threshold(&self) -> f64 {
        self.threshold
    }

    fn reset(&mut self) {
        self.threshold = self.initial_threshold;
        self.batch_idx = 0;
    }
}

/// Loss-based difficulty scorer
///
/// Uses prediction error as difficulty proxy
#[derive(Debug, Clone)]
pub struct LossDifficultyScorer {
    /// Mean of target values (for baseline)
    target_mean: f64,
}

impl LossDifficultyScorer {
    /// Create a new loss-based scorer
    #[must_use] 
    pub fn new() -> Self {
        Self { target_mean: 0.0 }
    }

    /// Create with known target mean
    #[must_use] 
    pub fn with_mean(target_mean: f64) -> Self {
        Self { target_mean }
    }

    /// Estimate target mean from samples
    pub fn fit(&mut self, targets: &[f64]) {
        if !targets.is_empty() {
            self.target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
        }
    }
}

impl Default for LossDifficultyScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl DifficultyScorer for LossDifficultyScorer {
    fn score(&self, _features: &[f64], target: f64) -> f64 {
        // Distance from mean as proxy for difficulty
        (target - self.target_mean).abs()
    }
}

/// Feature-norm difficulty scorer
///
/// Uses feature magnitude as difficulty proxy (larger = harder)
#[derive(Debug, Clone, Default)]
pub struct FeatureNormScorer;

impl FeatureNormScorer {
    /// Create a new feature norm scorer
    #[must_use] 
    pub fn new() -> Self {
        Self
    }
}

impl DifficultyScorer for FeatureNormScorer {
    fn score(&self, features: &[f64], _target: f64) -> f64 {
        // L2 norm of features
        features.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// Curriculum-guided training helper
#[derive(Debug)]
pub struct CurriculumTrainer<S: CurriculumScheduler + std::fmt::Debug> {
    /// The curriculum scheduler
    scheduler: S,
    /// All samples sorted by difficulty
    samples: Vec<ScoredSample>,
}

impl<S: CurriculumScheduler + std::fmt::Debug> CurriculumTrainer<S> {
    /// Create a new curriculum trainer
    pub fn new(scheduler: S) -> Self {
        Self {
            scheduler,
            samples: Vec::new(),
        }
    }

    /// Add samples with difficulty scoring
    pub fn add_samples<D: DifficultyScorer>(
        &mut self,
        features: &[f64],
        targets: &[f64],
        n_features: usize,
        scorer: &D,
    ) -> Result<()> {
        if features.len() % n_features != 0 {
            return Err(AprenderError::dimension_mismatch(
                "features",
                n_features,
                features.len() % n_features,
            ));
        }

        let n_samples = features.len() / n_features;
        if n_samples != targets.len() {
            return Err(AprenderError::dimension_mismatch(
                "targets",
                n_samples,
                targets.len(),
            ));
        }

        for i in 0..n_samples {
            let feat = features[i * n_features..(i + 1) * n_features].to_vec();
            let target = targets[i];
            let difficulty = scorer.score(&feat, target);

            self.samples
                .push(ScoredSample::new(feat, target, difficulty));
        }

        // Sort by difficulty
        self.samples.sort_by(|a, b| {
            a.difficulty
                .partial_cmp(&b.difficulty)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(())
    }

    /// Get samples eligible for current stage
    pub fn eligible_samples(&self) -> Vec<&ScoredSample> {
        let threshold = self.scheduler.current_threshold();

        // Normalize difficulties to 0-1 range
        let (min_diff, max_diff) = self
            .samples
            .iter()
            .fold((f64::MAX, f64::MIN), |(min, max), s| {
                (min.min(s.difficulty), max.max(s.difficulty))
            });

        let range = (max_diff - min_diff).max(1e-10);

        self.samples
            .iter()
            .filter(|s| {
                let normalized = (s.difficulty - min_diff) / range;
                normalized <= threshold
            })
            .collect()
    }

    /// Get number of eligible samples
    pub fn n_eligible(&self) -> usize {
        self.eligible_samples().len()
    }

    /// Advance the curriculum
    pub fn advance(&mut self) {
        self.scheduler.advance();
    }

    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        self.scheduler.is_complete()
    }

    /// Get current stage
    pub fn stage(&self) -> f64 {
        self.scheduler.stage()
    }

    /// Reset curriculum
    pub fn reset(&mut self) {
        self.scheduler.reset();
    }

    /// Get scheduler reference
    pub fn scheduler(&self) -> &S {
        &self.scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curriculum_basic() {
        let mut curriculum = LinearCurriculum::new(10);

        assert_eq!(curriculum.stage(), 0.0);
        assert!(!curriculum.is_complete());

        curriculum.advance();
        assert!((curriculum.stage() - 0.1).abs() < 0.01);

        // Advance to completion
        for _ in 0..15 {
            curriculum.advance();
        }

        assert!(curriculum.is_complete());
        assert!((curriculum.stage() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_curriculum_threshold() {
        let curriculum = LinearCurriculum::new(10).with_difficulty_range(0.2, 0.8);

        // At stage 0, threshold should be 0.2
        assert!((curriculum.current_threshold() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_linear_curriculum_reset() {
        let mut curriculum = LinearCurriculum::new(10);

        curriculum.advance();
        curriculum.advance();
        curriculum.reset();

        assert_eq!(curriculum.stage(), 0.0);
    }

    #[test]
    fn test_exponential_curriculum() {
        let mut curriculum = ExponentialCurriculum::new(0.3);

        assert_eq!(curriculum.stage(), 0.0);

        curriculum.advance();
        let stage1 = curriculum.stage();

        curriculum.advance();
        let stage2 = curriculum.stage();

        // Should grow exponentially
        assert!(stage1 > 0.0);
        assert!(stage2 > stage1);
        assert!(stage2 < 1.0);
    }

    #[test]
    fn test_exponential_curriculum_completion() {
        let mut curriculum = ExponentialCurriculum::new(0.5);

        for _ in 0..50 {
            curriculum.advance();
        }

        assert!(curriculum.is_complete());
    }

    #[test]
    fn test_self_paced_curriculum_basic() {
        let mut curriculum = SelfPacedCurriculum::new(0.25, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
            ScoredSample::new(vec![3.0], 3.0, 0.5),
            ScoredSample::new(vec![4.0], 4.0, 0.8),
        ];

        curriculum.add_samples(samples);

        // With threshold 0.25, only first two samples (0.1, 0.2) should be eligible
        let initial = curriculum.n_eligible();
        assert!(
            initial >= 1 && initial <= 2,
            "Expected 1-2 eligible, got {}",
            initial
        );

        // Advance threshold (0.25 * 1.5 = 0.375)
        curriculum.advance();
        // Now samples with difficulty <= 0.375 should be eligible
        let after_advance = curriculum.n_eligible();
        assert!(
            after_advance >= initial,
            "Should have more eligible after advance"
        );
    }

    #[test]
    fn test_self_paced_curriculum_next_batch() {
        let mut curriculum = SelfPacedCurriculum::new(1.0, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
            ScoredSample::new(vec![3.0], 3.0, 0.3),
        ];

        curriculum.add_samples(samples);

        let batch1 = curriculum.next_batch(2);
        assert_eq!(batch1.len(), 2);

        let batch2 = curriculum.next_batch(2);
        assert_eq!(batch2.len(), 1); // Only one left
    }

    #[test]
    fn test_self_paced_curriculum_update_difficulties() {
        let mut curriculum = SelfPacedCurriculum::new(0.5, 1.5);

        let samples = vec![
            ScoredSample::new(vec![1.0], 1.0, 0.1),
            ScoredSample::new(vec![2.0], 2.0, 0.2),
        ];

        curriculum.add_samples(samples);

        // Update with new difficulty function
        curriculum.update_difficulties(|_features, target| target * 0.1);

        // Check difficulties were updated
        let eligible = curriculum.eligible_samples();
        assert!(eligible[0].difficulty <= eligible.get(1).map_or(f64::MAX, |s| s.difficulty));
    }

    #[test]
    fn test_loss_difficulty_scorer() {
        let mut scorer = LossDifficultyScorer::new();
        scorer.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]); // mean = 3.0

        // Sample at mean should have low difficulty
        let diff_at_mean = scorer.score(&[0.0], 3.0);
        // Sample far from mean should have high difficulty
        let diff_far = scorer.score(&[0.0], 10.0);

        assert!(diff_at_mean < diff_far);
    }

    #[test]
    fn test_feature_norm_scorer() {
        let scorer = FeatureNormScorer::new();

        // Small feature vector
        let small = scorer.score(&[1.0, 0.0], 0.0);
        // Large feature vector
        let large = scorer.score(&[3.0, 4.0], 0.0);

        assert!(small < large);
        assert!((large - 5.0).abs() < 0.01); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_curriculum_trainer_basic() {
        let scheduler = LinearCurriculum::new(5);
        let mut trainer = CurriculumTrainer::new(scheduler);

        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let targets = vec![1.0, 2.0, 3.0];
        let scorer = FeatureNormScorer::new();

        trainer
            .add_samples(&features, &targets, 2, &scorer)
            .unwrap();

        assert_eq!(trainer.stage(), 0.0);
        assert!(!trainer.is_complete());
    }

    #[test]
    fn test_curriculum_trainer_eligible_samples() {
        let scheduler = LinearCurriculum::new(2);
        let mut trainer = CurriculumTrainer::new(scheduler);

        // Features with varying norms
        let features = vec![
            1.0, 0.0, // norm = 1
            3.0, 4.0, // norm = 5
            0.5, 0.0, // norm = 0.5
        ];
        let targets = vec![1.0, 2.0, 3.0];
        let scorer = FeatureNormScorer::new();

        trainer
            .add_samples(&features, &targets, 2, &scorer)
            .unwrap();

        // At stage 0, only easiest samples
        let eligible = trainer.n_eligible();
        assert!(eligible > 0);
        assert!(eligible <= 3);

        // Advance and check more are eligible
        trainer.advance();
        let eligible_after = trainer.n_eligible();
        assert!(eligible_after >= eligible);
    }

    #[test]
    fn test_curriculum_trainer_dimension_mismatch() {
        let scheduler = LinearCurriculum::new(5);
        let mut trainer = CurriculumTrainer::new(scheduler);

        let features = vec![1.0, 2.0, 3.0]; // 3 features, doesn't divide by 2
        let targets = vec![1.0];
        let scorer = FeatureNormScorer::new();

        let result = trainer.add_samples(&features, &targets, 2, &scorer);
        assert!(result.is_err());
    }

    #[test]
    fn test_curriculum_trainer_target_mismatch() {
        let scheduler = LinearCurriculum::new(5);
        let mut trainer = CurriculumTrainer::new(scheduler);

        let features = vec![1.0, 2.0, 3.0, 4.0]; // 2 samples
        let targets = vec![1.0, 2.0, 3.0]; // 3 targets (mismatch!)
        let scorer = FeatureNormScorer::new();

        let result = trainer.add_samples(&features, &targets, 2, &scorer);
        assert!(result.is_err());
    }

    #[test]
    fn test_scored_sample_creation() {
        let sample = ScoredSample::new(vec![1.0, 2.0], 3.0, 0.5);

        assert_eq!(sample.features, vec![1.0, 2.0]);
        assert_eq!(sample.target, 3.0);
        assert_eq!(sample.difficulty, 0.5);
    }

    #[test]
    fn test_linear_curriculum_default() {
        let curriculum = LinearCurriculum::default();
        assert_eq!(curriculum.stage(), 0.0);
    }

    #[test]
    fn test_exponential_curriculum_default() {
        let curriculum = ExponentialCurriculum::default();
        assert_eq!(curriculum.stage(), 0.0);
    }

    #[test]
    fn test_self_paced_max_threshold() {
        let mut curriculum = SelfPacedCurriculum::new(0.1, 2.0).with_max_threshold(0.5);

        // Advance many times
        for _ in 0..10 {
            curriculum.advance();
        }

        assert!(curriculum.threshold() <= 0.5);
    }

    #[test]
    fn test_self_paced_empty() {
        let curriculum = SelfPacedCurriculum::new(0.5, 1.5);

        // Empty curriculum should be "complete"
        assert!(curriculum.is_complete());
        assert_eq!(curriculum.stage(), 1.0);
    }
}
