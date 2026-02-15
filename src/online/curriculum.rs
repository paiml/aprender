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

include!("curriculum_part_02.rs");
include!("curriculum_part_03.rs");
