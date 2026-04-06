//! Curriculum scheduler trait

/// Trait for curriculum learning schedulers
///
/// Determines training difficulty/tier based on training progress.
pub trait CurriculumScheduler: Send {
    /// Get the current difficulty level (0.0 = easiest, 1.0 = hardest)
    fn difficulty(&self) -> f32;

    /// Get the current tier (for tiered training like CITL)
    fn tier(&self) -> usize;

    /// Advance the curriculum based on training progress
    fn step(&mut self, epoch: usize, accuracy: f32);

    /// Reset the curriculum to initial state
    fn reset(&mut self);

    /// Get sample weight for a given difficulty score
    ///
    /// Returns weight multiplier for loss (1.0 = normal weight)
    fn sample_weight(&self, sample_difficulty: f32) -> f32 {
        1.0 - (sample_difficulty - self.difficulty()).abs().min(1.0) * 0.5
    }

    /// Check if sample should be included at current difficulty
    fn include_sample(&self, sample_difficulty: f32) -> bool {
        sample_difficulty <= self.difficulty()
    }

    /// Name of the curriculum scheduler
    fn name(&self) -> &str;
}
