//! Linear curriculum scheduler

use super::CurriculumScheduler;

/// Linear curriculum that increases difficulty over epochs
///
/// Difficulty increases linearly from `start_difficulty` to `end_difficulty`
/// over `ramp_epochs` epochs.
///
/// # Example
///
/// ```
/// use entrenar::train::{LinearCurriculum, CurriculumScheduler};
///
/// let mut curriculum = LinearCurriculum::new(0.3, 1.0, 10);
///
/// // Initially at start difficulty
/// assert!((curriculum.difficulty() - 0.3).abs() < 1e-5);
///
/// // After 5 epochs at 100% accuracy
/// for _ in 0..5 {
///     curriculum.step(0, 1.0);
/// }
/// assert!(curriculum.difficulty() > 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct LinearCurriculum {
    start_difficulty: f32,
    end_difficulty: f32,
    ramp_epochs: usize,
    current_epoch: usize,
}

impl LinearCurriculum {
    /// Create a new linear curriculum
    ///
    /// # Arguments
    ///
    /// * `start_difficulty` - Initial difficulty (0.0-1.0)
    /// * `end_difficulty` - Final difficulty (0.0-1.0)
    /// * `ramp_epochs` - Epochs to reach full difficulty
    pub fn new(start_difficulty: f32, end_difficulty: f32, ramp_epochs: usize) -> Self {
        Self {
            start_difficulty: start_difficulty.clamp(0.0, 1.0),
            end_difficulty: end_difficulty.clamp(0.0, 1.0),
            ramp_epochs: ramp_epochs.max(1),
            current_epoch: 0,
        }
    }
}

impl CurriculumScheduler for LinearCurriculum {
    fn difficulty(&self) -> f32 {
        let progress = (self.current_epoch as f32 / self.ramp_epochs as f32).min(1.0);
        let difficulty =
            self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty);
        let (min, max) = if self.start_difficulty <= self.end_difficulty {
            (self.start_difficulty, self.end_difficulty)
        } else {
            (self.end_difficulty, self.start_difficulty)
        };
        difficulty.clamp(min, max)
    }

    fn tier(&self) -> usize {
        // Map difficulty to 4 tiers (1-4)
        let d = self.difficulty();
        if d < 0.25 {
            1
        } else if d < 0.5 {
            2
        } else if d < 0.75 {
            3
        } else {
            4
        }
    }

    fn step(&mut self, _epoch: usize, _accuracy: f32) {
        self.current_epoch += 1;
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
    }

    fn name(&self) -> &'static str {
        "LinearCurriculum"
    }
}
