#[allow(clippy::wildcard_imports)]
use super::*;

impl TeacherForcing {
    /// Create constant teacher forcing (ratio doesn't change).
    #[must_use]
    pub fn constant(ratio: f32) -> Self {
        assert!((0.0..=1.0).contains(&ratio), "Ratio must be in [0, 1]");
        Self {
            schedule: TeacherForcingSchedule::Constant,
            initial_ratio: ratio,
            final_ratio: ratio,
            num_steps: 1,
        }
    }

    /// Create linear decay schedule.
    #[must_use]
    pub fn linear(initial: f32, final_ratio: f32, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&initial),
            "Initial ratio must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&final_ratio),
            "Final ratio must be in [0, 1]"
        );
        Self {
            schedule: TeacherForcingSchedule::Linear,
            initial_ratio: initial,
            final_ratio,
            num_steps,
        }
    }

    /// Create exponential decay schedule.
    #[must_use]
    pub fn exponential(initial: f32, final_ratio: f32, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&initial),
            "Initial ratio must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&final_ratio),
            "Final ratio must be in [0, 1]"
        );
        Self {
            schedule: TeacherForcingSchedule::Exponential,
            initial_ratio: initial,
            final_ratio,
            num_steps,
        }
    }

    /// Create inverse square root decay schedule.
    #[must_use]
    pub fn inverse_sqrt(initial: f32, num_steps: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&initial),
            "Initial ratio must be in [0, 1]"
        );
        Self {
            schedule: TeacherForcingSchedule::InverseSquareRoot,
            initial_ratio: initial,
            final_ratio: 0.0,
            num_steps,
        }
    }

    /// Get teacher forcing ratio for given step.
    #[must_use]
    pub fn get_ratio(&self, step: usize) -> f32 {
        match self.schedule {
            TeacherForcingSchedule::Constant => self.initial_ratio,

            TeacherForcingSchedule::Linear => {
                if step >= self.num_steps {
                    self.final_ratio
                } else {
                    let progress = step as f32 / self.num_steps as f32;
                    self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress
                }
            }

            TeacherForcingSchedule::Exponential => {
                if step >= self.num_steps {
                    self.final_ratio
                } else {
                    let decay = (self.final_ratio / self.initial_ratio.max(1e-10))
                        .powf(step as f32 / self.num_steps as f32);
                    self.initial_ratio * decay
                }
            }

            TeacherForcingSchedule::InverseSquareRoot => {
                self.initial_ratio / (1.0 + step as f32).sqrt()
            }
        }
    }

    /// Decide whether to use teacher forcing at this step.
    ///
    /// Returns true with probability equal to the current ratio.
    #[must_use]
    pub fn should_use_teacher(&self, step: usize) -> bool {
        let ratio = self.get_ratio(step);
        rand::random::<f32>() < ratio
    }

    #[must_use]
    pub fn schedule(&self) -> TeacherForcingSchedule {
        self.schedule
    }

    #[must_use]
    pub fn initial_ratio(&self) -> f32 {
        self.initial_ratio
    }

    #[must_use]
    pub fn final_ratio(&self) -> f32 {
        self.final_ratio
    }
}

#[cfg(test)]
mod tests;
