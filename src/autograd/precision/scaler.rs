//! Gradient scaler for mixed-precision training.

use super::MixedPrecisionConfig;

/// Default number of successful steps before the loss scale is increased
const DEFAULT_SCALE_GROWTH_INTERVAL: usize = 2000;

/// Gradient scaler for mixed-precision training
///
/// Handles loss scaling to prevent gradient underflow in fp16 training.
#[derive(Debug)]
pub struct GradScaler {
    /// Current loss scale
    scale: f32,
    /// Growth factor
    growth_factor: f32,
    /// Backoff factor
    backoff_factor: f32,
    /// Growth interval
    pub(crate) growth_interval: usize,
    /// Steps since last growth
    steps_since_growth: usize,
    /// Whether dynamic scaling is enabled
    dynamic: bool,
    /// Number of overflows encountered
    overflow_count: usize,
    /// Number of successful steps
    successful_steps: usize,
}

impl GradScaler {
    /// Create a new gradient scaler
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: DEFAULT_SCALE_GROWTH_INTERVAL,
            steps_since_growth: 0,
            dynamic: true,
            overflow_count: 0,
            successful_steps: 0,
        }
    }

    /// Create from config
    pub fn from_config(config: &MixedPrecisionConfig) -> Self {
        Self {
            scale: config.initial_scale,
            growth_factor: config.scale_growth_factor,
            backoff_factor: config.scale_backoff_factor,
            growth_interval: config.scale_growth_interval,
            steps_since_growth: 0,
            dynamic: config.dynamic_scaling,
            overflow_count: 0,
            successful_steps: 0,
        }
    }

    /// Get current scale
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Scale a loss value
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Unscale a gradient value
    pub fn unscale_grad(&self, grad: f32) -> f32 {
        grad / self.scale
    }

    /// Unscale gradients in place and check for overflow
    ///
    /// Returns true if gradients are valid (no overflow), false otherwise.
    pub fn unscale_and_check(&self, grads: &mut [f32]) -> bool {
        let inv_scale = 1.0 / self.scale;
        let mut has_overflow = false;

        for grad in grads.iter_mut() {
            *grad *= inv_scale;
            if !grad.is_finite() {
                has_overflow = true;
            }
        }

        !has_overflow
    }

    /// Update the scale after a step
    ///
    /// Call this after each optimizer step. Pass `true` if gradients were valid.
    pub fn update(&mut self, grads_valid: bool) {
        contract_pre_update!();
        if !self.dynamic {
            return;
        }

        if grads_valid {
            self.successful_steps += 1;
            self.steps_since_growth += 1;

            // Grow scale after interval of successful steps
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        } else {
            // Overflow detected - reduce scale
            self.overflow_count += 1;
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;

            // Ensure scale doesn't go too low
            self.scale = self.scale.max(1.0);
        }
    }

    /// Get overflow count
    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    /// Get successful step count
    pub fn successful_steps(&self) -> usize {
        self.successful_steps
    }

    /// Check if dynamic scaling is enabled
    pub fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    /// Enable/disable dynamic scaling
    pub fn set_dynamic(&mut self, enabled: bool) {
        self.dynamic = enabled;
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new(65536.0)
    }
}
