//! Configuration for mixed-precision training.

use super::Precision;

/// Default number of successful steps before the loss scale is increased
const DEFAULT_SCALE_GROWTH_INTERVAL: usize = 2000;
/// Initial loss scale for fp16 mixed-precision (2^16)
const FP16_INITIAL_LOSS_SCALE: f32 = 65536.0;

/// Configuration for mixed-precision training
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision for activations and gradients
    pub compute_precision: Precision,
    /// Precision for master weights (always fp32 recommended)
    pub weight_precision: Precision,
    /// Initial loss scale factor
    pub initial_scale: f32,
    /// Factor to increase scale by on successful step
    pub scale_growth_factor: f32,
    /// Factor to decrease scale by on overflow
    pub scale_backoff_factor: f32,
    /// Number of successful steps before increasing scale
    pub scale_growth_interval: usize,
    /// Whether to use dynamic loss scaling
    pub dynamic_scaling: bool,
}

impl MixedPrecisionConfig {
    /// Create fp32 config (no mixed precision)
    pub fn fp32() -> Self {
        Self {
            compute_precision: Precision::Fp32,
            weight_precision: Precision::Fp32,
            initial_scale: 1.0,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: DEFAULT_SCALE_GROWTH_INTERVAL,
            dynamic_scaling: false,
        }
    }

    /// Create fp16 mixed-precision config
    pub fn fp16() -> Self {
        Self {
            compute_precision: Precision::Fp16,
            weight_precision: Precision::Fp32,
            initial_scale: FP16_INITIAL_LOSS_SCALE,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: DEFAULT_SCALE_GROWTH_INTERVAL,
            dynamic_scaling: true,
        }
    }

    /// Create bf16 mixed-precision config
    pub fn bf16() -> Self {
        Self {
            compute_precision: Precision::Bf16,
            weight_precision: Precision::Fp32,
            initial_scale: 1.0, // bf16 has larger dynamic range, less scaling needed
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: DEFAULT_SCALE_GROWTH_INTERVAL,
            dynamic_scaling: false, // Often not needed for bf16
        }
    }

    /// Check if mixed precision is enabled
    pub fn is_mixed(&self) -> bool {
        self.compute_precision.is_reduced()
    }

    /// Set initial loss scale
    pub fn with_initial_scale(mut self, scale: f32) -> Self {
        self.initial_scale = scale;
        self
    }

    /// Enable/disable dynamic scaling
    pub fn with_dynamic_scaling(mut self, enabled: bool) -> Self {
        self.dynamic_scaling = enabled;
        self
    }
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self::fp32()
    }
}
