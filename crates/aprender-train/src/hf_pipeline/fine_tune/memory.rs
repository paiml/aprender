//! Memory estimation and requirements for fine-tuning
//!
//! Provides utilities for estimating memory usage during training.

/// Mixed precision training options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedPrecision {
    /// FP16 mixed precision
    Fp16,
    /// BF16 mixed precision (better for training)
    Bf16,
}

/// Memory requirements breakdown
#[derive(Debug, Clone, Copy)]
pub struct MemoryRequirement {
    /// Model weights
    pub model: u64,
    /// Optimizer states
    pub optimizer: u64,
    /// Gradients
    pub gradients: u64,
    /// Activations
    pub activations: u64,
}

impl MemoryRequirement {
    /// Total memory required
    #[must_use]
    pub fn total(&self) -> u64 {
        self.model + self.optimizer + self.gradients + self.activations
    }

    /// Check if fits in available memory
    #[must_use]
    pub fn fits_in(&self, available: u64) -> bool {
        self.total() <= available
    }

    /// Memory savings compared to full fine-tuning
    #[must_use]
    pub fn savings_vs_full(&self, full_params: u64) -> f32 {
        let full_memory = full_params * 4 + full_params * 4 * 2 + full_params * 4;
        1.0 - (self.total() as f32 / full_memory as f32)
    }

    /// Format as human-readable string
    #[must_use]
    pub fn format_human(&self) -> String {
        format!(
            "Model: {:.1}GB, Optimizer: {:.1}GB, Gradients: {:.1}GB, Activations: {:.1}GB, Total: {:.1}GB",
            self.model as f64 / 1e9,
            self.optimizer as f64 / 1e9,
            self.gradients as f64 / 1e9,
            self.activations as f64 / 1e9,
            self.total() as f64 / 1e9
        )
    }
}
