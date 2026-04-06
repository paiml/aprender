//! End-to-end knowledge distillation CLI.
//!
//! This crate provides a complete pipeline for knowledge distillation:
//! - Fetch teacher models from HuggingFace
//! - Configure distillation parameters via YAML
//! - Train student models with progressive/attention distillation
//! - Export to SafeTensors, GGUF, or APR formats
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Pre-flight validation catches errors before expensive training
//! - **Heijunka**: Memory estimation enables level scheduling of GPU resources
//! - **Kaizen**: Configurable hyperparameters enable continuous improvement

pub mod config;
pub mod pipeline;
pub mod validation;
pub mod weights;

pub use config::DistillConfig;
pub use pipeline::{Pipeline, PipelineResult};
pub use validation::ConfigValidator;
pub use weights::load_safetensors_weights;

use entrenar_common::Result;

/// Run the distillation pipeline with the given configuration.
pub fn run(config: &DistillConfig) -> Result<PipelineResult> {
    // Validate configuration first (Jidoka)
    ConfigValidator::validate(config)?;

    // Execute pipeline
    Pipeline::new(config).execute()
}

/// Estimate memory requirements without running training.
pub fn estimate_memory(config: &DistillConfig) -> Result<MemoryEstimate> {
    ConfigValidator::validate(config)?;
    Pipeline::estimate_memory(config)
}

/// Memory estimation result.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Model weights memory in bytes
    pub model_bytes: u64,
    /// Activation memory in bytes
    pub activation_bytes: u64,
    /// Optimizer state memory in bytes
    pub optimizer_bytes: u64,
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Whether this fits in available VRAM
    pub fits_in_vram: bool,
    /// Recommended batch size for available memory
    pub recommended_batch_size: usize,
}

impl MemoryEstimate {
    /// Create a new memory estimate.
    pub fn new(model_params: u64, batch_size: usize, seq_len: usize, hidden_dim: usize) -> Self {
        // Model weights (assume FP16 for training)
        let model_bytes = model_params * 2;

        // Activations: batch * seq * hidden * layers * 2 (forward + backward)
        let activation_bytes = (batch_size * seq_len * hidden_dim * 32 * 2) as u64;

        // Optimizer state: 2x model size for Adam (momentum + variance)
        let optimizer_bytes = model_bytes * 2;

        let total_bytes = model_bytes + activation_bytes + optimizer_bytes;

        // Assume 24GB VRAM as target
        let available_vram = 24 * 1024 * 1024 * 1024u64;
        let fits_in_vram = total_bytes < available_vram;

        // Calculate recommended batch size to fit in 80% of available VRAM
        let target_memory = (available_vram as f64 * 0.8) as u64;
        let per_sample = (seq_len * hidden_dim * 32 * 2) as u64;
        let available_for_activations = target_memory.saturating_sub(model_bytes + optimizer_bytes);
        let recommended_batch_size = if per_sample > 0 {
            (available_for_activations / per_sample).max(1) as usize
        } else {
            1
        };

        Self {
            model_bytes,
            activation_bytes,
            optimizer_bytes,
            total_bytes,
            fits_in_vram,
            recommended_batch_size,
        }
    }

    /// Format as human-readable string.
    pub fn to_human_readable(&self) -> String {
        format!(
            "Memory Estimate:\n  Model: {:.1} GB\n  Activations: {:.1} GB\n  Optimizer: {:.1} GB\n  Total: {:.1} GB\n  Fits in 24GB VRAM: {}",
            self.model_bytes as f64 / 1e9,
            self.activation_bytes as f64 / 1e9,
            self.optimizer_bytes as f64 / 1e9,
            self.total_bytes as f64 / 1e9,
            if self.fits_in_vram { "Yes" } else { "No" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimate_calculation() {
        // 7B parameter model
        let estimate = MemoryEstimate::new(7_000_000_000, 32, 512, 4096);

        // Model weights should be ~14GB for FP16
        assert!(estimate.model_bytes > 10_000_000_000);
        assert!(estimate.model_bytes < 20_000_000_000);

        // Total should include model + activations + optimizer
        assert!(estimate.total_bytes > estimate.model_bytes);
    }

    #[test]
    fn test_memory_estimate_fits_calculation() {
        // Small model should fit
        let small = MemoryEstimate::new(100_000_000, 8, 256, 768);
        assert!(small.fits_in_vram);

        // Huge model shouldn't fit
        let huge = MemoryEstimate::new(70_000_000_000, 32, 2048, 8192);
        assert!(!huge.fits_in_vram);
    }

    #[test]
    fn test_recommended_batch_size_positive() {
        let estimate = MemoryEstimate::new(7_000_000_000, 32, 512, 4096);
        assert!(estimate.recommended_batch_size >= 1);
    }
}
