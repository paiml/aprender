//! Learning rate schedulers
//!
//! Provides learning rate scheduling strategies for training:
//! - `CosineAnnealingLR` - Smooth cosine decay
//! - `LinearWarmupLR` - Linear warmup from 0 to target
//! - `StepDecayLR` - Step decay by factor every N epochs
//! - `WarmupCosineDecayLR` - Combined warmup + cosine decay
//! - `WarmupLinearDecayLR` - Combined warmup + LINEAR decay to zero. This is the
//!   HuggingFace / `sentence-transformers` reference schedule
//!   (`get_linear_schedule_with_warmup`), and therefore SetFit's. `LinearWarmupLR` holds
//!   the peak constant after warmup and is NOT a substitute for it.

mod cosine_annealing;
mod linear_warmup;
mod step_decay;
mod warmup_cosine_decay;
mod warmup_linear_decay;

#[cfg(test)]
mod tests;

pub use cosine_annealing::CosineAnnealingLR;
pub use linear_warmup::LinearWarmupLR;
pub use step_decay::StepDecayLR;
pub use warmup_cosine_decay::WarmupCosineDecayLR;
pub use warmup_linear_decay::{warmup_steps_from_ratio, WarmupLinearDecayLR};

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get the current learning rate
    fn get_lr(&self) -> f32;

    /// Step the scheduler (typically called after each epoch or batch)
    fn step(&mut self);
}
