//! Learning rate schedulers
//!
//! Provides learning rate scheduling strategies for training:
//! - `CosineAnnealingLR` - Smooth cosine decay
//! - `LinearWarmupLR` - Linear warmup from 0 to target
//! - `StepDecayLR` - Step decay by factor every N epochs
//! - `WarmupCosineDecayLR` - Combined warmup + cosine decay

mod cosine_annealing;
mod linear_warmup;
mod step_decay;
mod warmup_cosine_decay;

#[cfg(test)]
mod tests;

pub use cosine_annealing::CosineAnnealingLR;
pub use linear_warmup::LinearWarmupLR;
pub use step_decay::StepDecayLR;
pub use warmup_cosine_decay::WarmupCosineDecayLR;

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get the current learning rate
    fn get_lr(&self) -> f32;

    /// Step the scheduler (typically called after each epoch or batch)
    fn step(&mut self);
}
