//! Mixed-precision training utilities
//!
//! Provides support for training with reduced precision (fp16/bf16) while
//! maintaining numerical stability through loss scaling and master weights.
//!
//! ## Overview
//!
//! Mixed-precision training uses lower precision (fp16/bf16) for:
//! - Forward pass activations (memory savings)
//! - Gradient computation (compute speedup)
//!
//! While maintaining full precision (fp32) for:
//! - Master weights (numerical stability)
//! - Loss scaling (gradient underflow prevention)
//!
//! ## Example
//!
//! ```ignore
//! use entrenar::autograd::precision::{MixedPrecisionConfig, Precision, GradScaler};
//!
//! let config = MixedPrecisionConfig::bf16();
//! let mut scaler = GradScaler::new(config.initial_scale);
//!
//! // Forward pass in reduced precision
//! let loss = model.forward(&input);
//!
//! // Scale loss before backward
//! let scaled_loss = scaler.scale(loss);
//! backward(&mut scaled_loss, None);
//!
//! // Unscale and update
//! scaler.unscale_grads(&mut params);
//! optimizer.step(&mut params);
//! scaler.update();
//! ```

mod config;
mod conversions;
mod precision_types;
mod scaler;

#[cfg(test)]
mod tests;

// Re-export all public types and functions
pub use config::MixedPrecisionConfig;
pub use conversions::{
    bf16_to_f32, bf16_truncate, estimate_memory_savings, f32_to_bf16, f32_to_fp16, fp16_to_f32,
    gemm_bf16_reference,
};
pub use precision_types::Precision;
pub use scaler::GradScaler;
