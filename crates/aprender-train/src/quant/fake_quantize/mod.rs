//! Fake Quantization for Quantization-Aware Training (QAT)
//!
//! Fake quantization simulates the effects of quantization during training:
//! - Forward: quantize → dequantize (simulates quantization noise)
//! - Backward: Straight-Through Estimator (STE) passes gradients unchanged
//!
//! This allows models to adapt to quantization noise during training,
//! resulting in better accuracy after actual quantization.

mod config;
mod ops;
mod quantize;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use config::FakeQuantConfig;
pub use ops::{fake_quantize, ste_backward};
pub use quantize::FakeQuantize;
