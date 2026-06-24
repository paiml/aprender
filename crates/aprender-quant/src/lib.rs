#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::disallowed_methods,
        clippy::float_cmp,
        clippy::panic,
        clippy::cast_precision_loss
    )
)]
//! K-Quantization formats for GGUF/APR model weights (Toyota Way: ONE source of truth)
//!
//! This crate provides quantization functions for converting F32 data to
//! K-quantization formats (`Q4_K`, `Q5_K`, `Q6_K`). This is the ONLY implementation
//! in the Sovereign AI Stack - aprender and realizar import from here.
//!
//! ## Stack Architecture (Toyota Way)
//!
//! ```text
//!        ┌─────────┐
//!        │ apr CLI │
//!        └────┬────┘
//!             │
//!     ┌───────┼───────┬───────────┐
//!     ▼       ▼       ▼           ▼
//! ┌────────┐ ┌────────┐ ┌─────────┐
//! │entrenar│ │aprender│ │realizar │
//! └───┬────┘ └───┬────┘ └────┬────┘
//!     │          │           │
//!     └────┬─────┴───────────┴────┘
//!          ▼
//!       ┌────────────────┐
//!       │  trueno-quant  │  ← YOU ARE HERE
//!       └───────┬────────┘
//!               ▼
//!       ┌────────────────┐
//!       │     trueno     │
//!       └────────────────┘
//! ```
//!
//! ## Format Specifications
//!
//! - `Q4_K`: 256-element super-blocks, 144 bytes (4.5 bits/weight)
//! - `Q5_K`: 256-element super-blocks, 176 bytes (5.5 bits/weight)
//! - `Q6_K`: 256-element super-blocks, 210 bytes (6.5 bits/weight)
//!
//! ## Usage
//!
//! ```rust
//! use trueno_quant::{quantize_q4_k, dequantize_q4_k_to_f32};
//!
//! let data: Vec<f32> = (0..256).map(|i| i as f32 / 10.0).collect();
//! let quantized = quantize_q4_k(&data);
//! let restored = dequantize_q4_k_to_f32(&quantized, 256);
//! ```

#![warn(missing_docs)]

mod dequantize;
mod quantize;
mod transpose;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod roundtrip_fidelity_tests;

// Re-export all public functions so the public API doesn't change
pub use dequantize::{dequantize_q4_k_to_f32, dequantize_q5_k_to_f32, dequantize_q6_k_to_f32};
pub use quantize::{
    quantize_q4_k, quantize_q4_k_matrix, quantize_q5_k, quantize_q5_k_matrix, quantize_q6_k,
    quantize_q6_k_matrix,
};
pub use transpose::{transpose_q4k_for_matmul, transpose_q5k_for_matmul, transpose_q6k_for_matmul};

// ============================================================================
// Constants
// ============================================================================

/// Minimum valid f16 normal value (~6.1e-5)
/// Prevents NaN on round-trip through f16 encoding
pub const F16_MIN_NORMAL: f32 = 6.1e-5;

/// `Q4_K` super-block size (elements per block)
pub const Q4_K_BLOCK_SIZE: usize = 256;

/// `Q4_K` super-block byte size
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// `Q5_K` super-block size (elements per block)
pub const Q5_K_BLOCK_SIZE: usize = 256;

/// `Q5_K` super-block byte size
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// `Q6_K` super-block size (elements per block)
pub const Q6_K_BLOCK_SIZE: usize = 256;

/// `Q6_K` super-block byte size
pub const Q6_K_BLOCK_BYTES: usize = 210;

// ============================================================================
// f16 Conversion Helpers
// ============================================================================

/// Convert f32 to f16 (using half crate)
#[inline]
#[must_use]
pub fn f32_to_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// Convert f16 to f32 (using half crate)
#[inline]
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
