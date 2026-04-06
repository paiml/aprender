//! Scalar Int8 Quantization for Embedding Retrieval
//!
//! Implements the scalar int8 rescoring retriever specification with:
//! - 4x memory reduction (f32 -> i8)
//! - 99% accuracy retention with rescoring
//! - 3.66x speedup via SIMD acceleration
//!
//! # References
//!
//! - Jacob et al. (2018) - Quantization and Training of Neural Networks
//! - Gholami et al. (2022) - Survey of Quantization Methods
//! - Wu et al. (2020) - Integer Quantization Principles
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Auto-stop on quantization error > threshold
//! - **Poka-Yoke**: Type-safe precision levels, compile-time checks
//! - **Heijunka**: Batched rescoring with backpressure
//! - **Kaizen**: Continuous calibration improvement
//! - **Genchi Genbutsu**: Hardware-specific benchmarks
//! - **Muda**: 4x memory reduction via quantization

// Library code - usage from examples and integration tests
#![allow(dead_code, unused_imports)]

mod calibration;
mod embedding;
mod error;
mod params;
mod retriever;
mod simd;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use calibration::CalibrationStats;
pub use embedding::{compute_hash, QuantizedEmbedding};
pub use error::{validate_embedding, QuantizationError};
pub use params::QuantizationParams;
pub use retriever::{RescoreResult, RescoreRetriever, RescoreRetrieverConfig};
pub use simd::{dot_i8_scalar, SimdBackend};
