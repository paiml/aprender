//! GPU Model Adapters (PMAT-106)
//!
//! Adapters for converting different model formats to GpuModel.
//!
//! # Supported Formats
//!
//! - **APR F32** - Native `.apr` format with F32 weights
//! - **APR Q4** - GGUF models with Q4_0 quantization
//! - **SafeTensors** - HuggingFace SafeTensors format (planned)
//!
//! # Coverage Impact
//!
//! These adapters drive coverage for:
//! - `apr_transformer/mod.rs` (F32)
//! - `apr_transformer/q4_simd.rs` (Q4)
//! - `gpu/scheduler/batch.rs`
//! - `api/openai_handlers.rs`

mod apr;
#[cfg(feature = "cuda")]
mod apr_q4;
#[cfg(all(test, feature = "cuda"))]
mod apr_q4_tests;
#[cfg(feature = "cuda")]
pub mod apr_q4k;
#[cfg(test)]
mod tests;
/// PMAT-333: WGPU adapter — dequantize quantized model for WGPU inference
pub mod wgpu_adapter;

pub use apr::{transpose_matrix, AprF32ToGpuAdapter, AprGpuError, AprToGpuAdapter};
#[cfg(feature = "cuda")]
pub use apr_q4::{AprQ4ToGpuAdapter, GpuModelQ4, LayerNorms};
#[cfg(feature = "cuda")]
pub use apr_q4k::{
    forward_token_apr_q4k, parse_apr_q4k_config, upload_apr_q4k_weights, AprQ4KConfig,
};
