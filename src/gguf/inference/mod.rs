//! CPU inference implementation for OwnedQuantizedModel
//!
//! This module contains the forward pass and generation logic
//! extracted from the monolith for better testability.
//!
//! ## Submodules
//!
//! - `forward/`: Forward pass methods (forward, forward_cached, forward_batch)
//! - `attention.rs`: Attention computation (apply_rope, causal_attention)
//! - `matmul.rs`: Quantized matrix operations (fused_matmul, qkv_matmul)
//! - `generation.rs`: Token generation (generate, sample_topk)
//! - `cached.rs`: Cached model wrappers for GPU inference

mod attention;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub(crate) mod cached;
mod forward;
mod generation;
mod matmul;

#[cfg(test)]
mod attention_flash_tests;
#[cfg(test)]
mod attention_gqa_tests;
#[cfg(test)]
mod generation_tests;
#[cfg(test)]
mod matmul_tests;

// Re-export cached model types for external use
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use cached::{
    DequantizedFFNWeights, DequantizedWeightCache, OwnedQuantizedModelCached,
    OwnedQuantizedModelCachedSync,
};

// PMAT-395: Re-export encoder-decoder types
pub use forward::EncoderOutput;

// Re-export impl extension for OwnedQuantizedModel
// The actual impl blocks are in each submodule
