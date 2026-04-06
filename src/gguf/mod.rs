//! GGUF (GPT-Generated Unified Format) parser
//!
//! Pure Rust implementation of GGUF binary format reader.
//! Used by llama.cpp, Ollama, and compatible tools.
//!
//! Format specification: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! ## Module Structure
//!
//! This module is being incrementally refactored from a 54K-line monolith
//! into focused submodules for better testability and coverage.

// GGUF Module Structure
//
// Incremental shatter of src/gguf.rs (54K lines) into domain modules.
// Each module should be ≤800 lines for testability.
//
// Shatter Plan (19 modules from 54K lines):
// 🚧 types.rs: Additional tests for constants (~50 lines)
// - header.rs: GGUFHeader, TensorInfo
// - model.rs: GGUFModel, MappedGGUFModel
// - config.rs: GGUFConfig
// - transformer.rs: GGUFTransformer, GGUFTransformerLayer
// - quantized.rs: Quantized tensor types
// - owned.rs: OwnedQuantized* types
// - cached.rs: Cached model variants
// - batching.rs: Batch processing
// - scheduling.rs: Request scheduling
// - gpu_buffer.rs: GPU buffer management
// - prefix_cache.rs: Prefix caching
// - kv_cache.rs: KV cache types
// - inference.rs: OwnedQuantizedModel inference impl
// - cuda.rs: CUDA-specific code
//
// Migration Strategy: Include monolith, gradually extract, re-export all

// Modular structure
mod batch_scheduler;
mod config;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_model;
mod inference;
mod inference_types;
mod io;
pub(crate) mod keys;
mod loader;
mod model;
mod owned;
#[cfg(feature = "cuda")]
pub mod parity;
mod quantized;
mod runtime;
mod transformer;
mod types;
pub(crate) mod utils;

// Pure math operations (shared between CPU and GPU paths)
// UCBD §4: pub for re-export of rms_norm at crate root
pub mod ops;

// Test helpers module - shared utilities for GGUF tests
#[cfg(test)]
pub(crate) mod test_helpers;

// Test factory module - synthesize valid GGUF files in memory
#[cfg(test)]
pub(crate) mod test_factory;

// Rosetta format factory - synthesize all model formats (GGUF, SafeTensors, APR)
#[cfg(test)]
pub(crate) mod format_factory;

// Re-export types from organized modules
pub use batch_scheduler::*;
pub use config::*;
#[cfg(feature = "cuda")]
pub use cuda::{BatchedDecodeState, CudaBackend, CudaInitError};
#[cfg(feature = "cuda")]
pub use cuda_model::*;
pub use model::*;
pub use quantized::*;
pub use runtime::*;
pub mod logprobs;
pub use logprobs::*;
pub use transformer::*;
pub use types::*;

// Re-export inference types
pub use inference_types::*;

// Re-export cached model types from inference module
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use inference::{
    DequantizedFFNWeights, DequantizedWeightCache, OwnedQuantizedModelCached,
    OwnedQuantizedModelCachedSync,
};

// Tests module - shattered from monolith into focused part files
#[cfg(test)]
mod format_factory_tests;
#[cfg(test)]
mod inference_types_tests;
#[cfg(test)]
mod io_tests;
#[cfg(test)]
mod quantized_tests;
#[cfg(test)]
mod tests;
