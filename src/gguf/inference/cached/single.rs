//! Single-threaded cached model wrapper (RefCell-based)
//!
//! `OwnedQuantizedModelCached` uses RefCell for interior mutability,
//! suitable for single-threaded inference without HTTP serving.

#[cfg(feature = "gpu")]
use provable_contracts_macros::contract;

use crate::error::{RealizarError, Result};
use crate::gguf::{
    OwnedQKVWeights, OwnedQuantizedModel, OwnedQuantizedTensor, QuantizedGenerateConfig,
};

/// Single-threaded cached model wrapper with RefCell-based scheduler caching
///
/// Uses `RefCell` for interior mutability to cache GPU schedulers. Not safe
/// for multi-threaded HTTP serving - use `OwnedQuantizedModelCachedSync` instead.
#[cfg(feature = "gpu")]
pub struct OwnedQuantizedModelCached {
    /// Inner model (not cached)
    model: OwnedQuantizedModel,
    /// Cached HybridScheduler for GPU operations (wgpu backend)
    /// Uses RefCell for interior mutability since scheduler requires &mut self
    scheduler: std::cell::RefCell<Option<crate::gpu::HybridScheduler>>,
    /// PARITY-103: Cached CudaScheduler for direct CUDA operations
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly
    #[cfg(feature = "cuda")]
    cuda_scheduler: std::cell::RefCell<Option<crate::gpu::CudaScheduler>>,
}

include!("single_true_batched_owned.rs");
