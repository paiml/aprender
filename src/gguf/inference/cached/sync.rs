//! Thread-safe cached model wrapper (Mutex-based)
//!
//! `OwnedQuantizedModelCachedSync` uses Mutex for interior mutability,
//! suitable for async HTTP servers and multi-threaded inference.

use super::weights::{DequantizedFFNWeights, DequantizedWeightCache};
use crate::error::{RealizarError, Result};
use crate::gguf::{
    BatchGenerationStats, DispatchMetrics, OwnedQuantizedKVCache, OwnedQuantizedModel,
    QuantizedGenerateConfig,
};

/// Thread-safe cached model wrapper with Mutex-based scheduler caching
///
/// Uses `Mutex` for interior mutability to cache GPU schedulers. Safe for
/// multi-threaded HTTP serving with async handlers.
pub struct OwnedQuantizedModelCachedSync {
    /// Inner model (not cached)
    model: OwnedQuantizedModel,
    /// Cached HybridScheduler for GPU operations (wgpu backend)
    /// Uses Mutex for thread-safe interior mutability
    scheduler: std::sync::Mutex<Option<crate::gpu::HybridScheduler>>,
    /// PARITY-103: Cached CudaScheduler for direct CUDA operations
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly
    #[cfg(feature = "cuda")]
    cuda_scheduler: std::sync::Mutex<Option<crate::gpu::CudaScheduler>>,
    /// Dequantized weight cache for GPU batch inference (PARITY-019)
    /// Uses RwLock for concurrent read access during batch inference
    dequant_cache: std::sync::RwLock<Option<DequantizedWeightCache>>,
}

// Explicitly implement Send + Sync for HTTP server usage
#[cfg(feature = "gpu")]
unsafe impl Send for OwnedQuantizedModelCachedSync {}
#[cfg(feature = "gpu")]
unsafe impl Sync for OwnedQuantizedModelCachedSync {}

include!("sync_owned_quantized.rs");
