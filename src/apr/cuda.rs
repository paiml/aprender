//! CUDA-accelerated APR model inference (PMAT-802)
//!
//! Extracted from apr/mod.rs - GPU acceleration for APR v2 models.
//!
//! ## Contents
//! - `AprV2ModelCuda` - CUDA wrapper for APR models (2x Ollama target)

use super::{
    apply_rope_norm, dtype_to_ggml_qtype, rms_norm, simple_attention, transpose_matrix, AprV2Model,
};
use crate::error::{RealizarError, Result};

// ============================================================================
// AprV2ModelCuda: GPU-accelerated APR inference (2x Ollama target)
// ============================================================================

/// CUDA-accelerated wrapper for APR v2 models.
///
/// Mirrors `OwnedQuantizedModelCuda` from GGUF to provide GPU acceleration
/// for APR format models. Achieves 2x+ Ollama performance on supported GPUs.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::apr::{AprV2Model, AprV2ModelCuda};
///
/// let model = AprV2Model::load("model.apr")?;
/// let mut cuda_model = AprV2ModelCuda::new(model, 0)?; // GPU 0
///
/// // GPU-accelerated forward pass
/// let logits = cuda_model.forward_cuda(&[1, 2, 3])?;
///
/// // GPU-accelerated generation
/// let tokens = cuda_model.generate_cuda(&[1, 2, 3], 32, 151643)?;
/// ```
#[cfg(feature = "cuda")]
pub struct AprV2ModelCuda {
    /// Inner APR model
    model: AprV2Model,
    /// Cached CUDA executor
    executor: crate::cuda::CudaExecutor,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
    /// Cached weight buffers on GPU (tensor_name -> gpu_ptr)
    weight_cache: std::collections::HashMap<String, u64>,
    /// Cached embedding table (F32 for fast lookup)
    embedding_cache: Option<Vec<f32>>,
    /// Hidden dimension (cached for embedding lookup)
    hidden_dim: usize,
    /// Current KV cache position (increments with each decoded token)
    kv_position: u32,
    /// PMAT-110: Track if KV cache was populated via FALLBACK PATH
    /// When true, decode must also use FALLBACK PATH for consistency
    fallback_kv_used: bool,
    /// Phase 45: Test executor for dependency injection
    ///
    /// When present, this executor is used instead of CudaExecutor for GEMM operations.
    /// Enables testing forward pass logic without actual CUDA hardware.
    test_executor: Option<Box<dyn crate::gpu::executor::GpuExecutorTrait + Send + Sync>>,
    /// GH-201: Streaming mode (true = layer-by-layer, false = full cache)
    ///
    /// In streaming mode, only one layer's weights are on GPU at a time.
    /// This reduces VRAM usage from ~6GB to ~1.5GB for 1.5B models.
    streaming_mode: bool,
    /// GH-201: Currently cached layer index in streaming mode
    ///
    /// When streaming, this tracks which layer's weights are currently on GPU.
    /// None means no layer weights are cached yet.
    cached_streaming_layer: Option<usize>,
}

include!("cache.rs");
