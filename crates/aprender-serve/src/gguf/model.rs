//! GGUF model types
//!
//! This module contains the core model structures for GGUF inference:
//!
//! - `MappedGGUFModel`: Memory-mapped GGUF file with zero-copy access
//! - `GGUFTransformer`: F32 transformer weights (dequantized from GGUF)
//! - `GGUFTransformerLayer`: Per-layer transformer weights
//! - `OwnedQuantizedModel`: Quantized model with owned weight data
//!
//! ## Design Philosophy
//!
//! Per Wulf & McKee (1995) "Hitting the Memory Wall", memory bandwidth is the
//! bottleneck for LLM inference. These types support:
//! - Zero-copy mmap loading (`MappedGGUFModel`)
//! - Quantized weights for 8x bandwidth reduction (`OwnedQuantizedModel`)
//! - Lazy dequantization during computation

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::config::GGUFConfig;
use super::quantized::{OwnedQuantizedLayer, OwnedQuantizedTensor};
use super::types::GGUFModel;
use crate::error::{RealizarError, Result};

// ============================================================================
// MappedGGUFModel - Zero-copy memory-mapped model
// ============================================================================

/// Memory-mapped GGUF model for zero-copy tensor access
///
/// Uses `memmap2` for efficient large model loading without copying
/// entire file contents into heap memory.
///
/// # Example
///
/// ```rust,ignore
/// let model = MappedGGUFModel::from_path("phi-2-q4_k_m.gguf")?;
/// println!("Loaded {} tensors", model.model.tensors.len());
/// ```
pub struct MappedGGUFModel {
    /// Parsed model metadata (header, tensors, etc.)
    pub model: GGUFModel,
    /// Memory-mapped file contents
    pub(crate) mmap: Mmap,
}

impl MappedGGUFModel {
    /// Load GGUF model via memory mapping (zero-copy)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to GGUF model file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File cannot be opened
    /// - Memory mapping fails
    /// - GGUF parsing fails (invalid format)
    ///
    /// # Performance
    ///
    /// Memory-mapped loading is faster than `std::fs::read` for large models:
    /// - No file content copy to heap memory
    /// - Kernel handles page management
    /// - Model remains accessible even if larger than RAM (via swap)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let model = MappedGGUFModel::from_path("phi-2-q4_k_m.gguf")?;
    /// println!("Loaded {} tensors", model.model.tensors.len());
    /// ```
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "open_model_file".to_string(),
            reason: format!("Failed to open {}: {}", path.as_ref().display(), e),
        })?;

        // PMAT-304: MAP_POPULATE + MADV_RANDOM (matching llama.cpp's llama-mmap.cpp)
        // MAP_POPULATE: synchronously pre-fault ALL pages at mmap time.
        //   Eliminates page faults during inference (was causing DRAM stalls).
        // MADV_RANDOM: disable sequential readahead (wastes BW on non-sequential access).
        // MADV_HUGEPAGE: reduce TLB misses for 900MB+ scans.
        // mlock: prevent swapping.
        //
        // Reference: llama.cpp llama-mmap.cpp line 424: flags |= MAP_POPULATE
        //            llama.cpp llama-mmap.cpp line 438: POSIX_MADV_RANDOM
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .populate()
                .map(&file)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "mmap_model_file".to_string(),
                    reason: format!("Failed to mmap {}: {}", path.as_ref().display(), e),
                })?
        };

        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_HUGEPAGE,
            );
            // MADV_RANDOM: disable readahead — weight access is strided, not sequential
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_RANDOM,
            );
            libc::mlock(mmap.as_ptr() as *const libc::c_void, mmap.len());
        }

        // Parse the memory-mapped data
        let model = GGUFModel::from_bytes(&mmap)?;

        Ok(Self { model, mmap })
    }

    /// Get the raw memory-mapped file data
    ///
    /// This provides direct access to the file contents without copying.
    /// Use this with tensor offsets to read quantized weights directly.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Get tensor data slice by offset and size
    ///
    /// Returns a slice pointing directly into the memory-mapped file.
    /// No data is copied.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset from start of file
    /// * `size` - Size in bytes
    ///
    /// # Returns
    ///
    /// Slice of tensor data, or None if out of bounds
    #[must_use]
    pub fn tensor_slice(&self, offset: usize, size: usize) -> Option<&[u8]> {
        let end = offset.checked_add(size)?;
        if end <= self.mmap.len() {
            Some(&self.mmap[offset..end])
        } else {
            None
        }
    }

    /// Get the size of the memory-mapped file
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Advise kernel to prefetch model data sequentially
    ///
    /// Per llama.cpp: Use madvise(MADV_SEQUENTIAL) to hint that the model
    /// will be read sequentially during loading. This improves prefetching.
    #[cfg(unix)]
    pub fn advise_sequential(&self) {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_SEQUENTIAL,
            );
        }
    }

    /// Advise kernel for random access pattern during inference
    ///
    /// Per llama.cpp: Use madvise(MADV_RANDOM) during inference when
    /// accessing weights non-sequentially.
    #[cfg(unix)]
    pub fn advise_random(&self) {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_RANDOM,
            );
        }
    }

    /// Advise kernel to keep model in memory (reduce swap pressure)
    ///
    /// Per llama.cpp: Use madvise(MADV_WILLNEED) to hint that the model
    /// will be needed soon, triggering prefetch.
    #[cfg(unix)]
    pub fn advise_willneed(&self) {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().cast_mut().cast::<libc::c_void>(),
                self.mmap.len(),
                libc::MADV_WILLNEED,
            );
        }
    }

    /// Lock model in memory to prevent swapping (requires privileges)
    ///
    /// Per llama.cpp: Use mlock() to ensure model stays in RAM.
    /// Returns true if successful, false if failed (often due to ulimit).
    #[cfg(unix)]
    pub fn lock_memory(&self) -> bool {
        // SAFETY: Memory safety ensured by mmap lifetime
        unsafe { libc::mlock(self.mmap.as_ptr().cast::<libc::c_void>(), self.mmap.len()) == 0 }
    }
}

// ============================================================================
// GGUFTransformer - F32 transformer weights
// ============================================================================

/// F32 transformer weights loaded from GGUF
///
/// This struct holds dequantized weights in F32 format.
/// Used for reference implementations and debugging.
/// For production inference, use `OwnedQuantizedModel` instead.
pub struct GGUFTransformer {
    /// Model configuration
    pub config: GGUFConfig,
    /// Token embedding weights [vocab_size, hidden_dim]
    pub token_embedding: Vec<f32>,
    /// GH-278: Position embedding weights [context_length, hidden_dim] (GPT-2 only)
    pub position_embedding: Option<Vec<f32>>,
    /// Attention weights per layer
    pub layers: Vec<GGUFTransformerLayer>,
    /// Output norm weight
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head / output projection weight
    pub lm_head_weight: Vec<f32>,
    /// LM head bias (optional)
    pub lm_head_bias: Option<Vec<f32>>,
}

// ============================================================================
// GGUFTransformerLayer - Per-layer F32 weights
// ============================================================================

/// Weights for a single transformer layer
pub struct GGUFTransformerLayer {
    /// Attention norm weight
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (combined for phi-2, concatenated Q+K+V for llama)
    pub qkv_weight: Vec<f32>,
    /// QKV bias (phi-2 has bias, llama doesn't)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weight
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate projection weight (SwiGLU models like llama)
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate projection bias
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection weight
    pub ffn_up_weight: Vec<f32>,
    /// FFN up projection bias
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weight
    pub ffn_down_weight: Vec<f32>,
    /// FFN down projection bias
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (for models with separate FFN normalization)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias
    pub ffn_norm_bias: Option<Vec<f32>>,
    /// GH-279: Per-head Q RMSNorm weight [head_dim] (Qwen3)
    pub attn_q_norm_weight: Option<Vec<f32>>,
    /// GH-279: Per-head K RMSNorm weight [head_dim] (Qwen3)
    pub attn_k_norm_weight: Option<Vec<f32>>,
}

// ============================================================================
// OwnedQuantizedModel - Quantized model with owned data
// ============================================================================

/// Owned quantized model with all weight data
///
/// IMP-100: This struct owns all quantized weight data, allowing storage
/// in `Arc` without lifetime parameters. Essential for async handlers.
///
/// # Memory Layout
///
/// - Token embedding: F32 for fast lookup
/// - Layer weights: Quantized (Q4_K, Q6_K, etc.)
/// - Output norm: F32 (small)
/// - LM head: Quantized
///
/// # GPU Acceleration
///
/// When the `cuda` feature is enabled, this struct includes a
/// `CudaExecutor` for GPU-accelerated matmul operations.
pub struct OwnedQuantizedModel {
    /// Model configuration
    pub(crate) config: GGUFConfig,
    /// Token embedding (f32 for fast lookup)
    pub(crate) token_embedding: Vec<f32>,
    /// GH-278: Position embedding [context_length, hidden_dim] (GPT-2 only)
    pub(crate) position_embedding: Option<Vec<f32>>,
    /// Owned quantized layers (decoder layers, or all layers for decoder-only)
    pub(crate) layers: Vec<OwnedQuantizedLayer>,
    /// GH-177: Encoder layers for encoder-decoder models (T5, Whisper).
    /// Empty for decoder-only architectures.
    pub(crate) encoder_layers: Vec<OwnedQuantizedLayer>,
    /// GH-177: Encoder output norm weight (T5/Whisper encoder final norm).
    /// None for decoder-only architectures.
    pub(crate) encoder_output_norm_weight: Option<Vec<f32>>,
    /// GH-177: Encoder output norm bias (optional).
    pub(crate) encoder_output_norm_bias: Option<Vec<f32>>,
    /// Output norm weight (f32)
    pub(crate) output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub(crate) output_norm_bias: Option<Vec<f32>>,
    /// LM head weight (owned quantized)
    pub(crate) lm_head_weight: OwnedQuantizedTensor,
    /// LM head bias (optional, f32)
    pub(crate) lm_head_bias: Option<Vec<f32>>,
    /// PARITY-113: Optional CUDA executor for GPU acceleration
    /// When present, fused_matmul routes to CUDA GEMM kernels
    /// Uses Mutex for thread-safety in async handlers
    #[cfg(feature = "cuda")]
    pub(crate) cuda_executor: Option<std::sync::Mutex<crate::cuda::CudaExecutor>>,
    /// Track CUDA kernel execution count for metrics
    /// Uses AtomicU64 for thread-safe counting
    #[cfg(feature = "cuda")]
    pub(crate) cuda_kernel_count: std::sync::atomic::AtomicU64,
    /// PARITY-003: Set of weight names that have been cached on GPU
    /// Used to avoid repeated dequantization for the same weight
    #[cfg(feature = "cuda")]
    pub(crate) cached_weight_names: std::sync::Mutex<std::collections::HashSet<String>>,
}

// Manual Debug implementation (skip CUDA executor which doesn't impl Debug)
impl std::fmt::Debug for OwnedQuantizedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("OwnedQuantizedModel");
        s.field("config", &self.config)
            .field("token_embedding_len", &self.token_embedding.len())
            .field("has_position_embedding", &self.position_embedding.is_some())
            .field("layers_count", &self.layers.len())
            .field("encoder_layers_count", &self.encoder_layers.len())
            .field(
                "has_encoder_output_norm_weight",
                &self.encoder_output_norm_weight.is_some(),
            )
            .field(
                "has_encoder_output_norm_bias",
                &self.encoder_output_norm_bias.is_some(),
            )
            .field("output_norm_weight_len", &self.output_norm_weight.len())
            .field("has_output_norm_bias", &self.output_norm_bias.is_some())
            .field("lm_head_weight", &self.lm_head_weight)
            .field("has_lm_head_bias", &self.lm_head_bias.is_some());

        #[cfg(feature = "cuda")]
        s.field("cuda_enabled", &self.cuda_executor.is_some())
            .field(
                "cuda_kernel_count",
                &self
                    .cuda_kernel_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "cached_weight_count",
                &self
                    .cached_weight_names
                    .lock()
                    .map(|g| g.len())
                    .unwrap_or(0),
            );

        s.finish()
    }
}

// Manual Clone implementation due to Mutex
impl Clone for OwnedQuantizedModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.clone(),
            position_embedding: self.position_embedding.clone(),
            layers: self.layers.clone(),
            encoder_layers: self.encoder_layers.clone(),
            encoder_output_norm_weight: self.encoder_output_norm_weight.clone(),
            encoder_output_norm_bias: self.encoder_output_norm_bias.clone(),
            output_norm_weight: self.output_norm_weight.clone(),
            output_norm_bias: self.output_norm_bias.clone(),
            lm_head_weight: self.lm_head_weight.clone(),
            lm_head_bias: self.lm_head_bias.clone(),
            // CUDA executor is not cloned - new instance must enable CUDA separately
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }
}

// =============================================================================
// PMAT-283: Public read-only accessors (fields sealed to pub(crate))
// =============================================================================

impl OwnedQuantizedModel {
    /// Model configuration (mutable access for overrides).
    pub fn config_mut(&mut self) -> &mut GGUFConfig {
        &mut self.config
    }

    /// Token embedding weights (f32, read-only).
    #[must_use]
    pub fn token_embedding(&self) -> &[f32] {
        &self.token_embedding
    }

    /// Position embedding weights (GPT-2 only, read-only).
    #[must_use]
    pub fn position_embedding(&self) -> Option<&[f32]> {
        contract_pre_position_embedding!();
        self.position_embedding.as_deref()
    }

    /// Transformer layers (read-only).
    #[must_use]
    pub fn layers(&self) -> &[OwnedQuantizedLayer] {
        &self.layers
    }

    /// Output layer norm weight (f32, read-only).
    #[must_use]
    pub fn output_norm_weight(&self) -> &[f32] {
        &self.output_norm_weight
    }

    /// Output layer norm bias (optional, read-only).
    #[must_use]
    pub fn output_norm_bias(&self) -> Option<&[f32]> {
        self.output_norm_bias.as_deref()
    }

    /// LM head weight (quantized, read-only).
    #[must_use]
    pub fn lm_head_weight(&self) -> &OwnedQuantizedTensor {
        &self.lm_head_weight
    }

    /// LM head bias (optional f32, read-only).
    #[must_use]
    pub fn lm_head_bias(&self) -> Option<&[f32]> {
        self.lm_head_bias.as_deref()
    }
}

include!("model_owned_quantized.rs");
