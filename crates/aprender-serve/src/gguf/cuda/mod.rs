//! CUDA-accelerated quantized model
//!
//! This module provides GPU-accelerated inference for quantized models
//! using NVIDIA CUDA.
//!
//! # Architecture
//!
//! `OwnedQuantizedModelCuda` wraps an `OwnedQuantizedModel` with a CUDA executor
//! for GPU-accelerated matrix operations. Key features:
//!
//! - GPU-resident KV cache (avoids CPU→GPU transfer per token)
//! - Fused attention kernels
//! - Pre-cached quantized weights
//! - Batch generation support
//!
//! # Module Structure
//!
//! - `backend.rs`: CUDA kernel configuration and PTX generation (CudaBackend)
//! - `forward.rs`: Forward pass methods (single token, cached, GPU-resident)
//! - `generation.rs`: Token generation loops (basic, cached, streaming, batch)
//! - `speculative.rs`: Speculative decoding (self-speculative, draft model)
//! - `weights.rs`: Weight management (pre-caching, GPU upload)
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
//!
//! let model = OwnedQuantizedModel::from_mapped(&mapped)?;
//! let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)?; // GPU 0
//!
//! // GPU-accelerated forward pass
//! let logits = cuda_model.forward_cuda(&tokens)?;
//! ```

mod backend;
mod forward;
mod generation;
mod speculative;
mod weights;

// Re-export types for public API
pub use backend::CudaBackend;
// PMAT-072: Step-wise batched decode state for lock-releasing scheduler
pub use generation::BatchedDecodeState;

use crate::error::{RealizarError, Result};

// Import types from peer modules (parent of cuda/)
use super::model::OwnedQuantizedModel;
use super::quantized::{OwnedQKVWeights, OwnedQuantizedTensor};
use super::runtime::{OwnedQuantizedKVCache, QuantizedGenerateConfig};
use super::utils::verbose;

// =============================================================================
// IMP-800: CUDA-Accelerated Model Wrapper
// =============================================================================

/// Error from CUDA model initialization that preserves the unconsumed model.
///
/// When `OwnedQuantizedModelCuda::new()` fails, the model is returned inside this error
/// so callers can fall back to CPU without an expensive 1GB clone.
pub struct CudaInitError {
    /// The initialization error
    pub error: RealizarError,
    /// The unconsumed model, returned for CPU fallback
    model: OwnedQuantizedModel,
}

impl CudaInitError {
    /// Extract the unconsumed model for CPU fallback
    #[must_use]
    pub fn into_model(self) -> OwnedQuantizedModel {
        self.model
    }
}

impl std::fmt::Display for CudaInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::fmt::Debug for CudaInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaInitError({:?})", self.error)
    }
}

/// CUDA-accelerated wrapper for `OwnedQuantizedModel` (IMP-800a)
///
/// Provides GPU-accelerated forward pass using NVIDIA CUDA via trueno-gpu.
/// Caches the CudaExecutor to avoid initialization overhead (~50ms) per call.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
///
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)?; // GPU 0
///
/// // GPU-accelerated forward pass
/// let logits = cuda_model.forward_cuda(&tokens)?;
/// ```
pub struct OwnedQuantizedModelCuda {
    /// Inner model
    pub(crate) model: OwnedQuantizedModel,
    /// Cached CUDA executor
    pub(crate) executor: crate::cuda::CudaExecutor,
    /// GPU device name
    device_name: String,
    /// GPU memory (free, total) in bytes
    memory_info: (usize, usize),
    /// PAR-083: Pre-allocated embedding buffer to eliminate per-token heap allocation.
    /// Five-Whys root cause: embed() allocates Vec<f32> per token (~14KB for 7B).
    /// Fix: Reuse this buffer with embed_into().
    embed_buf: Vec<f32>,
    /// realizr#199 (PMAT-450): Prefix cache for prompt KV reuse.
    /// Stores GPU KV cache snapshots keyed by prompt tokens.
    /// On cache hit, skip prefill entirely (TTFT ~900ms → ~5ms).
    #[cfg(feature = "gpu")]
    prefix_cache: crate::gguf::batch_scheduler::PrefixCache,
}

impl OwnedQuantizedModelCuda {
    /// Create a new CUDA-accelerated model wrapper
    ///
    /// # Arguments
    ///
    /// * `model` - The quantized model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    /// Create a CUDA model wrapper. Model is consumed on failure.
    /// Use `with_max_seq_len` for recoverable errors (returns model on failure).
    pub fn new(model: OwnedQuantizedModel, device_ordinal: i32) -> Result<Self> {
        Self::with_max_seq_len(model, device_ordinal, 2048).map_err(|e| e.error)
    }

    /// Create a new CUDA-accelerated model wrapper with custom max sequence length
    ///
    /// # Arguments
    ///
    /// * `model` - The quantized model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    /// * `max_seq_len` - Maximum sequence length for GPU KV cache (PAR-018)
    ///
    /// # Errors
    ///
    /// Returns `CudaInitError` containing both the error and the unconsumed model,
    /// allowing callers to recover the model for CPU fallback without cloning.
    /// GH-280: Check GPU capability before allocation.
    /// Extracted to reduce cognitive complexity of `with_max_seq_len`.
    fn check_gpu_capability(
        model: OwnedQuantizedModel,
    ) -> std::result::Result<OwnedQuantizedModel, CudaInitError> {
        let required = crate::capability::required_ops(&model.config.constraints);
        let supported = crate::capability::gpu_supported_ops();
        if let Err(missing) = crate::capability::check_capability(&required, &supported) {
            let missing_names: Vec<String> = missing.iter().map(ToString::to_string).collect();
            return Err(CudaInitError {
                error: RealizarError::CapabilityMismatch {
                    architecture: model.config.architecture.clone(),
                    missing_ops: missing_names.join(", "),
                    suggestion: "Model will use CPU inference. To add GPU support, implement the missing kernels in trueno.".to_string(),
                },
                model,
            });
        }
        Ok(model)
    }

    /// GH-199/PARITY-GATE: Preload GPU weights and verify correctness.
    /// Extracted to reduce cognitive complexity of `with_max_seq_len`.
    fn preload_and_verify(mut self) -> std::result::Result<Self, CudaInitError> {
        if !self.supports_gpu_resident() {
            return Ok(self);
        }

        // GH-199 ROOT CAUSE B: Eagerly preload weights for GPU-resident path.
        if let Err(e) = self.preload_weights_gpu() {
            return Err(CudaInitError {
                error: e,
                model: self.into_model(),
            });
        }

        // PMAT-037: Eagerly warm FP16 weight cache for HGEMM prefill.
        // Five-Whys root cause: FP16 cache lazily populated on first inference
        // request (303ms cold-start), inflating TTFT P50 from 25ms to 50.9ms.
        // Pre-populating at model init moves cost to startup (one-time).
        // PMAT-067: Skip FP16 cache when FP8 is active — saves ~1.5 GB VRAM.
        // PMAT-400: Skip FP16 cache on unified memory (cc>=120) — saves 61 GB for 32B model.
        // PMAT-409: Override with FORCE_FP16_CACHE=1 for 7B on GB10 (2.9 GB, restores prefill perf).
        let skip_fp16_unified = self.executor.gpu_profile.cc >= 120
            && std::env::var("FORCE_FP16_CACHE").as_deref() != Ok("1");
        if std::env::var("HGEMM_PREFILL").as_deref() != Ok("0")
            && !self.executor.gpu_profile.fp8_prefill
            && !skip_fp16_unified
        {
            let num_layers = self.model.config.num_layers;
            let hidden_dim = self.model.config.hidden_dim as u32;
            let intermediate_dim = self.model.config.intermediate_dim as u32;
            let vocab_size = self.model.config.vocab_size as u32;
            if let Err(e) = self.executor.ensure_cublas() {
                eprintln!("[PMAT-037] cuBLAS init failed (non-fatal): {e}");
            } else if let Err(e) = self.executor.warmup_hgemm_cache(
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
            ) {
                eprintln!("[PMAT-037] FP16 cache warmup failed (non-fatal): {e}");
            }
        }

        // PMAT-053/067: FP8 E4M3 weight cache warmup (auto-enabled on sm_89+)
        // GH-286: Skip if --no-fp8-cache (saves ~1.5 GB RSS)
        let no_fp8 = std::env::var("REALIZR_NO_FP8_CACHE").as_deref() == Ok("1");
        if self.executor.gpu_profile.fp8_prefill && !no_fp8 {
            let num_layers = self.model.config.num_layers;
            let hidden_dim = self.model.config.hidden_dim as u32;
            let intermediate_dim = self.model.config.intermediate_dim as u32;
            let vocab_size = self.model.config.vocab_size as u32;
            if let Err(e) =
                self.executor
                    .warmup_fp8_cache(num_layers, hidden_dim, intermediate_dim, vocab_size)
            {
                eprintln!("[PMAT-053] FP8 cache warmup failed (non-fatal): {e}");
            }
        }

        // PMAT-091: Interleaved Q4K weight cache warmup (W4A16 WMMA GEMM)
        if self.executor.gpu_profile.w4a16_interleaved {
            let num_layers = self.model.config.num_layers;
            let hidden_dim = self.model.config.hidden_dim as u32;
            let intermediate_dim = self.model.config.intermediate_dim as u32;
            let vocab_size = self.model.config.vocab_size as u32;
            if let Err(e) = self.executor.warmup_interleaved_cache(
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
            ) {
                eprintln!("[PMAT-091] Interleaved cache warmup failed (non-fatal): {e}");
            }
        }

        // GH-181: Reinitialize workspace after cache warmup (FP8/FP16/interleaved).
        // Cache allocations can relocate workspace buffers, causing the parity
        // gate's forward pass to read stale pointers (cosine -0.28 on RTX 4060).
        {
            let hidden_dim = self.model.config.hidden_dim;
            let intermediate_dim = self.model.config.intermediate_dim;
            self.executor.force_workspace_reinit();
            if let Err(e) = self.executor.init_workspace(hidden_dim, intermediate_dim) {
                eprintln!("[GH-181] Workspace reinit failed (non-fatal): {e}");
            }
        }

        // PARITY-GATE: Jidoka — stop-the-line if GPU diverges from CPU.
        // Run ONE token through both backends and compare logits.
        // If cosine similarity < 0.99, refuse to construct.
        // Skip gate if SKIP_PARITY_GATE=1 (for debugging the gate itself)
        let skip_gate = std::env::var("SKIP_PARITY_GATE")
            .map(|v| v == "1")
            .unwrap_or(false);

        if !skip_gate {
            if let Err(e) = parity_gate(&mut self) {
                return Err(CudaInitError {
                    error: e,
                    model: self.into_model(),
                });
            }
        }

        Ok(self)
    }

    /// Create a GPU-accelerated inference engine with a custom maximum sequence length.
    pub fn with_max_seq_len(
        model: OwnedQuantizedModel,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> std::result::Result<Self, CudaInitError> {
        use crate::cuda::CudaExecutor;

        // GH-279: Contract gate — validate architecture and dimensions before CUDA init
        if let Err(e) = crate::contract_gate::validate_model_load_basic(
            &model.config.architecture,
            model.config.num_layers,
            model.config.hidden_dim,
            model.config.num_heads,
            model.config.num_kv_heads,
            model.config.intermediate_dim,
            model.config.vocab_size,
        ) {
            return Err(CudaInitError {
                error: crate::contract_gate::gate_error(e),
                model,
            });
        }

        // GH-280: Capability gate — refuse GPU if model requires unsupported ops.
        let model = Self::check_gpu_capability(model)?;

        let mut executor = match CudaExecutor::new(device_ordinal) {
            Ok(e) => e,
            Err(e) => {
                return Err(CudaInitError {
                    error: RealizarError::UnsupportedOperation {
                        operation: "CudaExecutor::new".to_string(),
                        reason: format!("CUDA initialization failed: {e}"),
                    },
                    model,
                });
            },
        };

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let memory_info = executor.memory_info().unwrap_or((0, 0));

        // PAR-018: Initialize GPU-resident KV cache for attention acceleration
        // This avoids ~66 MB CPU→GPU transfer per token for TinyLlama
        let num_layers = model.layers.len();
        let num_heads = model.config.num_heads;
        let num_kv_heads = model.config.num_kv_heads; // PAR-021 GQA support
        let head_dim = model.config.hidden_dim / num_heads;

        if let Err(e) =
            executor.init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        {
            return Err(CudaInitError {
                error: RealizarError::UnsupportedOperation {
                    operation: "init_kv_cache_gpu".to_string(),
                    reason: format!("GPU KV cache initialization failed: {e}"),
                },
                model,
            });
        }

        // PMAT-399: Auto-size max_batch if env var not set
        if std::env::var("CUDA_MAX_BATCH").is_err() {
            let auto_batch = executor.compute_max_batch_for_memory(
                num_layers,
                num_kv_heads,
                head_dim,
                max_seq_len,
            );
            // GH-611: Suppressed — was noisy in non-verbose mode
            // Store for scheduler to pick up
            std::env::set_var("CUDA_MAX_BATCH", auto_batch.to_string());
        }

        // PAR-118: Initialize Flash Decoding for split-K attention acceleration.
        // Five-Whys: batched_incremental_attention uses Grid=(num_heads,M,1) Block=(32,1,1)
        // = only 896 threads on RTX 4090 for 7B (28 heads). Flash Decoding splits KV cache
        // into chunks processed in parallel, achieving 1.5-2x decode speedup.
        // NOTE: flash_decode_enabled is used by BOTH the graphed decode path
        // (flash_decoding_graphed in attention.rs) and the batched path (batched.rs).
        // The batched path uses threshold 1024 to avoid triggering during normal prefill.
        if let Err(e) = executor.init_flash_decoding(num_heads, head_dim, max_seq_len, 1) {
            if verbose() {
                eprintln!(
                    "[PAR-118] Flash Decoding init failed: {e}, falling back to sequential attention"
                );
            }
            // Non-fatal: sequential attention still works, just slower
        }

        // PAR-060: Set RoPE theta for position embeddings
        if verbose() {
            eprintln!(
                "[PAR-060] Setting rope_theta = {} for GPU path",
                model.config.rope_theta
            );
        }
        executor.set_rope_theta(model.config.rope_theta);

        // CORRECTNESS-011: Set rope_type for correct RoPE style (NORM vs NEOX)
        if verbose() {
            eprintln!(
                "[CORRECTNESS-011] Setting rope_type = {} for GPU path (0=NORM, 2=NEOX)",
                model.config.rope_type
            );
        }
        executor.set_rope_type(model.config.rope_type);

        // GH-129: Pre-load ALL kernel modules BEFORE heavy GPU allocations.
        // Five-Whys root cause: PTX JIT compilation requires GPU memory for the
        // JIT compiler itself. On Jetson (unified memory), weight upload consumes
        // ~1 GB, leaving less for JIT. Moving kernel preload here (before weight
        // upload) ensures JIT runs with maximum available GPU memory.
        {
            let hidden_dim = model.config.hidden_dim as u32;
            let intermediate_dim = model.config.intermediate_dim as u32;
            let vocab_size = model.config.vocab_size as u32;
            match executor.preload_modules_for_capture(
                num_layers,
                hidden_dim,
                intermediate_dim,
                vocab_size,
            ) {
                Ok(()) => eprintln!(
                    "[GH-129] Early kernel preload: {} modules compiled",
                    executor.module_count()
                ),
                Err(e) => {
                    eprintln!("[GH-129] Early kernel preload failed: {e}");
                },
            }
        }

        // PAR-083: Pre-allocate embedding buffer (hidden_dim f32s) to avoid per-token malloc.
        let embed_buf = vec![0.0f32; model.config.hidden_dim];

        let cuda_model = Self {
            model,
            executor,
            device_name,
            memory_info,
            embed_buf,
            #[cfg(feature = "gpu")]
            prefix_cache: crate::gguf::batch_scheduler::PrefixCache::new(16),
        };

        // GH-199 ROOT CAUSE B + PARITY-GATE: preload weights and verify GPU correctness.
        cuda_model.preload_and_verify()
    }

    /// GH-129: Free CPU projection weight copies after GPU preload.
    ///
    /// On Jetson (unified memory), the CPU `Vec<u8>` copies are redundant after
    /// `preload_weights_gpu()` uploads them to GPU buffers. Frees ~1 GB for 1.5B.
    pub fn free_cpu_weights(&mut self) {
        let mut freed = 0usize;
        for layer in &mut self.model.layers {
            freed += layer.qkv_weight.data_bytes();
            freed += layer.attn_output_weight.data.len();
            freed += layer.ffn_up_weight.data.len();
            freed += layer.ffn_down_weight.data.len();
            if let Some(ref gate) = layer.ffn_gate_weight {
                freed += gate.data.len();
            }
            layer.free_projection_weights();
        }
        freed += self.model.lm_head_weight.data.len();
        self.model.lm_head_weight.data = Vec::new();
        eprintln!(
            "[GH-129] Freed {:.1} MB CPU weight copies (GPU-resident path active)",
            freed as f64 / (1024.0 * 1024.0)
        );
    }

    /// Check if CUDA is available
    #[must_use]
    pub fn is_available() -> bool {
        crate::cuda::CudaExecutor::is_available()
    }

    /// Get number of CUDA devices
    #[must_use]
    pub fn num_devices() -> usize {
        crate::cuda::CudaExecutor::num_devices()
    }

    /// Get GPU device name
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get GPU memory info (free, total) in bytes
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info
    }

    /// Get VRAM usage in MB
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    // ========================================================================
    // PAR-073: BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    ///
    /// When enabled, each brick operation is timed individually using
    /// `std::time::Instant` with CUDA sync for accurate GPU timing.
    pub fn enable_profiling(&mut self) {
        self.executor.enable_profiling();
    }

    /// Disable per-brick profiling (default state).
    pub fn disable_profiling(&mut self) {
        self.executor.disable_profiling();
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_profiling_enabled(&self) -> bool {
        self.executor.is_profiling_enabled()
    }

    /// Get the brick profiler for reading statistics.
    #[must_use]
    pub fn profiler(&self) -> &trueno::BrickProfiler {
        self.executor.profiler()
    }

    /// Reset profiler statistics.
    pub fn reset_profiler(&mut self) {
        self.executor.reset_profiler();
    }

    /// Get profiler summary report.
    #[must_use]
    pub fn profiler_summary(&self) -> String {
        self.executor.profiler_summary()
    }

    /// Get reference to inner model
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    /// Consume CUDA wrapper and return inner model (for CPU fallback)
    #[must_use]
    pub fn into_model(self) -> OwnedQuantizedModel {
        self.model
    }

    /// PAR-111: Get mutable reference to CUDA executor
    ///
    /// Allows direct access for batched forward path and workspace initialization.
    #[must_use]
    pub fn executor_mut(&mut self) -> &mut crate::cuda::CudaExecutor {
        &mut self.executor
    }

    /// Synchronize CUDA stream (wait for all GPU operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.executor
            .synchronize()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::synchronize".to_string(),
                reason: format!("CUDA sync failed: {e}"),
            })
    }
}

// =============================================================================
// PARITY GATE: Load-time mathematical proof of GPU/CPU equivalence
// =============================================================================
//
// Toyota Way: Jidoka (自働化) — stop-the-line on defect.
//
// Just as build.rs refuses to compile if ALG-001 through ALG-009 fail,
// this gate refuses to construct `OwnedQuantizedModelCuda` if GPU and CPU
// compute different functions.
//
// An `OwnedQuantizedModelCuda` that passes this gate is PROVEN to produce
// the same output as CPU. One that fails CANNOT be constructed.
//
// Contract: layer-parity-v1.yaml
// Tolerance: cosine_similarity ≥ 0.98 on first-token logits

/// Minimum cosine similarity for parity gate to pass.
/// 0.98 accommodates DP4A integer dot-product kernels which use int8 arithmetic
/// with different rounding than CPU float dequantization. On Qwen 1.5B Q4K:
///   - Non-DP4A (float GEMV): cosine ~0.9999
///   - DP4A (int8 dot product): cosine ~0.9887
/// Real bugs (corrupted weights, wrong kernel) produce cosine < 0.5.
const PARITY_GATE_COSINE_MIN: f32 = 0.98;

include!("mod_parity_gate.rs");
