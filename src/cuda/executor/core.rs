//! Core CudaExecutor implementation: constructor, availability, profiling API, graph tracking
//!
//! This module contains the core functionality of the CudaExecutor including:
//! - Constructor and device initialization
//! - Availability checking
//! - BrickProfiler API for per-brick timing
//! - Execution graph tracking for ASCII tree visualization
//! - Tile-level profiling for cache hierarchy analysis
//! - Context, device, and pool management methods

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

impl CudaExecutor {
    /// Create a new CUDA executor for the specified device
    ///
    /// # Arguments
    ///
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(device_ordinal: i32) -> Result<Self, GpuError> {
        // Ensure process-level sentinel keeps the primary context alive.
        // This prevents cuDevicePrimaryCtxRelease from destroying the context
        // when individual executors drop (sentinel holds refcount ≥ 1).
        ensure_sentinel(device_ordinal)?;

        // Check out a pooled context to avoid cuDevicePrimaryCtxRetain churn.
        // First call creates a fresh context; subsequent calls reuse the pooled one.
        let context = checkout_context(device_ordinal)?;
        let (compute_stream, transfer_stream, stream) = checkout_streams(&context)?;

        // PMAT-044: Detect GPU profile FIRST so sm_target is available for CudaKernels.
        // All PTX generation uses this target — no hardcoded sm_70.
        let gpu_profile = GpuProfile::detect(&context);

        Ok(Self {
            // Initialize in struct declaration order (for clarity)
            kernels: CudaKernels::with_target(gpu_profile.sm_target.clone()),
            memory_pool: GpuMemoryPool::new(),
            staging_pool: StagingBufferPool::new(), // PARITY-042: pinned memory pool
            modules: std::mem::ManuallyDrop::new(HashMap::new()),
            weight_cache: HashMap::new(),
            named_fp16_weight_cache: HashMap::new(), // GH-174: SafeTensors F16
            quantized_weight_cache: HashMap::new(),  // PAR-005: quantized weight cache
            quantized_weight_types: HashMap::new(),  // PAR-058: weight quant types
            quantized_weight_pool: None,             // ALB-098: pool allocator
            quantized_weight_pool_entries: HashMap::new(),
            rmsnorm_cache: HashMap::new(), // PAR-023: RMSNorm gamma cache
            bias_cache: HashMap::new(),    // BIAS-FIX: QKV bias cache
            // PAR-043: Pre-indexed layer weights for O(1) access
            indexed_layer_weights: Vec::new(),
            output_norm_ptr: 0,
            output_norm_len: 0,
            lm_head_ptr: 0,
            lm_head_len: 0,
            lm_head_qtype: WeightQuantType::Q4K, // Default, updated on weight load
            lm_head_bias_ptr: 0,
            lm_head_bias_len: 0,
            logits_buffer: None,
            logits_buffer_size: 0,
            workspace: TransformerWorkspace::default(), // PAR-044: lazy init on first forward
            gemv_input_buffer: None,                    // PAR-007: lazy init on first GEMV
            gemv_output_buffer: None,
            gemv_input_size: 0,
            gemv_output_size: 0,
            // ALB-111: Extra output buffers for batched GEMV (batch Q/K/V, gate/up)
            gemv_output_buffer_b: None,
            gemv_output_buffer_c: None,
            gemv_output_size_b: 0,
            gemv_output_size_c: 0,
            kv_cache_gpu: HashMap::new(), // PAR-018 + PAR-021: GPU-resident KV cache
            kv_cache_lengths: HashMap::new(),
            kv_cache_max_len: 0,
            kv_num_heads: 0,
            kv_num_kv_heads: 0, // PAR-021 GQA
            kv_head_dim: 0,
            rope_theta: 10000.0, // PAR-060: default RoPE theta
            rope_type: 0,        // CORRECTNESS-011: default NORM style
            compute_stream: PoolableStream::new(compute_stream),
            transfer_stream: PoolableStream::new(transfer_stream),
            stream: PoolableStream::new(stream),
            // PAR-054: CUDA Graph Capture (lazy init on first decode)
            decode_graph: None,
            decode_event: None, // PMAT-283: lazily created on first graph capture
            attention_event: None, // GH-559-PERF: lazily created on first attention pass
            position_buf: None,
            seq_len_buf: None,
            // PAR-119: Batched KV caches (lazy init in init_batched_kv_cache)
            batched_kv_k_caches: HashMap::new(),
            batched_kv_v_caches: HashMap::new(),
            batched_kv_lengths: Vec::new(),
            batched_k_ptrs: None,
            batched_v_ptrs: None,
            batched_seq_lens_gpu: None,
            batched_k_ptrs_per_layer: HashMap::new(),
            batched_v_ptrs_per_layer: HashMap::new(),
            batched_kv_stride: 0,
            batched_kv_allocated_batch: 0,
            // PAR-121: Batched graph fields
            batched_decode_graphs: HashMap::new(),
            batched_graph_input_buf: None,
            batched_graph_positions_buf: None,
            batched_graph_seq_lens_buf: None,
            batched_graph_batch_size: 0,
            // PMAT-050: Prefill graph fields
            prefill_graphs: HashMap::new(),
            prefill_graph_input_buf: None,
            graph_input_buf: None,
            decode_token_count: 0,
            // PAR-068: Pre-allocated argmax buffers (lazy init on first use)
            argmax_block_vals: None,
            argmax_block_idxs: None,
            argmax_result: None,
            argmax_num_blocks: 0,
            batched_argmax_results: None,
            batched_argmax_results_cap: 0,
            batched_decode_input_buf: None,
            batched_decode_input_cap: 0,
            // PAR-118: Graph capture failure tracking
            graph_capture_failed: false,
            prefill_graph_capture_failed: false,
            is_capturing: false,
            graph_recording: false,
            graph_recorded_kernels: Vec::new(),
            is_prefilling: false,
            // PAR-118: Flash Decoding (disabled by default, enable via init_flash_decoding)
            flash_decode_partials: None,
            flash_decode_max_seq_len: 0,
            flash_decode_enabled: false,
            flash_decode_k_ptrs: HashMap::new(),
            flash_decode_v_ptrs: HashMap::new(),
            flash_decode_max_chunks: 0,
            flash_decode_seq_lens_buf: None,
            // QWEN-007: Q8 KV cache (disabled by default, enable via init_kv_cache_q8_gpu)
            kv_cache_q8_enabled: false,
            kv_cache_q8_k: HashMap::new(),
            kv_cache_q8_v: HashMap::new(),
            kv_cache_q8_k_scales: HashMap::new(),
            kv_cache_q8_v_scales: HashMap::new(),
            // PMAT-024: cuBLAS handle for prefill GEMM (lazy init on first prefill)
            cublas_handle: None,
            // PMAT-063: cuBLAS workspace for graph capture (lazy init)
            cublas_workspace: None,
            // PMAT-053: cuBLASLt handle for FP8 GEMM (lazy init on first FP8 prefill)
            cublaslt_handle: None,
            dequant_scratch: None,
            dequant_scratch_size: 0,
            // PMAT-031: FP16 weight cache for HGEMM prefill
            fp16_weight_cache: HashMap::new(),
            fp16_activation_scratch: None,
            fp16_activation_scratch_size: 0,
            fp16_dequant_temp: None, // PMAT-065: lazy init on first L2-cached HGEMM
            // PMAT-053: FP8 weight cache (sm_89+ only, enabled with FP8_PREFILL=1)
            fp8_weight_cache: HashMap::new(),
            fp8_activation_scratch: None,
            fp8_activation_scratch_size: 0,
            // PMAT-091: Interleaved Q4K weight cache (warmup at model init)
            interleaved_weight_cache: HashMap::new(),
            // PMAT-064: Q4K WMMA padded output scratch (lazy init)
            wmma_scratch: None,
            // PMAT-066: Q8 activation scratch for DP4A Q4K GEMM (lazy init)
            dp4a_q8_scratch: None,
            // PMAT-032: Prefill attention score scratch (lazy init)
            prefill_attn_scores: None,
            prefill_attn_scores_size: 0,
            // QWEN-010: Auto-tune tile size based on GPU
            // RTX 4090 (sm_89) has 72MB L2 cache - use 64x64 tiles
            // Default: 32x32 tiles for other GPUs
            optimal_tile_size: Self::detect_optimal_tile_size(&context),
            // PAR-073: BrickProfiler (disabled by default for zero overhead)
            // Enable with executor.enable_profiling() for per-brick timing
            profiler: trueno::BrickProfiler::new(),
            // Auto-detect kernel variants from GPU hardware (replaces env var tuning)
            gpu_profile,
            // SM count for grid scaling (fallback 8 = Jetson Orin Nano)
            num_sms: context.multiprocessor_count().unwrap_or(8) as u32,
            // PMAT-027: Q8 activation cache starts invalid
            q8_activation_valid: false,
            fp8_activation_cache_key: None,
            fp8_weight_scales: HashMap::new(),
            fp8_act_scale_buf: None,
            fp8_absmax_buf: None,
            fp8_act_dequant_buf: None,
            graph_dispatch_positions: Vec::new(),
            batched_done_mask: Vec::new(),
            hgemm_batched_decode_active: false,
            context: std::mem::ManuallyDrop::new(context), // Last field — ManuallyDrop skips cuDevicePrimaryCtxRelease - dropped last
        })
    }

    /// Check if CUDA is available on this system
    #[must_use]
    pub fn is_available() -> bool {
        cuda_available()
    }

    /// ALB-110: Set the CUDA context as current on the calling thread.
    ///
    /// CUDA contexts are thread-local state. When a `CudaExecutor` is created
    /// on one thread and moved to another (e.g., `std::thread::spawn`), the
    /// CUDA context is NOT automatically current on the new thread. All CUDA
    /// driver API calls (`cuMemAlloc`, `cuStreamCreate`, kernel launches, etc.)
    /// require a current context — without it, operations may silently corrupt
    /// GPU state and crash after N requests.
    ///
    /// **Must be called** at the start of any thread that receives a moved
    /// `CudaExecutor`, before any GPU operations.
    pub fn make_context_current(&self) -> Result<(), GpuError> {
        self.context.make_current()
    }

    /// QWEN-010: Detect optimal tile size based on GPU architecture
    ///
    /// RTX 4090 (Ada Lovelace, sm_89) has 72MB L2 cache vs A100's 40MB.
    /// Larger tiles (64x64) improve L2 cache utilization on RTX 4090.
    fn detect_optimal_tile_size(context: &CudaContext) -> u32 {
        // Get device name for GPU detection
        let device_name = context.device_name().unwrap_or_default();

        // RTX 4090, RTX 4080, RTX 4070 (Ada Lovelace, sm_89) benefit from 64x64 tiles
        // These GPUs have 72MB, 64MB, 48MB L2 cache respectively
        if device_name.contains("4090")
            || device_name.contains("4080")
            || device_name.contains("4070")
        {
            64
        } else {
            // Default: 32x32 tiles for other GPUs (A100, V100, older consumer cards)
            32
        }
    }

    /// Get the optimal tile size for this GPU
    ///
    /// Returns 64 for RTX 40-series (Ada Lovelace), 32 for other GPUs.
    #[must_use]
    pub fn optimal_tile_size(&self) -> u32 {
        self.optimal_tile_size
    }

    /// Compile PTX into a CUDA module, with process-level blocklisting.
    ///
    /// If the same PTX previously failed to compile (poisoning the CUDA
    /// context), this method returns an error immediately without calling
    /// `cuModuleLoadData`, preventing repeated context poisoning.
    pub(crate) fn compile_ptx(&self, ptx: &str) -> Result<CudaModule, GpuError> {
        contract_pre_jit_compilation_success!();
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ptx.hash(&mut hasher);
        let hash = hasher.finish();

        {
            let bl = BROKEN_PTX
                .lock()
                .expect("BROKEN_PTX mutex poisoned in blocklist check");
            if bl.contains(&hash) {
                return Err(GpuError::ModuleLoad(
                    "kernel blocklisted (previous compilation failure)".to_string(),
                ));
            }
        }
        match CudaModule::from_ptx(&self.context, ptx) {
            Ok(m) => Ok(m),
            Err(e) => {
                if verbose() {
                    eprintln!(
                        "[CUDA-POOL] PTX compilation failed (hash={hash:016x}), blocklisting"
                    );
                }
                BROKEN_PTX
                    .lock()
                    .expect("BROKEN_PTX mutex poisoned in blocklist insert")
                    .insert(hash);
                Err(e)
            },
        }
    }
}

/// Custom Drop: synchronize, then return streams and context to pools.
///
/// Erlang-style fail-fast: synchronize to detect any async kernel crashes
/// BEFORE returning resources to the pool.  A poisoned stream/context in the
/// pool would cascade failures to every subsequent test.
///
/// After this runs, Rust auto-drops remaining fields in declaration order:
/// - GPU buffers (cuMemFree) — context thread-local pointer is still set
/// - PoolableStream wrappers — inner is None (already extracted), no-op
/// - modules (ManuallyDrop) — intentionally leaked, no cuModuleUnload
/// - context (ManuallyDrop<CudaContext>) — returned to pool or leaked
impl Drop for CudaExecutor {
    fn drop(&mut self) {
        // Fail-fast: synchronize to detect async kernel crashes immediately.
        // Without this, a crashing kernel silently poisons the pooled
        // streams/context and cascades failures to ALL subsequent tests.
        let ctx_healthy = self.context.synchronize().is_ok();

        // Extract streams from PoolableStream wrappers.
        let compute = self.compute_stream.take();
        let transfer = self.transfer_stream.take();
        let legacy = self.stream.take();

        if ctx_healthy {
            // Context is healthy — return resources to pools for reuse.
            if let (Some(s1), Some(s2), Some(s3)) = (compute, transfer, legacy) {
                checkin_streams(s1, s2, s3);
            }
            // SAFETY: ManuallyDrop::take called exactly once; context is valid and healthy
            let ctx = unsafe { std::mem::ManuallyDrop::take(&mut self.context) };
            checkin_context(ctx);
        } else {
            // Context is POISONED — do NOT return resources to pools.
            // Let streams drop normally (cuStreamDestroy — will fail, that's OK).
            // Let context drop normally (cuDevicePrimaryCtxRelease).
            // The sentinel still holds one retain, keeping the primary context
            // alive.  The NEXT executor will get a fresh context from the pool
            // (empty → CudaContext::new) which retains the same primary context.
            // If the primary context itself is irrecoverable, the next
            // CudaContext::new will also fail — surfacing the error immediately.
            eprintln!(
                "[CUDA-FAILFAST] Context poisoned during executor lifetime — \
                 streams and context NOT returned to pool. \
                 Next executor will create fresh resources."
            );
            // SAFETY: ManuallyDrop::take called exactly once; letting poisoned context drop
            let _ctx = unsafe { std::mem::ManuallyDrop::take(&mut self.context) };
        }
    }
}

include!("executor_api.rs");
include!("core_executor_tests.rs");
