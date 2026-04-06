//! GPU-resident transformer trainer (ALB-040)
//!
//! Wires the existing `CudaTransformerBlock` forward/backward/optimizer_step
//! into the pretraining path. Follows the proven `classify_pipeline.rs` pattern.
//!
//! # Architecture
//!
//! ```text
//! CudaTransformerTrainer
//! ├── model: Transformer                 (CPU — embed + save)
//! ├── cuda_trainer: CudaTrainer          (GPU device context)
//! ├── cuda_blocks: Vec<CudaBlock>            (fp32 or NF4)
//! ├── cuda_grad_workspace: CudaGradWorkspace
//! ├── gpu_training: GpuPretrainState     (layer_inputs, grad bufs, opt states)
//! ├── lm_head_weight_gpu: GpuBuffer      (V × H on GPU)
//! ├── lm_head_grad_gpu: GpuBuffer        (V × H gradient scratch)
//! ├── lm_head_m/v: GpuBuffer             (AdamW moment states)
//! └── config: TransformerTrainConfig
//! ```
//!
//! # Transfer budget (C-GPUTRAIN-002, updated KAIZEN-050/052)
//!
//! 1 PCIe transfer per training step (+ tiny control transfers):
//! 1. H2D: hidden states after embedding (seq×H×4 bytes)
//! 2. H2D: target_ids for fused cross-entropy (seq×4 bytes — ~512B)
//! 3. D2H: loss_partials from fused cross-entropy (seq×4 bytes — ~512B)
//!
//! Eliminated by KAIZEN-050:
//! - D2H logits (was seq×V×4 = 77.8MB for Qwen3-4B)
//! - H2D grad_logits (was seq×V×4 = 77.8MB)
//!
//! Eliminated by KAIZEN-052:
//! - grad_gpu buffer allocation (was seq×V×4 = 77.8MB per step)

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaStream, GpuBuffer};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::{gemm_backward_a, gemm_backward_b, rms_norm_backward};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{gemm_forward, pre_warm_forward_kernels, rms_norm_forward};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::{
    adamw_step_cuda, clip_scale_reduce_cuda, fused_cross_entropy_cuda, gradient_clip_cuda,
    gradient_clip_gpu_scale_cuda, squared_sum_collect, squared_sum_cuda, squared_sum_launch_cuda,
    squared_sum_launch_into, FusedClipState,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::{cuda_training_available, CudaTrainer};
#[cfg(feature = "cuda")]
use crate::autograd::precision::GradScaler;
#[cfg(feature = "cuda")]
use crate::autograd::Tensor;
#[cfg(feature = "cuda")]
use crate::io::{save_model, Model, ModelFormat, ModelMetadata, SaveConfig};
#[cfg(feature = "cuda")]
use crate::optim::{AdamW, Optimizer};
#[cfg(feature = "cuda")]
use crate::train::MetricsTracker;
#[cfg(feature = "cuda")]
use crate::transformer::{
    CudaBlock, CudaBlockScratch, CudaGradWorkspace, CudaLoraGradWorkspace, CudaTransformerBlock,
    GpuBlockOptimizerState, GpuLoraOptimizerState, Transformer,
};

#[cfg(feature = "cuda")]
use super::batch::LMBatch;
#[cfg(feature = "cuda")]
use super::config::TransformerTrainConfig;
#[cfg(feature = "cuda")]
use super::step_profiler::StepProfiler;

/// Compute gradient L2 norm of the shared workspace via GPU reduction (KAIZEN-054).
///
/// Uses `squared_sum_cuda` per buffer (~1KB D2H each) instead of downloading entire
/// gradient buffers to CPU (was 58 MB+ per block, disabled in ALB-067).
///
/// Free function to avoid borrow conflicts with `&mut self`.
#[cfg(feature = "cuda")]
fn compute_workspace_clip_scale_gpu(
    ws: &CudaGradWorkspace,
    max_norm: f32,
    stream: &CudaStream,
) -> (f32, f32) {
    use crate::autograd::cuda_optim::PendingSquaredSum;

    let all_bufs: [&GpuBuffer<f32>; 9] = [
        &ws.grad_w_q,
        &ws.grad_w_k,
        &ws.grad_w_v,
        &ws.grad_w_o,
        &ws.grad_gate,
        &ws.grad_up,
        &ws.grad_down,
        &ws.grad_input_norm,
        &ws.grad_post_attn_norm,
    ];

    // KAIZEN-055: Launch all 9 squared_sum kernels back-to-back without syncing.
    // Single sync after all launches — reduces 9 pipeline flushes to 1 per block.
    let mut pending: Vec<PendingSquaredSum> = Vec::with_capacity(9);
    for buf in &all_bufs {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        if let Ok(p) = squared_sum_launch_cuda(buf, n, stream) {
            pending.push(p);
        }
    }

    // Single sync point for all 9 kernel launches.
    if stream.synchronize().is_err() {
        return (1.0, 0.0);
    }

    // Collect results: download partial sums (~1KB each) and combine.
    // C-CLIP-001: squared_sum_collect returns sum(x²) = ||g||².
    // Accumulate directly — do NOT re-square (entrenar#311 fix).
    let mut total_sq = 0.0f64;
    for p in &pending {
        if let Ok(sq_norm) = squared_sum_collect(p) {
            total_sq += f64::from(sq_norm); // sq_norm is already ||g||²
        }
    }

    let grad_norm = total_sq.sqrt() as f32; // L2 norm = sqrt(sum of squared norms)
    let scale = if grad_norm > max_norm { max_norm / grad_norm } else { 1.0 };
    (scale, grad_norm)
}

/// Clip all gradient buffers in the shared workspace using GPU-computed L2 norm (KAIZEN-054).
///
/// R-004: Returns pre-clip gradient L2 norm for observability logging.
#[cfg(feature = "cuda")]
fn clip_workspace_gradients(ws: &mut CudaGradWorkspace, max_norm: f32, stream: &CudaStream) -> f32 {
    let (scale, grad_norm) = compute_workspace_clip_scale_gpu(ws, max_norm, stream);
    if (scale - 1.0).abs() < 1e-7 {
        return grad_norm;
    }

    let n_wq = ws.grad_w_q.len() as u32;
    let n_wk = ws.grad_w_k.len() as u32;
    let n_wv = ws.grad_w_v.len() as u32;
    let n_wo = ws.grad_w_o.len() as u32;
    let n_gate = ws.grad_gate.len() as u32;
    let n_up = ws.grad_up.len() as u32;
    let n_down = ws.grad_down.len() as u32;
    let n_inorm = ws.grad_input_norm.len() as u32;
    let n_panorm = ws.grad_post_attn_norm.len() as u32;

    let _ = gradient_clip_cuda(&mut ws.grad_w_q, scale, n_wq, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_k, scale, n_wk, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_v, scale, n_wv, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_o, scale, n_wo, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_gate, scale, n_gate, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_up, scale, n_up, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_down, scale, n_down, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_input_norm, scale, n_inorm, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_post_attn_norm, scale, n_panorm, stream);
    grad_norm
}

/// ALB-078: Fused gradient clipping — entire pipeline stays on GPU.
///
/// Replaces `clip_workspace_gradients` by eliminating the stream.synchronize()
/// and D2H partial-sum download. All computation happens on GPU:
///
/// 1. 9× SquaredSumKernel → write partials to pre-allocated contiguous buffer
/// 2. 1× ClipScaleReduceKernel → reduce partials, compute scale on GPU
/// 3. 9× GradientClipGpuScaleKernel → read scale from GPU, apply to gradients
///
/// Zero sync points, zero D2H transfers per block.
#[cfg(feature = "cuda")]
fn fused_clip_workspace_gradients(
    ws: &mut CudaGradWorkspace,
    max_norm: f32,
    state: &FusedClipState,
    stream: &CudaStream,
) {
    let all_bufs: [&GpuBuffer<f32>; 9] = [
        &ws.grad_w_q,
        &ws.grad_w_k,
        &ws.grad_w_v,
        &ws.grad_w_o,
        &ws.grad_gate,
        &ws.grad_up,
        &ws.grad_down,
        &ws.grad_input_norm,
        &ws.grad_post_attn_norm,
    ];

    // Phase 1: Launch 9 squared_sum kernels into contiguous partials buffer.
    // Each writes to state.partials_buf at its pre-computed offset.
    for (i, buf) in all_bufs.iter().enumerate() {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        let output_ptr = state.partials_buf.as_ptr() + u64::from(state.offsets[i]) * 4;
        let _ = squared_sum_launch_into(buf, n, output_ptr, stream);
    }

    // Phase 2: Reduce all partials and compute clip_scale on GPU.
    // Stream ordering guarantees all squared_sum kernels complete before this runs.
    let _ = clip_scale_reduce_cuda(
        &state.partials_buf,
        state.total_partials,
        max_norm,
        &state.scale_buf,
        stream,
    );

    // Phase 3: Apply clip scale to all 9 gradient buffers.
    // Scale is read from GPU memory — no D2H needed.
    let scale_ptr = state.scale_buf.as_ptr(); // output[0] = clip_scale
    let mut all_bufs_mut: [&mut GpuBuffer<f32>; 9] = [
        &mut ws.grad_w_q,
        &mut ws.grad_w_k,
        &mut ws.grad_w_v,
        &mut ws.grad_w_o,
        &mut ws.grad_gate,
        &mut ws.grad_up,
        &mut ws.grad_down,
        &mut ws.grad_input_norm,
        &mut ws.grad_post_attn_norm,
    ];
    for buf in &mut all_bufs_mut {
        let n = buf.len() as u32;
        if n == 0 {
            continue;
        }
        let _ = gradient_clip_gpu_scale_cuda(buf, scale_ptr, n, stream);
    }
}

/// R-004: Compute gradient L2 norm without clipping (for observability only).
///
/// Uses GPU reduction (KAIZEN-054). Only ~9KB D2H per call.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn compute_workspace_grad_norm(ws: &CudaGradWorkspace, stream: &CudaStream) -> f32 {
    let (_, norm) = compute_workspace_clip_scale_gpu(ws, f32::MAX, stream);
    norm
}

/// ALB-072: Unscale all gradient buffers in the shared workspace by `inv_scale`.
///
/// In fp16 AMP, the fused cross-entropy kernel multiplies loss_scale into the
/// gradient output. All subsequent backward gradients carry this scaling. The
/// GPU block optimizer (AdamW) must receive unscaled gradients — otherwise the
/// second moment `v` overflows f32, producing NaN in early layers.
///
/// This is the GPU-side equivalent of `GradScaler::unscale_and_check()` used
/// for CPU embedding gradients.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn unscale_workspace_gradients(ws: &mut CudaGradWorkspace, inv_scale: f32, stream: &CudaStream) {
    if (inv_scale - 1.0).abs() < 1e-7 {
        return;
    }

    let n_wq = ws.grad_w_q.len() as u32;
    let n_wk = ws.grad_w_k.len() as u32;
    let n_wv = ws.grad_w_v.len() as u32;
    let n_wo = ws.grad_w_o.len() as u32;
    let n_gate = ws.grad_gate.len() as u32;
    let n_up = ws.grad_up.len() as u32;
    let n_down = ws.grad_down.len() as u32;
    let n_inorm = ws.grad_input_norm.len() as u32;
    let n_panorm = ws.grad_post_attn_norm.len() as u32;

    let _ = gradient_clip_cuda(&mut ws.grad_w_q, inv_scale, n_wq, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_k, inv_scale, n_wk, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_v, inv_scale, n_wv, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_w_o, inv_scale, n_wo, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_gate, inv_scale, n_gate, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_up, inv_scale, n_up, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_down, inv_scale, n_down, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_input_norm, inv_scale, n_inorm, stream);
    let _ = gradient_clip_cuda(&mut ws.grad_post_attn_norm, inv_scale, n_panorm, stream);
}

/// GPU-resident training state for pretraining.
///
/// # Contract (C-GPUTRAIN-001)
///
/// - `layer_inputs.len() == num_layers`
/// - All buffers preallocated at init; zero GPU allocations during training
/// - `step` increments monotonically
#[cfg(feature = "cuda")]
struct GpuPretrainState {
    /// Saved layer inputs for backward [num_layers][seq_len * hidden_size]
    layer_inputs: Vec<GpuBuffer<f32>>,
    /// Which layer inputs were saved during forward (activation checkpointing).
    /// When checkpointing is enabled, only checkpoint boundary layers are saved.
    /// Non-saved layers are recomputed from the nearest checkpoint before backward.
    saved_layer_mask: Vec<bool>,
    /// Temporary buffer for activation recomputation [seq_len * hidden_size].
    /// Used as the initial input when recomputing from a checkpoint boundary.
    /// Only allocated when activation checkpointing is enabled.
    recompute_buf: Option<GpuBuffer<f32>>,
    /// Final RMSNorm weight on GPU [hidden_size]
    final_norm_weight: GpuBuffer<f32>,
    /// Final block output (pre-norm) for RMSNorm backward [seq_len * hidden_size]
    blocks_output: GpuBuffer<f32>,
    /// Alternating gradient buffer A [seq_len * hidden_size]
    grad_buf_a: GpuBuffer<f32>,
    /// Alternating gradient buffer B [seq_len * hidden_size]
    grad_buf_b: GpuBuffer<f32>,
    /// Gradient for final norm weight [hidden_size]
    grad_final_norm_weight: GpuBuffer<f32>,
    /// RMSNorm output buffer (reused each step) [seq_len * hidden_size]
    norm_output: GpuBuffer<f32>,
    /// Logits buffer (reused each step) [seq_len * vocab_size]
    logits_buf: GpuBuffer<f32>,
    /// LM head gradient buffer [seq_len * hidden_size] (grad w.r.t. normed hidden)
    lm_head_grad_hidden: GpuBuffer<f32>,
    /// Per-block optimizer states
    optimizer_states: Vec<GpuBlockOptimizerState>,
    /// Optimizer step counter
    step: u32,
}

/// GPU-resident transformer trainer for pretraining.
///
/// Uses `CudaTransformerBlock` forward/backward/optimizer_step on GPU,
/// keeping only embedding lookup and cross-entropy loss on CPU.
///
/// # Contract (C-GPUTRAIN-002)
///
/// - Exactly 3 PCIe transfers per training step
/// - Graceful fallback to CPU `TransformerTrainer` on any CUDA failure
/// - Weight sync via `sync_weights_to_cpu()` before save
#[cfg(feature = "cuda")]
pub struct CudaTransformerTrainer {
    /// CPU model (for embedding, saving, fallback)
    model: Transformer,
    /// CUDA device context
    cuda_trainer: CudaTrainer,
    /// GPU-resident transformer blocks (fp32 or NF4 via CudaBlock enum)
    cuda_blocks: Vec<CudaBlock>,
    /// Shared gradient workspace (one set, reused across layers; fp32 path only)
    cuda_grad_workspace: CudaGradWorkspace,
    /// ENT-263: Shared scratch for NF4 blocks (C-SCRATCH-001). None when fp32.
    nf4_shared_scratch: Option<CudaBlockScratch>,
    /// ENT-263: Shared LoRA gradient workspace for NF4 backward. None when fp32.
    nf4_lora_grad_workspace: Option<CudaLoraGradWorkspace>,
    /// ENT-263: Per-block LoRA optimizer states for NF4. None when fp32.
    nf4_lora_optimizer_states: Option<Vec<GpuLoraOptimizerState>>,
    /// GPU training state (layer inputs, grad bufs, optimizer states)
    gpu_training: GpuPretrainState,
    /// LM head weight on GPU [vocab_size * hidden_size]
    lm_head_weight_gpu: GpuBuffer<f32>,
    /// LM head weight gradient on GPU [vocab_size * hidden_size]
    lm_head_grad_gpu: GpuBuffer<f32>,
    /// LM head AdamW first moment [vocab_size * hidden_size]
    lm_head_m: GpuBuffer<f32>,
    /// LM head AdamW second moment [vocab_size * hidden_size]
    lm_head_v: GpuBuffer<f32>,
    /// Final norm weight AdamW first moment [hidden_size]
    final_norm_m: GpuBuffer<f32>,
    /// Final norm weight AdamW second moment [hidden_size]
    final_norm_v: GpuBuffer<f32>,
    /// CPU optimizer for embedding weights only
    embed_optimizer: AdamW,
    /// Training configuration
    config: TransformerTrainConfig,
    /// Metrics tracker
    pub metrics: MetricsTracker,
    /// Current optimizer step
    step: usize,
    /// Accumulated loss (for gradient accumulation)
    accumulated_loss: f32,
    /// Accumulated batch count
    accumulated_batches: usize,
    /// R-004: Last observed LM head gradient L2 norm (proxy for global grad norm)
    last_grad_norm: f32,
    /// R-040: Last observed embedding activation gradient L2 norm
    last_embed_grad_norm: f32,
    /// R-038: Per-block gradient accumulation for true multi-step gradient accumulation.
    /// Only allocated when accumulation_steps > 1. CPU-side buffers (~335 MB for 350M).
    grad_accum: Option<super::grad_accumulator::PerBlockGradientAccumulator>,
    /// ALB-091: GPU-resident gradient accumulation (replaces CPU accum when available).
    /// Eliminates 24 × ga stream.synchronize() + D2H transfers per optimizer step.
    gpu_grad_accum: Option<super::gpu_grad_accumulator::GpuGradientAccumulator>,
    /// R-002: Gradient scaler for mixed-precision training.
    /// For BF16: no-op (scale=1.0, dynamic=false).
    /// For FP16: dynamic loss scaling to prevent gradient underflow.
    grad_scaler: GradScaler,
    /// KAIZEN-047: Per-step wall-clock profiler.
    /// Reports timing breakdown for each training phase.
    profiler: StepProfiler,
    /// KAIZEN-053: Pre-allocated forward scratch buffers [max_seq_len * hidden_size].
    /// Reused every step — eliminates 2 × cuMemAlloc/Free per training step.
    fwd_scratch_a: GpuBuffer<f32>,
    fwd_scratch_b: GpuBuffer<f32>,
    /// KAIZEN-056: Pre-allocated CPU staging buffer for H2D hidden state upload.
    /// Eliminates vec![0.0; max_seq_len * hidden_size] allocation per step.
    h2d_staging: Vec<f32>,
    /// KAIZEN-059: Pre-allocated CPU staging buffer for D2H gradient downloads
    /// during gradient accumulation. Sized to max(h*intermediate, vocab*h).
    /// Eliminates ~15GB of per-step heap churn (36 × vec![0.0; h*i] + vec![0.0; vocab*h]
    /// per micro-batch × accumulation_steps).
    d2h_staging: Vec<f32>,
    /// ALB-078: Pre-allocated state for fused gradient clipping pipeline.
    /// Eliminates 24 stream.synchronize() calls per step.
    fused_clip: Option<FusedClipState>,
    /// Pre-allocated host zero buffer for zeroing final norm grad [hidden_size].
    /// BatchedRmsNormBackwardKernel accumulates grad_gamma via atomicAdd,
    /// so the buffer must be zeroed before each rms_norm_backward call.
    final_norm_zero_buf: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl CudaTransformerTrainer {
    /// Create a new GPU-resident trainer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if CUDA initialization, kernel pre-warming, or block upload fails.
    /// Caller should fall back to CPU `TransformerTrainer` on error.
    pub fn new(config: TransformerTrainConfig) -> crate::Result<Self> {
        let model = Transformer::new(&config.model_config);
        Self::with_model(model, config)
    }

    /// ALB-089: Load SafeTensors checkpoint for GPU inference (forward-only).
    ///
    /// Creates a `CudaTransformerTrainer` in inference mode. The optimizer
    /// state is allocated (wasteful but simple), but `forward_logits()` only
    /// uses the forward path. Call `forward_logits(&tokens)` to generate.
    ///
    /// # Arguments
    /// * `checkpoint_dir` - Directory containing model.safetensors + config.json
    /// * `model_config` - Transformer architecture config
    ///
    /// # Errors
    ///
    /// Returns `Err` if SafeTensors loading or CUDA initialization fails.
    pub fn for_inference(
        checkpoint_dir: impl AsRef<std::path::Path>,
        model_config: crate::transformer::TransformerConfig,
    ) -> crate::Result<Self> {
        let dir = checkpoint_dir.as_ref();

        // ALB-089: Try APR format first (our native checkpoint format), then SafeTensors
        let model = if let Some((Some(m), _step)) =
            crate::config::try_load_apr_for_inference(dir, &model_config)
        {
            m
        } else {
            Transformer::from_safetensors(dir, &model_config)?
        };

        let mut config = TransformerTrainConfig::new(model_config);
        config.max_seq_len = config.model_config.max_position_embeddings;
        Self::with_model(model, config)
    }

    /// Create a GPU-resident trainer from an existing model.
    ///
    /// # Errors
    ///
    /// Returns `Err` if CUDA initialization fails.
    pub fn with_model(model: Transformer, config: TransformerTrainConfig) -> crate::Result<Self> {
        if !cuda_training_available() {
            return Err(crate::error::Error::ConfigError("CUDA not available".into()));
        }

        let mc = &config.model_config;
        let max_seq_len = config.max_seq_len;
        let hidden_size = mc.hidden_size;
        let vocab_size = mc.vocab_size;
        let num_layers = mc.num_hidden_layers;

        // Step 1: Create CUDA trainer (initializes kernel caches)
        let cuda_trainer = CudaTrainer::new().map_err(|e| {
            crate::error::Error::ConfigError(format!("CUDA trainer init failed: {e:?}"))
        })?;

        println!(
            "  GPU: {} ({:.1} GB)",
            cuda_trainer.device_name(),
            cuda_trainer.total_memory() as f64 / 1e9
        );

        let ctx = cuda_trainer.context().clone();
        let stream = cuda_trainer.stream();

        // Step 2: Pre-warm forward kernels (C-PREWARM-001)
        // Must happen before block upload — JIT compilation needs free VRAM
        pre_warm_forward_kernels(
            hidden_size,
            mc.intermediate_size,
            mc.num_attention_heads,
            mc.num_kv_heads,
            mc.head_dim(),
            max_seq_len,
        )
        .map_err(|e| crate::error::Error::ConfigError(format!("Kernel pre-warm failed: {e:?}")))?;

        // Step 2a: Pre-warm backward kernels (trueno#200)
        // MUST happen before any GPU work — Blackwell's cuModuleLoadData fails
        // with ILLEGAL_ADDRESS when called during active GPU computation.
        {
            use crate::autograd::cuda_backward::pre_warm_lora_backward_kernels;
            let head_dim = mc.head_dim();
            pre_warm_lora_backward_kernels(
                hidden_size,
                mc.num_attention_heads * head_dim,
                mc.num_kv_heads * head_dim,
                max_seq_len,
                config.lora_rank.unwrap_or(0),
                mc.intermediate_size,
                mc.num_attention_heads,
                config.quantize_nf4 && config.is_lora(),
            )
            .map_err(|e| {
                crate::error::Error::ConfigError(format!("Backward kernel pre-warm failed: {e:?}"))
            })?;
            eprintln!("  ✓ Backward kernels pre-warmed (silu_backward, rms_norm_backward, etc.)");
        }

        // Step 2b: Bind cuBLAS handles to training stream (ALB-075)
        // Must happen after kernel cache init, before any GEMM calls.
        if let Err(e) = crate::autograd::cuda_forward::set_forward_cublas_stream(stream) {
            println!("[WARN] cuBLAS forward stream bind failed: {e:?} — falling back to PTX");
        }
        if let Err(e) = crate::autograd::cuda_backward::set_backward_cublas_stream(stream) {
            println!("[WARN] cuBLAS backward stream bind failed: {e:?} — falling back to PTX");
        }

        // Step 3: Upload transformer blocks to GPU
        let use_nf4 = config.quantize_nf4 && config.is_lora();
        let cuda_blocks = Self::upload_blocks(
            &model,
            mc,
            &config,
            &ctx,
            use_nf4,
            num_layers,
            hidden_size,
            max_seq_len,
        )?;

        // Step 4: Allocate shared gradient workspace
        let cuda_grad_workspace = CudaGradWorkspace::new(&ctx, mc).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad workspace alloc failed: {e:?}"))
        })?;

        // Step 5: Allocate GPU training state
        let buf_size = max_seq_len * hidden_size;
        let logits_size = max_seq_len * vocab_size;

        // Activation checkpointing: determine which layers save their inputs.
        // Checkpoint boundary layers (every segment_size layers) are always saved.
        // Non-boundary layers are recomputed from the nearest checkpoint during backward.
        let checkpointing = config.checkpoint_config.enabled;
        let segment_size = if checkpointing {
            let ns = config.checkpoint_config.num_segments.max(1);
            num_layers.div_ceil(ns)
        } else {
            1 // Every layer is a checkpoint (no recomputation)
        };
        let saved_layer_mask: Vec<bool> =
            (0..num_layers).map(|i| !checkpointing || i % segment_size == 0).collect();

        let mut layer_inputs = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layer_inputs.push(GpuBuffer::new(&ctx, buf_size).map_err(|e| {
                crate::error::Error::ConfigError(format!("Layer input alloc failed: {e:?}"))
            })?);
        }

        // Allocate recompute buffer if checkpointing is enabled
        let recompute_buf = if checkpointing {
            Some(GpuBuffer::new(&ctx, buf_size).map_err(|e| {
                crate::error::Error::ConfigError(format!("Recompute buf alloc failed: {e:?}"))
            })?)
        } else {
            None
        };

        if checkpointing {
            let saved_count = saved_layer_mask.iter().filter(|&&x| x).count();
            println!(
                "  ✓ Activation checkpointing: {} segments, saving {}/{} layer inputs",
                config.checkpoint_config.num_segments, saved_count, num_layers
            );
        }

        // Upload final RMSNorm weight
        let norm_slice = model.norm.weight.data().as_slice().expect("contiguous");
        let final_norm_weight = GpuBuffer::from_host(&ctx, norm_slice).map_err(|e| {
            crate::error::Error::ConfigError(format!("Norm weight upload failed: {e:?}"))
        })?;

        let blocks_output = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Blocks output alloc failed: {e:?}"))
        })?;
        let grad_buf_a = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad buf A alloc failed: {e:?}"))
        })?;
        let grad_buf_b = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad buf B alloc failed: {e:?}"))
        })?;
        let grad_final_norm_weight = GpuBuffer::new(&ctx, hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Grad norm alloc failed: {e:?}"))
        })?;
        let norm_output = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Norm output alloc failed: {e:?}"))
        })?;
        let logits_buf = GpuBuffer::new(&ctx, logits_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Logits buf alloc failed: {e:?}"))
        })?;
        let lm_head_grad_hidden = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head grad alloc failed: {e:?}"))
        })?;

        // Initialize per-block optimizer states (fp32 path only; NF4 uses LoRA states)
        let mut optimizer_states = Vec::new();
        if !use_nf4 {
            optimizer_states.reserve(num_layers);
            for (i, block) in cuda_blocks.iter().enumerate() {
                optimizer_states.push(block.init_optimizer_state().map_err(|e| {
                    crate::error::Error::ConfigError(format!("Block {i} opt state failed: {e:?}"))
                })?);
            }
        }

        let gpu_training = GpuPretrainState {
            layer_inputs,
            saved_layer_mask,
            recompute_buf,
            final_norm_weight,
            blocks_output,
            grad_buf_a,
            grad_buf_b,
            grad_final_norm_weight,
            norm_output,
            logits_buf,
            lm_head_grad_hidden,
            optimizer_states,
            step: 0,
        };

        // Step 6: Upload LM head weight to GPU
        // Use tied weights (embed_tokens.weight) or separate lm_head
        let lm_head_data = model.lm_head.as_ref().unwrap_or(&model.embed_tokens.weight).data();
        let lm_head_slice = lm_head_data.as_slice().expect("contiguous");
        let lm_head_weight_gpu = GpuBuffer::from_host(&ctx, lm_head_slice).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head upload failed: {e:?}"))
        })?;
        let lm_head_grad_gpu = GpuBuffer::new(&ctx, vocab_size * hidden_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("LM head grad alloc failed: {e:?}"))
        })?;
        // CRITICAL: Must zero-initialize m/v buffers. GpuBuffer::new() does NOT
        // zero memory (cuMemAlloc returns uninitialized VRAM).
        let lm_head_m = GpuBuffer::from_host(&ctx, &vec![0.0f32; vocab_size * hidden_size])
            .map_err(|e| {
                crate::error::Error::ConfigError(format!("LM head m alloc failed: {e:?}"))
            })?;
        let lm_head_v = GpuBuffer::from_host(&ctx, &vec![0.0f32; vocab_size * hidden_size])
            .map_err(|e| {
                crate::error::Error::ConfigError(format!("LM head v alloc failed: {e:?}"))
            })?;

        // Final norm optimizer states
        let final_norm_m = GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("Final norm m alloc failed: {e:?}"))
        })?;
        let final_norm_v = GpuBuffer::from_host(&ctx, &vec![0.0f32; hidden_size]).map_err(|e| {
            crate::error::Error::ConfigError(format!("Final norm v alloc failed: {e:?}"))
        })?;

        // KAIZEN-053: Pre-allocate forward scratch buffers (reused every step)
        let buf_size = max_seq_len * hidden_size;
        let fwd_scratch_a = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Fwd scratch A alloc failed: {e:?}"))
        })?;
        let fwd_scratch_b = GpuBuffer::new(&ctx, buf_size).map_err(|e| {
            crate::error::Error::ConfigError(format!("Fwd scratch B alloc failed: {e:?}"))
        })?;

        // Sync to ensure all uploads completed
        stream
            .synchronize()
            .map_err(|e| crate::error::Error::ConfigError(format!("Stream sync failed: {e:?}")))?;

        println!(
            "  ✓ GPU training state allocated (LM head: {:.1} MB)",
            (vocab_size * hidden_size * 4) as f64 / 1e6
        );

        // ENT-263: Allocate NF4 infrastructure (shared scratch, LoRA grad workspace, optimizer states)
        let (nf4_shared_scratch, nf4_lora_grad_workspace, nf4_lora_optimizer_states) = if use_nf4 {
            let lora_rank = config.lora_rank.unwrap_or(16);

            // C-SCRATCH-001: Shared scratch for NF4 blocks (reused across all layers)
            let scratch = CudaBlockScratch::new(mc, max_seq_len, &ctx, lora_rank).map_err(|e| {
                crate::error::Error::ConfigError(format!("NF4 shared scratch alloc failed: {e:?}"))
            })?;

            // LoRA gradient workspace (shared, reused per-block like CudaGradWorkspace)
            let grad_ws = CudaLoraGradWorkspace::new(&ctx, mc, lora_rank).map_err(|e| {
                crate::error::Error::ConfigError(format!(
                    "NF4 LoRA grad workspace alloc failed: {e:?}"
                ))
            })?;

            // Per-block LoRA optimizer states
            let mut lora_opt_states = Vec::with_capacity(num_layers);
            for (i, block) in cuda_blocks.iter().enumerate() {
                lora_opt_states.push(block.init_lora_optimizer_state().map_err(|e| {
                    crate::error::Error::ConfigError(format!(
                        "Block {i} LoRA opt state failed: {e:?}"
                    ))
                })?);
            }

            println!(
                "  ✓ NF4 training infrastructure allocated (shared scratch + LoRA optimizer × {num_layers})"
            );
            (Some(scratch), Some(grad_ws), Some(lora_opt_states))
        } else {
            (None, None, None)
        };

        // KAIZEN-050: loss_fn removed — cross-entropy computed by fused GPU kernel
        // C-EMBED-GRAD-001: CPU optimizer must match YAML hyperparams (not defaults)
        let embed_optimizer =
            AdamW::new(config.lr, config.beta1, config.beta2, 1e-8, config.weight_decay);

        // R-038: Allocate per-block gradient accumulation buffers (CPU-side)
        // when accumulation_steps > 1 for true gradient accumulation.
        let grad_accum = if config.accumulation_steps > 1 {
            let kv_hidden = mc.num_kv_heads * mc.head_dim();
            let block_sizes =
                super::grad_accumulator::PerBlockGradientAccumulator::compute_block_sizes(
                    hidden_size,
                    kv_hidden,
                    mc.intermediate_size,
                );
            let accum = super::grad_accumulator::PerBlockGradientAccumulator::new(
                num_layers,
                block_sizes,
                vocab_size,
                hidden_size,
            );
            println!(
                "  ✓ Gradient accumulation: {} steps, CPU buffers ({:.1} MB)",
                config.accumulation_steps,
                (accum
                    .block_grads
                    .iter()
                    .map(super::grad_accumulator::BlockGradientSet::total_elements)
                    .sum::<usize>()
                    + accum.lm_head_grad.len()
                    + accum.final_norm_grad.len()
                    + accum.embedding_grad.len()) as f64
                    * 4.0
                    / 1e6,
            );
            Some(accum)
        } else {
            None
        };

        // ALB-091: GPU-resident gradient accumulation (eliminates D2H bottleneck).
        // Falls back to CPU accum if GPU allocation fails.
        let gpu_grad_accum = if config.accumulation_steps > 1 {
            match super::gpu_grad_accumulator::GpuGradientAccumulator::new(&ctx, mc) {
                Ok(accum) => {
                    println!("  ✓ GPU gradient accumulation enabled (ALB-091)");
                    Some(accum)
                }
                Err(e) => {
                    eprintln!(
                        "  [WARN] GPU gradient accumulation failed ({e}), using CPU fallback"
                    );
                    None
                }
            }
        } else {
            None
        };

        // KAIZEN-059: Pre-allocate D2H staging buffer for gradient accumulation
        // downloads. Only needed when GPU accum is unavailable (CPU fallback path).
        let d2h_staging = if config.accumulation_steps > 1 && gpu_grad_accum.is_none() {
            let ws_max = hidden_size * mc.intermediate_size;
            let lm_max = vocab_size * hidden_size;
            vec![0.0f32; ws_max.max(lm_max)]
        } else {
            Vec::new()
        };

        // ALB-078: Pre-allocate fused gradient clipping state.
        // Eliminates 24 stream syncs per step by keeping norm+clip on GPU.
        let kv_hidden = mc.num_kv_heads * mc.head_dim();
        let fused_clip = Self::init_fused_clip(&ctx, &config, hidden_size, kv_hidden, mc);

        // R-002: Initialize gradient scaler from precision config
        let grad_scaler = GradScaler::from_config(&config.precision_config);
        if config.precision_config.is_mixed() {
            println!(
                "  ✓ Mixed precision: {} (loss scale={}, dynamic={})",
                config.precision_config.compute_precision,
                grad_scaler.scale(),
                grad_scaler.is_dynamic(),
            );
        }

        Ok(Self {
            model,
            cuda_trainer,
            cuda_blocks,
            cuda_grad_workspace,
            nf4_shared_scratch,
            nf4_lora_grad_workspace,
            nf4_lora_optimizer_states,
            gpu_training,
            lm_head_weight_gpu,
            lm_head_grad_gpu,
            lm_head_m,
            lm_head_v,
            final_norm_m,
            final_norm_v,
            embed_optimizer,
            // KAIZEN-047: Read profile_interval before moving config into struct.
            profiler: if config.profile_interval > 0 {
                StepProfiler::new(true, config.profile_interval)
            } else {
                StepProfiler::disabled()
            },
            config,
            metrics: MetricsTracker::new(),
            step: 0,
            accumulated_loss: 0.0,
            accumulated_batches: 0,
            last_grad_norm: 0.0,
            last_embed_grad_norm: 0.0,
            grad_accum,
            gpu_grad_accum,
            grad_scaler,
            fwd_scratch_a,
            fwd_scratch_b,
            h2d_staging: vec![0.0f32; max_seq_len * hidden_size],
            d2h_staging,
            fused_clip,
            final_norm_zero_buf: vec![0.0f32; hidden_size],
        })
    }

    /// Upload transformer blocks to GPU (NF4 or fp32 path).
    #[allow(clippy::too_many_arguments)]
    fn upload_blocks(
        model: &Transformer,
        mc: &crate::transformer::TransformerConfig,
        config: &TransformerTrainConfig,
        ctx: &std::sync::Arc<trueno_gpu::driver::CudaContext>,
        use_nf4: bool,
        num_layers: usize,
        hidden_size: usize,
        max_seq_len: usize,
    ) -> crate::Result<Vec<CudaBlock>> {
        let mut cuda_blocks: Vec<CudaBlock> = Vec::with_capacity(num_layers);

        if use_nf4 {
            let lora_rank = config.lora_rank.unwrap_or(16);
            let lora_alpha = config.lora_alpha.unwrap_or(2.0 * lora_rank as f32);
            let lora_scale = lora_alpha / lora_rank as f32;
            let head_dim = mc.head_dim();
            let q_dim = mc.num_attention_heads * head_dim;
            let kv_hidden = mc.num_kv_heads * head_dim;

            for (i, layer) in model.layers.iter().enumerate() {
                let lora_a_q: Vec<f32> = (0..hidden_size * lora_rank)
                    .map(|j| ((j as f32 + i as f32 * 1000.0) * 0.1).sin() * 0.01)
                    .collect();
                let lora_b_q = vec![0.0f32; lora_rank * q_dim];
                let lora_a_v: Vec<f32> = (0..hidden_size * lora_rank)
                    .map(|j| ((j as f32 + i as f32 * 2000.0 + 500.0) * 0.1).sin() * 0.01)
                    .collect();
                let lora_b_v = vec![0.0f32; lora_rank * kv_hidden];

                let q_norm_data = layer
                    .self_attn
                    .q_norm
                    .as_ref()
                    .map(|t| t.data().as_slice().expect("contiguous q_norm").to_vec());
                let k_norm_data = layer
                    .self_attn
                    .k_norm
                    .as_ref()
                    .map(|t| t.data().as_slice().expect("contiguous k_norm").to_vec());

                let block = crate::transformer::CudaNf4TransformerBlock::new(
                    mc,
                    i,
                    ctx.clone(),
                    layer.input_norm.weight.data().as_slice().expect("contiguous"),
                    layer.post_attn_norm.weight.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_q.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_k.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_v.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_o.data().as_slice().expect("contiguous"),
                    layer.ffn.w_gate.data().as_slice().expect("contiguous"),
                    layer.ffn.w_up.data().as_slice().expect("contiguous"),
                    layer.ffn.w_down.data().as_slice().expect("contiguous"),
                    max_seq_len,
                    Some((&lora_a_q, &lora_b_q)),
                    Some((&lora_a_v, &lora_b_v)),
                    lora_scale,
                    lora_rank,
                    q_norm_data.as_deref(),
                    k_norm_data.as_deref(),
                )
                .map_err(|e| {
                    crate::error::Error::ConfigError(format!("NF4 block {i} upload failed: {e:?}"))
                })?;
                cuda_blocks.push(CudaBlock::Nf4(block));
            }
            println!("  ✓ {num_layers} NF4 transformer blocks uploaded (LoRA rank={lora_rank}, alpha={lora_alpha})");
        } else {
            for (i, layer) in model.layers.iter().enumerate() {
                let block = CudaTransformerBlock::new(
                    mc,
                    i,
                    ctx.clone(),
                    layer.input_norm.weight.data().as_slice().expect("contiguous"),
                    layer.post_attn_norm.weight.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_q.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_k.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_v.data().as_slice().expect("contiguous"),
                    layer.self_attn.w_o.data().as_slice().expect("contiguous"),
                    layer.ffn.w_gate.data().as_slice().expect("contiguous"),
                    layer.ffn.w_up.data().as_slice().expect("contiguous"),
                    layer.ffn.w_down.data().as_slice().expect("contiguous"),
                    max_seq_len,
                )
                .map_err(|e| {
                    crate::error::Error::ConfigError(format!("Block {i} upload failed: {e:?}"))
                })?;
                cuda_blocks.push(CudaBlock::Fp32(block));
            }
            println!("  ✓ {num_layers} transformer blocks uploaded to GPU");
        }

        Ok(cuda_blocks)
    }

    /// ALB-078: Initialize fused gradient clipping state (extracted for complexity).
    fn init_fused_clip(
        ctx: &std::sync::Arc<trueno_gpu::driver::CudaContext>,
        config: &TransformerTrainConfig,
        hidden_size: usize,
        kv_hidden: usize,
        mc: &crate::transformer::TransformerConfig,
    ) -> Option<FusedClipState> {
        config.base.max_grad_norm?;
        let grad_sizes: [u32; 9] = [
            (hidden_size * hidden_size) as u32,
            (hidden_size * kv_hidden) as u32,
            (hidden_size * kv_hidden) as u32,
            (hidden_size * hidden_size) as u32,
            (hidden_size * mc.intermediate_size) as u32,
            (hidden_size * mc.intermediate_size) as u32,
            (mc.intermediate_size * hidden_size) as u32,
            hidden_size as u32,
            hidden_size as u32,
        ];
        match FusedClipState::new(ctx, &grad_sizes) {
            Ok(state) => {
                println!(
                    "  ✓ Fused gradient clipping: {} partials ({:.1} KB)",
                    state.total_partials,
                    f64::from(state.total_partials) * 4.0 / 1024.0,
                );
                Some(state)
            }
            Err(e) => {
                println!("  ⚠ Fused clip alloc failed ({e:?}), using sync fallback");
                None
            }
        }
    }

    /// Run one forward+backward step for a single sequence.
    ///
    /// # Contract (C-GPUSTEP-001)
    ///
    /// - Precondition: `input_ids.len() == target_ids.len() <= max_seq_len`
    /// - Postcondition: If `accumulate_only` is false, all GPU weights updated.
    ///   If true, gradients accumulated into CPU buffers (no weight updates).
    /// - Transfer count: 1 PCIe H2D + ~1KB control (KAIZEN-050, + 24×9 D2H if accumulating)
    fn train_step_single(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        accumulate_only: bool,
    ) -> Option<f32> {
        self.profiler.begin_step();
        let result = self.train_step_inner(input_ids, target_ids, accumulate_only);
        self.profiler.finish_step();
        result
    }

    /// Inner training step — separated so profiler always records the step.
    fn train_step_inner(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        accumulate_only: bool,
    ) -> Option<f32> {
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;

        // Truncate to max_seq_len — GPU buffers are pre-allocated for this size
        let max_sl = self.config.max_seq_len;
        let input_ids = if input_ids.len() > max_sl { &input_ids[..max_sl] } else { input_ids };
        let target_ids = if target_ids.len() > max_sl { &target_ids[..max_sl] } else { target_ids };
        let seq_len = input_ids.len();

        // Steps 1-6: GPU forward pass — logits stay GPU-resident (KAIZEN-050)
        // (sub-phases embed, h2d, forward, norm_lm instrumented inside gpu_forward)
        if self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size).is_none() {
            eprintln!(
                "[train_step_inner] gpu_forward returned None (seq_len={seq_len}, \
                 hidden={hidden_size}, vocab={vocab_size}) — CUDA context likely poisoned"
            );
            return None;
        }

        // Step 7: Fused GPU cross-entropy loss + softmax backward (KAIZEN-050)
        // Eliminates: logits D2H (77.8MB) + CPU softmax (40ms) + grad H2D (77.8MB)
        self.profiler.begin(StepProfiler::LOSS);
        let stream = self.cuda_trainer.stream();

        // Compute combined scale: (1/seq_len) * (1/accum_steps)
        //
        // ALB-072: Do NOT multiply by grad_scaler.scale() here. All backward
        // computation uses f32 GpuBuffers — there is no fp16 gradient underflow
        // risk. The 65536x loss scaling caused gradient overflow in early layers
        // (blocks 0-1 went NaN). The GradScaler remains active for the CPU
        // embedding path (unscale_and_check in optimizer_step) as a safety check,
        // but it operates with scale=1.0 effective for GPU gradients.
        let mut loss_scale = 1.0 / seq_len as f32;
        if self.config.accumulation_steps > 1 {
            loss_scale /= self.config.accumulation_steps as f32;
        }

        // KAIZEN-052: In-place — gradient written directly to logits_buf.
        let loss_val = fused_cross_entropy_cuda(
            &mut self.gpu_training.logits_buf,
            target_ids,
            seq_len as u32,
            vocab_size as u32,
            loss_scale,
            stream,
        )
        .ok()?;

        // NaN guard (replaces logits NaN check — NaN logits → NaN loss via kernel)
        if !loss_val.is_finite() {
            return None;
        }
        self.profiler.end(StepProfiler::LOSS);

        // Steps 8-11: GPU backward pass (with or without optimizer)
        // (sub-phases lm_bwd, norm_bwd, blk_bwd instrumented inside gpu_backward)
        // KAIZEN-050: grad_logits on GPU. KAIZEN-052: grad lives in logits_buf (in-place).
        //
        // ENT-263 fix: Capture loss regardless of backward success. The NF4 backward
        // path may fail (e.g., gemm_nf4_backward_a stub) but the loss was already
        // computed by fused_cross_entropy_cuda. Dropping the loss silently causes
        // loss=0.0 reporting despite valid forward passes.
        if let Some(grad_output_is_a) =
            self.gpu_backward(seq_len, hidden_size, vocab_size, accumulate_only)
        {
            // Step 12: Embedding backward (CPU scatter-add always accumulates)
            self.profiler.begin(StepProfiler::EMBED_BWD);
            self.embed_backward(input_ids, seq_len, hidden_size, vocab_size, grad_output_is_a);

            self.profiler.end(StepProfiler::EMBED_BWD);
        }

        Some(loss_val)
    }

    /// GPU forward pass: embed → blocks → norm → LM head.
    ///
    /// Logits stay GPU-resident in `self.gpu_training.logits_buf` (KAIZEN-050).
    /// Transfers: 1 H2D (hidden states). No D2H — logits consumed by fused kernel.
    #[allow(unsafe_code)]
    fn gpu_forward(
        &mut self,
        input_ids: &[u32],
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<()> {
        contract_pre_gpu_forward!();
        let stream = self.cuda_trainer.stream();

        // Embedding lookup (CPU)
        self.profiler.begin(StepProfiler::EMBED);
        let hidden = self.model.embed_tokens.forward(input_ids);
        let hidden_slice = hidden.data().as_slice()?;
        self.profiler.end(StepProfiler::EMBED);

        // Upload hidden states to GPU (Transfer 1: H2D)
        // Pad to max_seq_len so D2D copies to pre-allocated layer_inputs match.
        // KAIZEN-053: Reuse pre-allocated scratch buffers instead of cuMemAlloc per step.
        // KAIZEN-056: Reuse pre-allocated h2d_staging instead of alloc per step.
        self.profiler.begin(StepProfiler::H2D);
        self.h2d_staging[..hidden_slice.len()].copy_from_slice(hidden_slice);
        self.h2d_staging[hidden_slice.len()..].fill(0.0);
        if let Err(e) = self.fwd_scratch_a.copy_from_host(&self.h2d_staging) {
            eprintln!("[gpu_forward] H2D copy failed: {e:?} — CUDA context may be poisoned");
            return None;
        }
        self.profiler.end(StepProfiler::H2D);

        // Forward through CUDA blocks using pre-allocated ping-pong buffers.
        // KAIZEN-053: fwd_scratch_a/b are top-level fields (not in gpu_training)
        // so borrowing them doesn't conflict with gpu_training.layer_inputs.
        self.profiler.begin(StepProfiler::FORWARD);
        let mut input_is_a = true; // Track which scratch buffer is "input"
        for (i, block) in self.cuda_blocks.iter_mut().enumerate() {
            // Use raw pointers for the ping-pong to avoid borrow conflicts
            // with self.gpu_training.layer_inputs
            let (input_ptr, output_ptr): (*const GpuBuffer<f32>, *mut GpuBuffer<f32>) =
                if input_is_a {
                    (
                        std::ptr::from_ref(&self.fwd_scratch_a),
                        std::ptr::from_mut(&mut self.fwd_scratch_b),
                    )
                } else {
                    (
                        std::ptr::from_ref(&self.fwd_scratch_b),
                        std::ptr::from_mut(&mut self.fwd_scratch_a),
                    )
                };
            if self.gpu_training.saved_layer_mask[i] {
                // SAFETY: Both buffers are valid GPU allocations with matching max_seq_len size.
                // Copy completes before block.forward() reads from input (same stream ordering).
                unsafe {
                    self.gpu_training.layer_inputs[i]
                        .copy_from_buffer_async(&*input_ptr, stream)
                        .ok()?;
                }
            }
            // SAFETY: input_ptr and output_ptr point to disjoint fwd_scratch_{a,b}.
            // ENT-263: Pass shared scratch for NF4 blocks (C-SCRATCH-001).
            self.profiler.begin_layer();
            unsafe {
                block
                    .forward(
                        &*input_ptr,
                        &mut *output_ptr,
                        seq_len,
                        stream,
                        self.nf4_shared_scratch.as_mut(),
                    )
                    .ok()?;
            }
            self.profiler.end_layer_fwd(i);
            input_is_a = !input_is_a;
        }
        self.profiler.end(StepProfiler::FORWARD);

        // After the loop, input_is_a tells us which buffer has the final output
        let final_output: &GpuBuffer<f32> =
            if input_is_a { &self.fwd_scratch_a } else { &self.fwd_scratch_b };

        // Save blocks output for final norm backward
        // SAFETY: Disjoint GPU buffers with matching max_seq_len sizes.
        self.profiler.begin(StepProfiler::NORM_LM);
        unsafe {
            self.gpu_training.blocks_output.copy_from_buffer_async(final_output, stream).ok()?;
        }

        // Final RMSNorm forward (GPU)
        rms_norm_forward(
            final_output,
            &self.gpu_training.final_norm_weight,
            &mut self.gpu_training.norm_output,
            seq_len as u32,
            hidden_size as u32,
            stream,
        )
        .ok()?;

        // LM head GEMM forward (GPU)
        // gemm_forward treats flat (V,H) memory as (H,V) row-major, which
        // implicitly transposes — matching the CPU matmul's tied-weight behavior.
        gemm_forward(
            &self.gpu_training.norm_output,
            &self.lm_head_weight_gpu,
            &mut self.gpu_training.logits_buf,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .ok()?;

        // KAIZEN-050: Logits stay GPU-resident — no D2H transfer.
        // Fused cross-entropy kernel reads logits_buf directly on GPU.
        self.profiler.end(StepProfiler::NORM_LM);

        Some(())
    }

    /// ALB-089: Forward-only pass that returns last-position logits on CPU.
    ///
    /// Runs the same GPU forward as training but downloads only the last
    /// position's logits (vocab_size floats) for token sampling. No backward
    /// pass, no loss computation.
    ///
    /// # Contract (C-CUDA-INF-001)
    ///
    /// - Same forward path as `gpu_forward()` — identical logits
    /// - Only downloads `logits[seq_len-1, :]` (128 KB for 32K vocab)
    /// - stream.synchronize() before D2H (C-STREAMSYNC-001)
    pub fn forward_logits(&mut self, input_ids: &[u32]) -> Option<Vec<f32>> {
        let seq_len = input_ids.len();
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;

        if seq_len == 0 || seq_len > self.config.max_seq_len {
            return None;
        }

        // Reuse gpu_forward for the actual computation
        self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size)?;

        // C-STREAMSYNC-001: synchronize before D2H
        let stream = self.cuda_trainer.stream();
        stream.synchronize().ok()?;

        // Download last position logits only: logits_buf[seq_len-1, :]
        let offset = (seq_len - 1) * vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        self.gpu_training.logits_buf.copy_to_host_at(&mut logits, offset).ok()?;

        Some(logits)
    }

    /// GPU backward pass with interleaved per-block optimizer step.
    ///
    /// Each block's backward writes weight gradients to the shared `CudaGradWorkspace`.
    /// Recompute layer inputs for a segment during backward (activation checkpointing).
    ///
    /// When checkpointing is enabled, non-checkpoint layers don't save their inputs
    /// during forward. Before their backward pass, we recompute from the nearest
    /// checkpoint by re-running forward through intermediate blocks.
    ///
    /// This recomputes the entire segment [checkpoint..=target_layer], storing
    /// intermediate layer_inputs so subsequent layers in the same segment don't
    /// need redundant recomputation.
    ///
    /// # Contract (R-021)
    ///
    /// After this call, `layer_inputs[i]` is valid for all i in [checkpoint..=target_layer].
    #[allow(unsafe_code)]
    fn recompute_segment(
        gpu_training: &mut GpuPretrainState,
        cuda_blocks: &mut [CudaBlock],
        nf4_shared_scratch: &mut Option<CudaBlockScratch>,
        target_layer: usize,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Option<()> {
        // Find nearest saved checkpoint at or before target
        let seg_start = (0..=target_layer).rev().find(|&i| gpu_training.saved_layer_mask[i])?;

        if seg_start == target_layer {
            return Some(()); // Already saved
        }

        // Copy checkpoint input to recompute_buf as starting point.
        // SAFETY: recompute_buf and layer_inputs are disjoint allocations.
        let recompute_buf = gpu_training.recompute_buf.as_mut()?;
        unsafe {
            recompute_buf
                .copy_from_buffer_async(&gpu_training.layer_inputs[seg_start], stream)
                .ok()?;
        }

        // Forward through blocks [seg_start..target_layer], saving intermediate inputs.
        // For block i, input → block i → output becomes input for block i+1.
        // We save output (= input to block i+1) in layer_inputs[i+1].
        //
        // Buffer pattern:
        //   i == seg_start: input = recompute_buf, output = layer_inputs[seg_start+1]
        //   i > seg_start:  input = layer_inputs[i], output = layer_inputs[i+1]
        //
        // SAFETY: split_at_mut ensures non-overlapping borrows of layer_inputs.
        // recompute_buf is separate from layer_inputs.
        for i in seg_start..target_layer {
            if i == seg_start {
                // Input is in recompute_buf, output goes to layer_inputs[i+1]
                let recompute_ptr: *const GpuBuffer<f32> = recompute_buf;
                let li = &mut gpu_training.layer_inputs;
                unsafe {
                    cuda_blocks[i]
                        .forward(
                            &*recompute_ptr,
                            &mut li[i + 1],
                            seq_len,
                            stream,
                            nf4_shared_scratch.as_mut(),
                        )
                        .ok()?;
                }
            } else {
                // Input = layer_inputs[i], output = layer_inputs[i+1]
                let li = &mut gpu_training.layer_inputs;
                let (left, right) = li.split_at_mut(i + 1);
                cuda_blocks[i]
                    .forward(&left[i], &mut right[0], seq_len, stream, nf4_shared_scratch.as_mut())
                    .ok()?;
            }
        }

        Some(())
    }

    /// Since `gemm_backward_b` overwrites (not accumulates), we must run each block's
    /// optimizer step immediately after its backward, before the next block overwrites
    /// the workspace. This also enables per-block gradient clipping.
    ///
    /// When `accumulate_only` is true (R-038 gradient accumulation), the per-block
    /// optimizer steps are skipped and workspace gradients are downloaded to CPU-side
    /// `PerBlockGradientAccumulator` instead. LM head and final norm gradients are
    /// also downloaded and accumulated. The optimizer step is deferred until
    /// `gpu_optimizer_from_accum()` is called.
    ///
    /// Returns `grad_output_is_a` flag for embedding backward.
    /// Transfer: 0 H2D (KAIZEN-050/052: grad in logits_buf) + 24×9 D2H if accumulating.
    #[allow(unsafe_code)]
    fn gpu_backward(
        &mut self,
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        accumulate_only: bool,
    ) -> Option<bool> {
        let stream = self.cuda_trainer.stream();
        let max_grad_norm = self.config.base.max_grad_norm;
        let lr = self.current_lr();
        // ALB-072: No inv_scale needed — loss_scale no longer includes grad_scaler.
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let weight_decay = self.config.weight_decay;

        // KAIZEN-050: grad_logits GPU-resident. KAIZEN-052: grad lives in logits_buf (in-place).
        // No separate grad buffer. No GRAD_H2D transfer.

        // LM head GEMM backward
        self.profiler.begin(StepProfiler::LM_BWD);
        gemm_backward_a(
            &self.gpu_training.logits_buf,
            &self.lm_head_weight_gpu,
            &mut self.gpu_training.lm_head_grad_hidden,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .ok()?;

        gemm_backward_b(
            &self.gpu_training.norm_output,
            &self.gpu_training.logits_buf,
            &mut self.lm_head_grad_gpu,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .ok()?;

        // Clip LM head weight gradient
        // KAIZEN-049: GPU norm reduction.
        // KAIZEN-051: No explicit sync needed — same stream ordering.
        // ALB-071: Always compute LM head grad norm for observability (R-004).
        // C-CLIP-001: squared_sum_cuda returns ||g||². Take sqrt for L2 norm (entrenar#311).
        let lm_sq_norm =
            squared_sum_cuda(&self.lm_head_grad_gpu, self.lm_head_grad_gpu.len() as u32, stream)
                .unwrap_or(0.0);
        let lm_norm = lm_sq_norm.sqrt(); // L2 norm, NOT squared
        self.last_grad_norm = lm_norm; // R-004: capture for observability
                                       // C-BACKPARITY-001: LM head gradient norm tracing (pre-clip).
        if std::env::var("ENTRENAR_TRACE_GRADIENTS").is_ok() {
            eprintln!("[grad-trace] lm_head gnorm={lm_norm:.6}");
            // Also trace the grad_hidden flowing to blocks
            let gh_sq = squared_sum_cuda(
                &self.gpu_training.lm_head_grad_hidden,
                self.gpu_training.lm_head_grad_hidden.len() as u32,
                stream,
            )
            .unwrap_or(0.0);
            eprintln!("[grad-trace] lm_head_grad_hidden gnorm={:.6}", gh_sq.sqrt());
        }
        if let Some(max_norm) = max_grad_norm {
            let clip_scale = if lm_norm > max_norm { max_norm / lm_norm } else { 1.0 };
            let n = self.lm_head_grad_gpu.len() as u32;
            let _ = gradient_clip_cuda(&mut self.lm_head_grad_gpu, clip_scale, n, stream);
        }
        self.profiler.end(StepProfiler::LM_BWD);

        // Final RMSNorm backward
        self.profiler.begin(StepProfiler::NORM_BWD);
        // Zero grad_final_norm_weight before backward — kernel accumulates via atomicAdd
        self.gpu_training.grad_final_norm_weight.copy_from_host(&self.final_norm_zero_buf).ok()?;
        rms_norm_backward(
            &self.gpu_training.blocks_output,
            &self.gpu_training.final_norm_weight,
            &self.gpu_training.lm_head_grad_hidden,
            &mut self.gpu_training.grad_buf_a,
            &mut self.gpu_training.grad_final_norm_weight,
            seq_len as u32,
            hidden_size as u32,
            1e-5_f32,
            stream,
        )
        .ok()?;

        // Clip final norm weight gradient
        // KAIZEN-051: No explicit sync needed — same stream ordering as LM head clip.
        if let Some(max_norm) = max_grad_norm {
            let (scale, _) = Self::compute_clip_scale_with_norm(
                &self.gpu_training.grad_final_norm_weight,
                max_norm,
                stream,
            );
            let n = self.gpu_training.grad_final_norm_weight.len() as u32;
            let _ =
                gradient_clip_cuda(&mut self.gpu_training.grad_final_norm_weight, scale, n, stream);
        }
        self.profiler.end(StepProfiler::NORM_BWD);

        // R-038: Either accumulate non-block grads or run non-block optimizer.
        if accumulate_only {
            // ALB-091: GPU-resident accumulation (no sync, no D2H) or CPU fallback.
            if let Some(ref mut gpu_accum) = self.gpu_grad_accum {
                let _ = gpu_accum.accumulate_nonblock(
                    &self.lm_head_grad_gpu,
                    &self.gpu_training.grad_final_norm_weight,
                    stream,
                );
            } else {
                stream.synchronize().ok()?;
                Self::download_nonblock_grads_to_accum(
                    &self.lm_head_grad_gpu,
                    &self.gpu_training.grad_final_norm_weight,
                    &mut self.grad_accum,
                    &mut self.d2h_staging,
                )?;
            }
        } else {
            Self::run_nonblock_optimizer_step(
                &mut self.gpu_training,
                Some(&mut self.lm_head_weight_gpu),
                &self.lm_head_grad_gpu,
                &mut self.lm_head_m,
                &mut self.lm_head_v,
                &mut self.final_norm_m,
                &mut self.final_norm_v,
                lr,
                beta1,
                beta2,
                weight_decay,
                stream,
            );
        }

        // Backward through blocks in reverse, with interleaved clip + optimizer.
        // Each block's backward writes weight gradients to shared CudaGradWorkspace.
        //
        // SAFETY: grad_buf_a and grad_buf_b are disjoint fields. Raw pointers
        // allow alternating read/write without violating aliasing rules.
        self.profiler.begin(StepProfiler::BLK_BWD);
        let grad_a_ptr: *mut GpuBuffer<f32> = &raw mut self.gpu_training.grad_buf_a;
        let grad_b_ptr: *mut GpuBuffer<f32> = &raw mut self.gpu_training.grad_buf_b;
        let mut grad_output_is_a = true;
        let use_nf4 = self.config.quantize_nf4 && self.config.is_lora();

        for layer_idx in (0..self.cuda_blocks.len()).rev() {
            // Activation checkpointing: if this layer's input wasn't saved during
            // forward, recompute the segment from the nearest checkpoint.
            if !self.gpu_training.saved_layer_mask[layer_idx] {
                Self::recompute_segment(
                    &mut self.gpu_training,
                    &mut self.cuda_blocks,
                    &mut self.nf4_shared_scratch,
                    layer_idx,
                    seq_len,
                    stream,
                )?;
            }

            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };

            self.profiler.begin_layer();
            if use_nf4 {
                // ENT-263: NF4 backward — LoRA gradient computation
                // Uses backward_nf4() which computes gradients for LoRA weights and norms only.
                // output_scratch reuses grad_buf_a/b as temporary storage for recomputed forward.
                let _output_scratch_ptr: *mut GpuBuffer<f32> = if grad_output_is_a {
                    grad_b_ptr // grad_input is in b, use as output_scratch too (will be overwritten)
                } else {
                    grad_a_ptr
                };
                // We need a separate output_scratch. Reuse blocks_output as scratch since
                // it was already consumed for norm backward above.
                match self.cuda_blocks[layer_idx].backward_nf4(
                    &self.gpu_training.layer_inputs[layer_idx],
                    grad_output,
                    grad_input,
                    &mut self.gpu_training.blocks_output, // reuse as output_scratch
                    seq_len,
                    stream,
                    self.nf4_shared_scratch.as_mut().expect("NF4 requires shared scratch"),
                    self.nf4_lora_grad_workspace
                        .as_mut()
                        .expect("NF4 requires LoRA grad workspace"),
                ) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!(
                            "[backward_nf4] Layer {} FAILED: {:?} (seq_len={}, hidden={})",
                            layer_idx, e, seq_len, self.config.model_config.hidden_size
                        );
                        return None;
                    }
                }

                // ENT-265: Clip LoRA gradients before optimizer step.
                // Without this, NF4 LoRA grads are unbounded — causes weight
                // divergence and embedding grad explosion (Run 7c: 26M at step 225).
                if let Some(max_norm) = max_grad_norm {
                    self.nf4_lora_grad_workspace
                        .as_mut()
                        .expect("NF4 requires LoRA grad ws")
                        .clip_gradients(max_norm, stream);
                }

                // NF4 LoRA optimizer step — always runs, even during accumulation.
                //
                // BUG FIX (entrenar#264): Previously gated by `if !accumulate_only`.
                // Design: NF4 LoRA has ~6M params, so we scale lr by 1/accum_steps
                // for micro-batches instead of accumulating gradients.
                {
                    let step = self.gpu_training.step;
                    let effective_lr = if accumulate_only {
                        lr / self.config.accumulation_steps as f32
                    } else {
                        lr
                    };
                    if let Some(ref mut opt_states) = self.nf4_lora_optimizer_states {
                        let _ = self.cuda_blocks[layer_idx].lora_optimizer_step(
                            &mut opt_states[layer_idx],
                            step,
                            effective_lr,
                            beta1,
                            beta2,
                            1e-8,
                            weight_decay,
                            stream,
                            self.nf4_lora_grad_workspace
                                .as_ref()
                                .expect("NF4 requires LoRA grad ws"),
                        );
                    }
                }
            } else {
                // Standard fp32 backward path
                self.cuda_blocks[layer_idx]
                    .backward(
                        &self.gpu_training.layer_inputs[layer_idx],
                        grad_output,
                        grad_input,
                        seq_len,
                        stream,
                        &mut self.cuda_grad_workspace,
                    )
                    .ok()?;

                // C-CLIP-001 / entrenar#312: DISABLED per-block gradient clipping.
                // Per-block clipping distorts gradient flow across layers.

                // C-BACKPARITY-001: Per-block gradient norm tracing for parity testing.
                // Only runs when ENTRENAR_TRACE_GRADIENTS=1 — zero overhead in production.
                if std::env::var("ENTRENAR_TRACE_GRADIENTS").is_ok() {
                    let (_, block_gnorm) = compute_workspace_clip_scale_gpu(
                        &self.cuda_grad_workspace,
                        f32::MAX,
                        stream,
                    );
                    // Also trace the activation gradient (flows between blocks)
                    let act_sq = squared_sum_cuda(grad_input, grad_input.len() as u32, stream)
                        .unwrap_or(0.0);
                    let act_gnorm = act_sq.sqrt();
                    eprintln!(
                        "[grad-trace] block={layer_idx} weight_gnorm={block_gnorm:.6} act_gnorm={act_gnorm:.6}"
                    );
                }

                // R-038: Either accumulate workspace grads or run optimizer per-block.
                if accumulate_only {
                    // ALB-091: GPU-resident accumulation (no sync, no D2H) or CPU fallback.
                    if let Some(ref mut gpu_accum) = self.gpu_grad_accum {
                        let _ = gpu_accum.accumulate_block(
                            &self.cuda_grad_workspace,
                            layer_idx,
                            stream,
                        );
                    } else {
                        // CPU fallback: SYNC + D2H (ALB-065 / Rule 6).
                        stream.synchronize().ok()?;
                        if let Some(accum) = &mut self.grad_accum {
                            Self::download_workspace_to_accum(
                                &self.cuda_grad_workspace,
                                accum,
                                layer_idx,
                                &mut self.d2h_staging,
                            )?;
                        }
                    }
                } else {
                    // Per-block optimizer step: consume workspace gradients before next block overwrites
                    let step = self.gpu_training.step;
                    let _ = self.cuda_blocks[layer_idx].optimizer_step(
                        &mut self.gpu_training.optimizer_states[layer_idx],
                        step,
                        lr,
                        beta1,
                        beta2,
                        1e-8,
                        weight_decay,
                        stream,
                        &self.cuda_grad_workspace,
                    );
                }
            }

            self.profiler.end_layer_bwd(layer_idx);
            grad_output_is_a = !grad_output_is_a;
        }

        stream.synchronize().ok()?;
        self.profiler.end(StepProfiler::BLK_BWD);

        Some(grad_output_is_a)
    }

    /// R-038: Download non-block (LM head + final norm) gradients to CPU accumulator.
    /// Static method to avoid borrow conflicts.
    // KAIZEN-044: Pre-allocate single buffer for LM head + norm D2H downloads.
    // lm_head_grad is vocab×hidden (389M elements = 1.5 GB for Qwen3-4B).
    // KAIZEN-059: Host buffer now passed in (d2h_staging) — zero per-call allocations.
    fn download_nonblock_grads_to_accum(
        lm_head_grad: &GpuBuffer<f32>,
        final_norm_grad: &GpuBuffer<f32>,
        grad_accum: &mut Option<super::grad_accumulator::PerBlockGradientAccumulator>,
        host: &mut [f32],
    ) -> Option<()> {
        let accum = grad_accum.as_mut()?;

        let lm_slice = &mut host[..lm_head_grad.len()];
        lm_head_grad.copy_to_host_at(lm_slice, 0).ok()?;
        for (d, s) in accum.lm_head_grad.iter_mut().zip(lm_slice.iter()) {
            *d += s;
        }

        let norm_slice = &mut host[..final_norm_grad.len()];
        final_norm_grad.copy_to_host_at(norm_slice, 0).ok()?;
        for (d, s) in accum.final_norm_grad.iter_mut().zip(norm_slice.iter()) {
            *d += s;
        }
        Some(())
    }

    /// Run LM head + final norm optimizer step (non-accumulating path).
    /// Static method to avoid borrow conflicts with `stream`.
    #[allow(clippy::too_many_arguments)]
    fn run_nonblock_optimizer_step(
        gpu_training: &mut GpuPretrainState,
        lm_head_weight_gpu: Option<&mut GpuBuffer<f32>>,
        lm_head_grad_gpu: &GpuBuffer<f32>,
        lm_head_m: &mut GpuBuffer<f32>,
        lm_head_v: &mut GpuBuffer<f32>,
        final_norm_m: &mut GpuBuffer<f32>,
        final_norm_v: &mut GpuBuffer<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        stream: &CudaStream,
    ) {
        gpu_training.step += 1;
        let step = gpu_training.step;

        if let Some(lm_head_weight) = lm_head_weight_gpu {
            let n_lm = lm_head_weight.len() as u32;
            let _ = adamw_step_cuda(
                lm_head_weight,
                lm_head_grad_gpu,
                lm_head_m,
                lm_head_v,
                lr,
                beta1,
                beta2,
                1e-8,
                weight_decay,
                step,
                n_lm,
                stream,
            );
        }

        let n_norm = gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut gpu_training.final_norm_weight,
            &gpu_training.grad_final_norm_weight,
            final_norm_m,
            final_norm_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_norm,
            stream,
        );
    }

    /// R-038: Download shared CudaGradWorkspace to CPU per-block accumulation buffers.
    ///
    /// Static method to avoid borrow conflicts with `stream` (same pattern as
    /// `recompute_segment`). Must be called after stream.synchronize() (ALB-065 / Rule 6).
    // KAIZEN-044: Pre-allocate a single host buffer for all D2H downloads
    // in download_workspace_to_accum. Was allocating vec![0.0f32; len] × 9 buffers.
    // KAIZEN-059: Host buffer now passed in (d2h_staging) — zero per-call allocations.
    fn download_workspace_to_accum(
        ws: &CudaGradWorkspace,
        accum: &mut super::grad_accumulator::PerBlockGradientAccumulator,
        layer_idx: usize,
        host: &mut [f32],
    ) -> Option<()> {
        let bg = &mut accum.block_grads[layer_idx];

        use super::grad_accumulator::component;
        let bufs_and_components: [(&GpuBuffer<f32>, usize); 9] = [
            (&ws.grad_w_q, component::W_Q),
            (&ws.grad_w_k, component::W_K),
            (&ws.grad_w_v, component::W_V),
            (&ws.grad_w_o, component::W_O),
            (&ws.grad_gate, component::GATE),
            (&ws.grad_up, component::UP),
            (&ws.grad_down, component::DOWN),
            (&ws.grad_input_norm, component::INPUT_NORM),
            (&ws.grad_post_attn_norm, component::POST_ATTN_NORM),
        ];

        for (gpu_buf, comp_idx) in &bufs_and_components {
            let slice = &mut host[..gpu_buf.len()];
            gpu_buf.copy_to_host_at(slice, 0).ok()?;
            for (d, s) in bg.components[*comp_idx].iter_mut().zip(slice.iter()) {
                *d += s;
            }
        }
        Some(())
    }

    /// R-038: Upload averaged CPU accumulation buffers to GPU workspace and run
    /// optimizer step for all blocks + LM head + final norm.
    ///
    /// Called once after `accumulation_steps` micro-batches have been accumulated.
    /// ALB-091: Run optimizer step from GPU-resident accumulated gradients.
    /// D2D copy accum → workspace, then run per-block optimizer. Zero accum after.
    fn gpu_optimizer_from_gpu_accum(&mut self) -> Option<()> {
        let stream = self.cuda_trainer.stream();
        let lr = self.current_lr();
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let weight_decay = self.config.weight_decay;

        // Sync once to ensure all accumulation kernels complete
        stream.synchronize().ok()?;

        self.gpu_training.step += 1;
        let step = self.gpu_training.step;

        // Upload GPU accum → workspace (D2D) and run optimizer per block
        let gpu_accum = self.gpu_grad_accum.as_ref()?;
        for layer_idx in 0..self.cuda_blocks.len() {
            gpu_accum.upload_to_workspace(&mut self.cuda_grad_workspace, layer_idx).ok()?;

            let _ = self.cuda_blocks[layer_idx].optimizer_step(
                &mut self.gpu_training.optimizer_states[layer_idx],
                step,
                lr,
                beta1,
                beta2,
                1e-8,
                weight_decay,
                stream,
                &self.cuda_grad_workspace,
            );
        }

        // LM head: D2D copy accum → grad buffer, then optimizer step
        gpu_accum
            .upload_nonblock(
                &mut self.lm_head_grad_gpu,
                &mut self.gpu_training.grad_final_norm_weight,
            )
            .ok()?;

        let n_lm = self.lm_head_weight_gpu.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.lm_head_weight_gpu,
            &self.lm_head_grad_gpu,
            &mut self.lm_head_m,
            &mut self.lm_head_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_lm,
            stream,
        );

        // Final norm optimizer step
        let n_norm = self.gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.gpu_training.final_norm_weight,
            &self.gpu_training.grad_final_norm_weight,
            &mut self.final_norm_m,
            &mut self.final_norm_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_norm,
            stream,
        );

        stream.synchronize().ok()?;

        // Zero accum for next window
        if let Some(ref mut gpu_accum) = self.gpu_grad_accum {
            let _ = gpu_accum.zero_all();
        }

        Some(())
    }

    #[allow(unsafe_code)]
    fn gpu_optimizer_from_accum(&mut self) -> Option<()> {
        let stream = self.cuda_trainer.stream();
        let lr = self.current_lr();
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let weight_decay = self.config.weight_decay;

        // Average accumulated gradients
        let accum = self.grad_accum.as_mut()?;
        accum.average();

        // Jidoka: check for NaN/Inf before applying
        if accum.has_non_finite() {
            println!("[WARN] R-038: NaN/Inf in accumulated gradients, skipping optimizer step");
            accum.zero_all();
            return Some(());
        }

        self.gpu_training.step += 1;
        let step = self.gpu_training.step;

        // Upload accumulated gradients and run optimizer for each block
        use super::grad_accumulator::component;
        for layer_idx in 0..self.cuda_blocks.len() {
            let bg = &accum.block_grads[layer_idx];

            // Upload accumulated gradients to shared workspace
            // SAFETY: async host-to-device copies within the training stream; host buffers
            // (bg.components) are stable for the duration of the stream operations.
            unsafe {
                self.cuda_grad_workspace
                    .grad_w_q
                    .copy_from_host_async(&bg.components[component::W_Q], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_w_k
                    .copy_from_host_async(&bg.components[component::W_K], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_w_v
                    .copy_from_host_async(&bg.components[component::W_V], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_w_o
                    .copy_from_host_async(&bg.components[component::W_O], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_gate
                    .copy_from_host_async(&bg.components[component::GATE], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_up
                    .copy_from_host_async(&bg.components[component::UP], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_down
                    .copy_from_host_async(&bg.components[component::DOWN], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_input_norm
                    .copy_from_host_async(&bg.components[component::INPUT_NORM], stream)
                    .ok()?;
                self.cuda_grad_workspace
                    .grad_post_attn_norm
                    .copy_from_host_async(&bg.components[component::POST_ATTN_NORM], stream)
                    .ok()?;
            }

            // Run optimizer step with uploaded averaged gradients
            let _ = self.cuda_blocks[layer_idx].optimizer_step(
                &mut self.gpu_training.optimizer_states[layer_idx],
                step,
                lr,
                beta1,
                beta2,
                1e-8,
                weight_decay,
                stream,
                &self.cuda_grad_workspace,
            );
        }

        // Upload accumulated LM head gradients and run AdamW step
        // entrenar#314: Skip GPU LM head optimizer for tied weights.
        // SAFETY: async host-to-device copy; host buffer (accum.lm_head_grad) is stable.
        unsafe {
            self.lm_head_grad_gpu.copy_from_host_async(&accum.lm_head_grad, stream).ok()?;
        }
        let n_lm = self.lm_head_weight_gpu.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.lm_head_weight_gpu,
            &self.lm_head_grad_gpu,
            &mut self.lm_head_m,
            &mut self.lm_head_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_lm,
            stream,
        );

        // Upload accumulated final norm gradients and run AdamW step
        // SAFETY: async host-to-device copy; host buffer (accum.final_norm_grad) is stable.
        unsafe {
            self.gpu_training
                .grad_final_norm_weight
                .copy_from_host_async(&accum.final_norm_grad, stream)
                .ok()?;
        }
        let n_norm = self.gpu_training.final_norm_weight.len() as u32;
        let _ = adamw_step_cuda(
            &mut self.gpu_training.final_norm_weight,
            &self.gpu_training.grad_final_norm_weight,
            &mut self.final_norm_m,
            &mut self.final_norm_v,
            lr,
            beta1,
            beta2,
            1e-8,
            weight_decay,
            step,
            n_norm,
            stream,
        );

        stream.synchronize().ok()?;

        // Zero accum for next window
        accum.zero_all();
        Some(())
    }

    /// Compute gradient L2 norm via GPU reduction kernel (KAIZEN-049).
    ///
    /// Runs `SquaredSumKernel` on GPU, downloads only `num_blocks` partial sums (~1KB)
    /// instead of the full buffer (128MB for lm_head). Falls back to CPU download on error.
    ///
    /// # Contract (C-CLIPNORM-GPU-001)
    ///
    /// - **Precondition**: `buf.len() > 0`, stream is synchronized with prior kernel
    /// - **Postcondition**: `grad_norm ≈ sqrt(sum(buf[i]^2))`, `scale = min(1, max_norm/norm)`
    /// - **Transfer**: ~1KB D2H (num_blocks × 4B) vs n×4B (128MB for 32M elements)
    ///
    /// R-004: Returns `(clip_scale, grad_norm)` for observability.
    fn compute_clip_scale_with_norm(
        buf: &GpuBuffer<f32>,
        max_norm: f32,
        stream: &CudaStream,
    ) -> (f32, f32) {
        let n = buf.len() as u32;
        // Try GPU reduction first — ~1KB D2H instead of n×4 bytes
        let grad_norm = match squared_sum_cuda(buf, n, stream) {
            Ok(norm) => norm,
            Err(_) => {
                // Fallback: full D2H (original path)
                let mut host = vec![0.0f32; buf.len()];
                if buf.copy_to_host_at(&mut host, 0).is_err() {
                    return (1.0, 0.0);
                }
                let sq_sum: f64 = host.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                sq_sum.sqrt() as f32
            }
        };
        let scale = if grad_norm > max_norm { max_norm / grad_norm } else { 1.0 };
        (scale, grad_norm)
    }

    /// Download embedding gradient from GPU, clip, and scatter-add into CPU weight.
    ///
    /// # Contract (C-EMBED-GRAD-001)
    ///
    /// The activation gradient from block[0]'s backward is unclipped (per-block clipping
    /// only applies to weight gradients in the shared workspace). For deep networks with
    /// random init, this gradient can overflow f32, producing NaN in the CPU AdamW.
    /// We clip the activation gradient to max_grad_norm before scatter-adding.
    #[allow(unsafe_code)]
    fn embed_backward(
        &mut self,
        input_ids: &[u32],
        _seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        grad_output_is_a: bool,
    ) -> Option<()> {
        // The final backward output is in whichever buffer was last written
        let grad_a_ptr: *const GpuBuffer<f32> = &raw const self.gpu_training.grad_buf_a;
        let grad_b_ptr: *const GpuBuffer<f32> = &raw const self.gpu_training.grad_buf_b;
        let embed_grad_buf = unsafe {
            if grad_output_is_a {
                &*grad_a_ptr
            } else {
                &*grad_b_ptr
            }
        };
        let mut embed_grad_data = self.cuda_trainer.download(embed_grad_buf).ok()?;

        // C-EMBED-GRAD-001: ALWAYS clip activation gradient before scatter-add.
        // Without this, 24-layer random-init backward amplifies gradients to ~1e35,
        // which overflows the CPU AdamW's second moment buffer.
        //
        // ALB-071: Decoupled from general grad_clip config. Embed activation gradient
        // clipping is a SAFETY constraint (prevents NaN), not a training hyperparameter.
        // Uses dedicated max_embed_grad_norm (default 1.0) independent of weight grad_clip.
        let embed_clip_norm = self.config.base.max_grad_norm.unwrap_or(1.0);
        {
            let sq_sum: f64 = embed_grad_data.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let grad_norm = sq_sum.sqrt() as f32;
            self.last_embed_grad_norm = grad_norm; // R-040: per-parameter-group tracking
            if grad_norm > embed_clip_norm {
                let scale = embed_clip_norm / grad_norm;
                for g in &mut embed_grad_data {
                    *g *= scale;
                }
            }
        }

        // KAIZEN-048: In-place scatter-add via grad_cell().borrow_mut().
        // Before: 3 × 128MB clones per step (grad() deep-copies Array1).
        // After: zero clones — mutate existing gradient buffer directly.
        let embed_weight = &mut self.model.embed_tokens.weight;
        let grad_cell = embed_weight.grad_cell();
        let mut grad_ref = grad_cell.borrow_mut();
        if grad_ref.is_none() {
            *grad_ref = Some(ndarray::Array1::zeros(embed_weight.len()));
        }
        if let Some(grad) = grad_ref.as_mut() {
            for (pos, &token_id) in input_ids.iter().enumerate() {
                let tid = token_id as usize;
                if tid < vocab_size {
                    let src = pos * hidden_size;
                    let dst = tid * hidden_size;
                    for h in 0..hidden_size {
                        grad[dst + h] += embed_grad_data[src + h];
                    }
                }
            }
        }
        Some(())
    }

    /// Apply optimizer step to CPU embedding and update metrics.
    ///
    /// GPU block optimizer steps now run interleaved with backward in `gpu_backward()`.
    /// LM head and final norm optimizer steps also run in `gpu_backward()`.
    /// This method handles only CPU embedding and bookkeeping.
    fn optimizer_step(&mut self) {
        // ALB-072: Gradients are no longer scaled by grad_scaler (loss_scale excludes
        // grad_scaler.scale()). All backward computation uses f32 — no fp16 underflow
        // risk. Skip unscaling; just update scaler as successful.
        self.grad_scaler.update(true);

        // ALB-079: Sync CPU embedding optimizer lr with cosine schedule
        self.embed_optimizer.set_lr(self.current_lr());
        // CPU optimizer step for embedding weight
        let mut embed_params = vec![&mut self.model.embed_tokens.weight];
        self.embed_optimizer.step_refs(&mut embed_params);

        self.step += 1;
        self.metrics.losses.push(self.accumulated_loss);
        self.metrics.increment_step();

        self.accumulated_loss = 0.0;
        self.accumulated_batches = 0;
    }

    /// Process a batch (forward + backward + optimizer step with accumulation).
    ///
    /// R-038: When `accumulation_steps > 1`, runs forward+backward without optimizer
    /// for each micro-batch, downloading per-block weight gradients to CPU-side
    /// `PerBlockGradientAccumulator`. After `accumulation_steps` batches, averages
    /// the accumulated gradients, uploads them to GPU, and runs a single optimizer step.
    ///
    /// When `accumulation_steps == 1` (default), runs forward+backward+optimizer
    /// immediately per sequence (original behavior).
    ///
    /// Returns average loss for the batch.
    pub fn train_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        let accumulating = self.grad_accum.is_some() || self.gpu_grad_accum.is_some();

        if self.accumulated_batches == 0 {
            // Zero embedding gradients at start of accumulation window
            self.embed_optimizer.zero_grad_refs(&mut vec![&mut self.model.embed_tokens.weight]);
        }

        let mut total_loss = 0.0;
        let mut valid_count = 0;

        for i in 0..batch.batch_size {
            let Some(input_ids) = batch.get_input(i) else {
                continue;
            };
            let Some(target_ids) = batch.get_target(i) else {
                continue;
            };

            // R-038: When accumulating, run backward without optimizer (accumulate_only=true).
            // Gradients are downloaded to CPU per-block accum buffers. Embedding grads are
            // scatter-added normally (they're already on CPU).
            if let Some(loss) = self.train_step_single(input_ids, target_ids, accumulating) {
                total_loss += loss;
                valid_count += 1;
                if accumulating {
                    if let Some(accum) = &mut self.gpu_grad_accum {
                        accum.accumulated_count += 1;
                    } else if let Some(accum) = &mut self.grad_accum {
                        accum.accumulated_count += 1;
                    }
                }
            }
        }

        let avg_loss = if valid_count > 0 { total_loss / valid_count as f32 } else { 0.0 };

        // Debug: help diagnose loss=0.0 when gradients are non-zero
        if avg_loss == 0.0 && valid_count > 0 {
            eprintln!(
                "[train_batch DEBUG] avg_loss=0.0 but valid_count={}, total_loss={}, batch_size={}",
                valid_count, total_loss, batch.batch_size
            );
        }

        self.accumulated_loss += avg_loss / self.config.accumulation_steps as f32;
        self.accumulated_batches += 1;

        if self.accumulated_batches >= self.config.accumulation_steps {
            if accumulating {
                // ALB-091: Prefer GPU-resident accum path (zero D2H), fall back to CPU.
                if self.gpu_grad_accum.is_some() {
                    self.gpu_optimizer_from_gpu_accum();
                } else {
                    self.gpu_optimizer_from_accum();
                }
            }
            self.optimizer_step();
        }

        avg_loss
    }

    /// R-005: Evaluate a batch without backward pass or weight updates.
    /// Returns average cross-entropy loss, or 0.0 if no valid items.
    /// KAIZEN-050: Uses fused GPU cross-entropy (no logits D2H).
    pub fn eval_batch(&mut self, batch: &LMBatch) -> f32 {
        let hidden_size = self.config.model_config.hidden_size;
        let vocab_size = self.config.model_config.vocab_size;
        let max_sl = self.config.max_seq_len;
        let mut total_loss = 0.0;
        let mut valid_count = 0;
        for i in 0..batch.batch_size {
            if let Some(loss) = self.eval_single_sequence(batch, i, max_sl, hidden_size, vocab_size)
            {
                total_loss += loss;
                valid_count += 1;
            }
        }
        if valid_count > 0 {
            total_loss / valid_count as f32
        } else {
            0.0
        }
    }

    /// Evaluate a single sequence from a batch. Returns None if invalid.
    fn eval_single_sequence(
        &mut self,
        batch: &LMBatch,
        i: usize,
        max_sl: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<f32> {
        let input_ids = batch.get_input(i)?;
        let target_ids = batch.get_target(i)?;
        // Truncate to max_seq_len — GPU buffers are pre-allocated for this size
        let input_ids = if input_ids.len() > max_sl { &input_ids[..max_sl] } else { input_ids };
        let target_ids = if target_ids.len() > max_sl { &target_ids[..max_sl] } else { target_ids };
        let seq_len = input_ids.len();
        self.gpu_forward(input_ids, seq_len, hidden_size, vocab_size)?;
        let stream = self.cuda_trainer.stream();
        let scale = 1.0 / seq_len as f32;
        let loss = fused_cross_entropy_cuda(
            &mut self.gpu_training.logits_buf,
            target_ids,
            seq_len as u32,
            vocab_size as u32,
            scale,
            stream,
        )
        .ok()?;
        if loss.is_finite() {
            Some(loss)
        } else {
            None
        }
    }

    /// Train for one epoch over batches.
    pub fn train_epoch(&mut self, batches: &[LMBatch]) -> f32 {
        self.train_epoch_with_callback(batches, |_, _, _| {})
    }

    /// Train for one epoch with a per-step callback.
    ///
    /// Stops early if `max_steps` is set and reached.
    pub fn train_epoch_with_callback<F>(&mut self, batches: &[LMBatch], mut on_batch: F) -> f32
    where
        F: FnMut(usize, f32, &Self),
    {
        if batches.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        for (i, batch) in batches.iter().enumerate() {
            if let Some(max) = self.config.max_steps {
                if self.step >= max {
                    break;
                }
            }

            let batch_loss = self.train_batch(batch);
            total_loss += batch_loss;
            batches_processed += 1;
            on_batch(i, batch_loss, self);
        }

        // KAIZEN-047: Print profiler summary at end of epoch
        if self.profiler.is_enabled() && self.profiler.step_count() > 0 {
            self.profiler.print_report();
        }

        total_loss / batches_processed.max(1) as f32
    }

    // --- DDP (data-parallel) support methods ---

    /// Ensure the per-block gradient accumulator exists.
    ///
    /// For DDP, we always need accumulation buffers (even with accumulation_steps=1)
    /// because gradients must be downloaded to CPU for AllReduce before optimizer step.
    pub(crate) fn ensure_grad_accum(&mut self) {
        if self.grad_accum.is_some() {
            return;
        }
        let mc = &self.config.model_config;
        let hidden_size = mc.hidden_size;
        let kv_hidden = mc.num_kv_heads * mc.head_dim();
        let block_sizes = super::grad_accumulator::PerBlockGradientAccumulator::compute_block_sizes(
            hidden_size,
            kv_hidden,
            mc.intermediate_size,
        );
        self.grad_accum = Some(super::grad_accumulator::PerBlockGradientAccumulator::new(
            self.cuda_blocks.len(),
            block_sizes,
            mc.vocab_size,
            hidden_size,
        ));
    }

    /// Forward + backward for one batch, always accumulating (no optimizer step).
    ///
    /// Used by `DistributedCudaTrainer` to compute local gradients before AllReduce.
    /// Returns average loss for the batch.
    pub(crate) fn forward_backward_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        if self.accumulated_batches == 0 {
            self.embed_optimizer.zero_grad_refs(&mut vec![&mut self.model.embed_tokens.weight]);
        }

        let mut total_loss = 0.0;
        let mut valid_count = 0;

        for i in 0..batch.batch_size {
            let Some(input_ids) = batch.get_input(i) else { continue };
            let Some(target_ids) = batch.get_target(i) else { continue };

            // Always accumulate_only=true: gradients go to CPU accum buffers
            if let Some(loss) = self.train_step_single(input_ids, target_ids, true) {
                total_loss += loss;
                valid_count += 1;
                if let Some(accum) = &mut self.grad_accum {
                    accum.accumulated_count += 1;
                }
            }
        }

        if valid_count > 0 {
            total_loss / valid_count as f32
        } else {
            0.0
        }
    }

    /// Apply DDP-averaged gradients: upload to GPU and run optimizer step.
    ///
    /// Called after AllReduce has written averaged gradients into the grad_accum.
    /// Runs gpu_optimizer_from_accum() for blocks + LM head + final norm,
    /// then optimizer_step() for embedding.
    pub(crate) fn apply_ddp_gradients(&mut self) {
        self.accumulated_loss = 0.0;
        self.accumulated_batches = 0;
        self.gpu_optimizer_from_accum();
        self.optimizer_step();
    }

    /// Get a reference to the gradient accumulator (for DDP AllReduce).
    pub(crate) fn grad_accum_ref(
        &self,
    ) -> Option<&super::grad_accumulator::PerBlockGradientAccumulator> {
        self.grad_accum.as_ref()
    }

    /// Get a mutable reference to the gradient accumulator (for DDP AllReduce).
    pub(crate) fn grad_accum_mut(
        &mut self,
    ) -> Option<&mut super::grad_accumulator::PerBlockGradientAccumulator> {
        self.grad_accum.as_mut()
    }

    /// Get the training config.
    pub(crate) fn config(&self) -> &TransformerTrainConfig {
        &self.config
    }

    /// Get CPU embedding gradient as flat Vec for AllReduce.
    pub(crate) fn embed_grad_vec(&self) -> Option<Vec<f32>> {
        self.model.embed_tokens.weight.grad().map(|g| g.to_vec())
    }

    /// Set CPU embedding gradient from AllReduced flat Vec.
    pub(crate) fn set_embed_grad(&mut self, grad: Vec<f32>) {
        self.model.embed_tokens.weight.set_grad(ndarray::Array1::from(grad));
    }

    /// Returns true if max_steps has been reached.
    pub fn reached_max_steps(&self) -> bool {
        self.config.max_steps.is_some_and(|max| self.step >= max)
    }

    /// Get current step count.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Set initial step for resume from checkpoint.
    ///
    /// Updates both the outer step counter (LR schedule, logging) and the
    /// GPU-side AdamW step counter (bias correction). Must be called before
    /// any `train_batch()` calls.
    pub fn set_initial_step(&mut self, step: usize) {
        self.step = step;
        self.gpu_training.step = step as u32;
    }

    /// Set max_steps for cosine LR scheduler (ENT-275).
    ///
    /// Called by `train_loop_cuda` when `max_steps` is not explicitly set in
    /// the YAML config — auto-computes `epochs × batches_per_epoch` so cosine
    /// decay activates instead of falling back to constant lr.
    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.config.max_steps = Some(max_steps);
    }

    /// Get current learning rate (warmup + cosine decay).
    ///
    /// ALB-079: Phase 1 = linear warmup (0 → lr_max), Phase 2 = cosine decay
    /// (lr_max → 0) over remaining steps. Requires `max_steps` for decay;
    /// without it, falls back to constant lr after warmup.
    pub fn current_lr(&self) -> f32 {
        let base_lr = self.config.lr;
        if self.step < self.config.warmup_steps {
            // Phase 1: Linear warmup
            base_lr * (self.step as f32 / self.config.warmup_steps.max(1) as f32)
        } else if let Some(max_steps) = self.config.max_steps {
            // Phase 2: Cosine decay from lr_max to 0
            let decay_steps = max_steps.saturating_sub(self.config.warmup_steps);
            if decay_steps == 0 {
                return base_lr;
            }
            let decay_step = self.step - self.config.warmup_steps;
            let progress = (decay_step as f32 / decay_steps as f32).min(1.0);
            0.5 * base_lr * (1.0 + (std::f32::consts::PI * progress).cos())
        } else {
            // No max_steps: constant lr (legacy behavior)
            base_lr
        }
    }

    /// KAIZEN-047: Enable step profiling with a report every `interval` steps.
    ///
    /// When enabled, prints a table of wall-clock timings per training phase
    /// every `interval` training steps. Use interval=0 for manual-only reporting.
    ///
    /// # Contract (C-STEPPROF-001)
    ///
    /// - No additional GPU synchronization points (relies on existing syncs)
    /// - Overhead: ~11 `Instant::now()` calls per step (~1µs total on Linux)
    /// - Timings include async dispatch overhead (not pure kernel time)
    pub fn enable_profiler(&mut self, interval: usize) {
        self.profiler = StepProfiler::new(true, interval);
    }

    /// Print the profiler report (if profiling is enabled).
    pub fn print_profiler_report(&self) {
        self.profiler.print_report();
    }

    /// R-004: Get last observed gradient L2 norm (LM head proxy).
    pub fn last_grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    /// R-040: Get per-parameter-group gradient norms.
    /// Returns (lm_head_grad_norm, embed_grad_norm).
    pub fn param_grad_norms(&self) -> (f32, f32) {
        (self.last_grad_norm, self.last_embed_grad_norm)
    }

    /// R-012: Get total trainable parameter count for MFU calculation.
    pub fn num_params(&self) -> usize {
        self.model.parameters().iter().map(|t| t.len()).sum()
    }

    /// R-013: Query GPU memory usage (used_mb, total_mb).
    pub fn gpu_memory_mb(&self) -> (u64, u64) {
        match self.cuda_trainer.context().memory_info() {
            Ok((free, total)) => {
                let total_mb = (total / (1024 * 1024)) as u64;
                let used_mb = ((total - free) / (1024 * 1024)) as u64;
                (used_mb, total_mb)
            }
            Err(_) => (0, 0),
        }
    }

    /// Sync all GPU weights back to CPU model.
    ///
    /// # Contract (C-SYNCWT-001)
    ///
    /// Must be called before save or any CPU model access after training.
    pub fn sync_weights_to_cpu(&mut self) {
        let use_nf4 = self.config.quantize_nf4 && self.config.is_lora();

        if use_nf4 {
            // ENT-263: NF4 blocks are frozen — base weights don't change.
            // Only download LoRA adapter weights for checkpoint saving.
            // The base model on CPU stays as-is (original pretrained weights).
            // LoRA weights are saved separately (adapter_config.json + adapter.safetensors).
            // For now, skip per-layer sync — base weights are unchanged.
        } else {
            for (layer_idx, block) in self.cuda_blocks.iter().enumerate() {
                if let Ok(weights) = block.download_weights() {
                    let layer = &mut self.model.layers[layer_idx];

                    layer.self_attn.w_q = Tensor::from_vec(weights.w_q, false);
                    layer.self_attn.w_k = Tensor::from_vec(weights.w_k, false);
                    layer.self_attn.w_v = Tensor::from_vec(weights.w_v, false);
                    layer.self_attn.w_o = Tensor::from_vec(weights.w_o, false);

                    layer.ffn.w_gate = Tensor::from_vec(weights.w_gate, false);
                    layer.ffn.w_up = Tensor::from_vec(weights.w_up, false);
                    layer.ffn.w_down = Tensor::from_vec(weights.w_down, false);

                    layer.input_norm.weight = Tensor::from_vec(weights.input_norm_weight, false);
                    layer.post_attn_norm.weight =
                        Tensor::from_vec(weights.post_attn_norm_weight, false);
                }
            }
        }

        // Sync final norm weight
        if let Ok(norm_data) = self.cuda_trainer.download(&self.gpu_training.final_norm_weight) {
            self.model.norm.weight = Tensor::from_vec(norm_data, false);
        }

        // Sync LM head weight
        // ALB-097: ALWAYS save GPU-trained LM head, even for tied-weight models.
        // During GPU training, lm_head diverges from embed_tokens because they have
        // separate optimizers (GPU AdamW vs CPU AdamW). If we skip the sync for tied
        // weights, the checkpoint loses 500+ steps of GPU LM head training → random-init
        // loss on resume (Five Whys root cause of ALB-097).
        if let Ok(lm_data) = self.cuda_trainer.download(&self.lm_head_weight_gpu) {
            self.model.lm_head = Some(Tensor::from_vec(lm_data, false));
        }
    }

    /// Get reference to model (syncs weights first).
    pub fn model(&self) -> &Transformer {
        &self.model
    }

    /// Get mutable reference to model.
    pub fn model_mut(&mut self) -> &mut Transformer {
        &mut self.model
    }

    /// Check if using mixed precision.
    pub fn is_mixed_precision(&self) -> bool {
        self.config.precision_config.is_mixed()
    }

    /// Get the gradient scaler (R-002: loss scaling for mixed precision).
    pub fn grad_scaler(&self) -> &GradScaler {
        &self.grad_scaler
    }

    /// Check if using gradient checkpointing.
    pub fn is_checkpointing(&self) -> bool {
        self.config.checkpoint_config.enabled
    }

    /// Save model weights (syncs GPU→CPU first).
    pub fn save(
        &mut self,
        path: impl AsRef<std::path::Path>,
        name: &str,
        architecture: &str,
    ) -> crate::Result<()> {
        self.sync_weights_to_cpu();

        // Use named_parameters() for correct name mapping (handles attention biases etc.)
        let params: Vec<(String, Tensor)> = self
            .model
            .named_parameters()
            .into_iter()
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect();

        let metadata = ModelMetadata::new(name, architecture);
        let model = Model::new(metadata, params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        save_model(&model, path, &config)
    }

    /// R-011: Prepare checkpoint data for async save.
    /// Syncs GPU weights to CPU and snapshots tensor data as Send-able Vec<f32>.
    /// Returns a closure that writes the checkpoint file from another thread.
    pub fn prepare_async_save(
        &mut self,
        name: &str,
        architecture: &str,
    ) -> Box<dyn FnOnce(&std::path::Path) -> crate::Result<()> + Send> {
        self.sync_weights_to_cpu();

        // Use named_parameters() for correct name mapping (handles attention biases etc.)
        let param_data: Vec<(String, Vec<f32>)> = self
            .model
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, t.data().to_vec()))
            .collect();

        let name = name.to_string();
        let architecture = architecture.to_string();

        Box::new(move |path: &std::path::Path| {
            let params: Vec<(String, Tensor)> =
                param_data.into_iter().map(|(n, d)| (n, Tensor::from_vec(d, false))).collect();
            let metadata = ModelMetadata::new(&name, &architecture);
            let model = Model::new(metadata, params);
            let config = SaveConfig::new(ModelFormat::SafeTensors);
            save_model(&model, path, &config)
        })
    }

    /// ALB-096: Save model weights as APR checkpoint (syncs GPU→CPU first).
    ///
    /// Single atomic file containing all model weights. Use `save_apr_checkpoint()`
    /// to include optimizer state and training metadata in the same file.
    pub fn save_apr(
        &mut self,
        path: impl AsRef<std::path::Path>,
        name: &str,
        architecture: &str,
    ) -> crate::Result<()> {
        self.sync_weights_to_cpu();

        let params: Vec<(String, Tensor)> = self
            .model
            .named_parameters()
            .into_iter()
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect();

        let metadata = ModelMetadata::new(name, architecture);
        let model = Model::new(metadata, params);
        let config = SaveConfig::new(ModelFormat::Apr);

        save_model(&model, path, &config)
    }

    /// ALB-096: Prepare APR checkpoint data for async save.
    ///
    /// Syncs GPU weights to CPU and snapshots tensor data + optimizer state as
    /// Send-able `Vec<f32>`. Returns a closure that writes a single atomic APR
    /// file from another thread. Includes model weights + CPU embedding optimizer
    /// state + training metadata — all in one file.
    fn snapshot_param_data(&self) -> Vec<(String, Vec<f32>)> {
        let use_nf4 = self.config.quantize_nf4 && self.config.is_lora();
        if use_nf4 {
            let frozen_suffixes = [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            ];
            self.model
                .named_parameters()
                .into_iter()
                .filter(|(n, _)| !frozen_suffixes.iter().any(|s| n.ends_with(s)))
                .map(|(n, t)| (n, t.data().to_vec()))
                .collect()
        } else {
            self.model.named_parameters().into_iter().map(|(n, t)| (n, t.data().to_vec())).collect()
        }
    }

    fn snapshot_lora_data(&self) -> Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.config.quantize_nf4 && self.config.is_lora() {
            self.cuda_blocks
                .iter()
                .enumerate()
                .filter_map(|(i, block)| {
                    block
                        .download_lora_weights()
                        .ok()
                        .map(|(a_q, b_q, a_v, b_v)| (i, a_q, b_q, a_v, b_v))
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn prepare_async_apr_save(
        &mut self,
        name: &str,
        architecture: &str,
        step: usize,
        loss: f64,
        lr: f64,
    ) -> Box<dyn FnOnce(&std::path::Path) -> crate::Result<()> + Send> {
        self.prepare_async_apr_save_with_tokenizer(name, architecture, step, loss, lr, None)
    }

    /// ALB-130: Prepare APR checkpoint with embedded tokenizer for inference.
    ///
    /// Training checkpoints must be self-contained for eval (`apr eval --task humaneval`).
    /// Without embedded tokenizer, inference falls back to structural validation (fake 100%).
    /// The tokenizer path comes from `spec.data.tokenizer` in the training YAML.
    pub fn prepare_async_apr_save_with_tokenizer(
        &mut self,
        name: &str,
        architecture: &str,
        step: usize,
        loss: f64,
        lr: f64,
        tokenizer_path: Option<&std::path::Path>,
    ) -> Box<dyn FnOnce(&std::path::Path) -> crate::Result<()> + Send> {
        self.sync_weights_to_cpu();

        let param_data = self.snapshot_param_data();
        let lora_data = self.snapshot_lora_data();

        // Snapshot CPU embedding optimizer state
        let embed_m: Vec<Vec<f32>> = self
            .embed_optimizer
            .first_moments()
            .iter()
            .filter_map(|opt| opt.as_ref().map(ndarray::ArrayBase::to_vec))
            .collect();
        let embed_v: Vec<Vec<f32>> = self
            .embed_optimizer
            .second_moments()
            .iter()
            .filter_map(|opt| opt.as_ref().map(ndarray::ArrayBase::to_vec))
            .collect();
        let embed_step = self.embed_optimizer.step_count();

        // ALB-118: Download GPU block optimizer states (m/v moments) for checkpointing.
        // Without this, resume re-initializes all 24 blocks' AdamW state to zero,
        // causing loss spikes and convergence failure (v10/v11/v12 post-mortems).
        // Transfer cost: ~2.3 GB D2H, <6ms on PCIe4/5.
        let block_optim_data: Vec<Vec<(String, Vec<f32>)>> = self
            .gpu_training
            .optimizer_states
            .iter()
            .map(|state| state.download_to_host().unwrap_or_default())
            .collect();

        // ALB-118: Download LM head and final norm optimizer states
        let lm_head_m_host = {
            let mut buf = vec![0.0f32; self.lm_head_m.len()];
            let _ = self.lm_head_m.copy_to_host(&mut buf);
            buf
        };
        let lm_head_v_host = {
            let mut buf = vec![0.0f32; self.lm_head_v.len()];
            let _ = self.lm_head_v.copy_to_host(&mut buf);
            buf
        };
        let final_norm_m_host = {
            let mut buf = vec![0.0f32; self.final_norm_m.len()];
            let _ = self.final_norm_m.copy_to_host(&mut buf);
            buf
        };
        let final_norm_v_host = {
            let mut buf = vec![0.0f32; self.final_norm_v.len()];
            let _ = self.final_norm_v.copy_to_host(&mut buf);
            buf
        };

        let name = name.to_string();
        let architecture = architecture.to_string();
        let model_config_json = serde_json::to_string(&self.config.model_config).ok();
        let is_delta_checkpoint = self.config.quantize_nf4 && self.config.is_lora();

        // ALB-130: Pre-read tokenizer.json for embedding in checkpoint.
        // Parse HuggingFace tokenizer format → extract vocab + merges + special token IDs.
        let tokenizer_data: Option<(Vec<String>, Vec<String>, Option<u64>, Option<u64>)> =
            tokenizer_path.and_then(|p| {
                let json_bytes = std::fs::read(p).ok()?;
                let tok: serde_json::Value = serde_json::from_slice(&json_bytes).ok()?;
                let model = tok.get("model")?;
                let vocab_obj = model.get("vocab")?.as_object()?;
                // Build sorted-by-id vocab list
                let mut vocab_pairs: Vec<(String, u64)> =
                    vocab_obj.iter().filter_map(|(k, v)| Some((k.clone(), v.as_u64()?))).collect();
                vocab_pairs.sort_by_key(|(_, id)| *id);
                let vocab: Vec<String> = vocab_pairs.into_iter().map(|(k, _)| k).collect();
                // Merges as "token1 token2" strings
                let merges: Vec<String> = model
                    .get("merges")?
                    .as_array()?
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                // Special tokens: BOS=<s>=1, EOS=</s>=2 (from added_tokens)
                let added = tok.get("added_tokens").and_then(|a| a.as_array());
                let bos_id = added.and_then(|arr| {
                    arr.iter()
                        .find(|t| t.get("content").and_then(|c| c.as_str()) == Some("<s>"))
                        .and_then(|t| t.get("id")?.as_u64())
                });
                let eos_id = added.and_then(|arr| {
                    arr.iter()
                        .find(|t| t.get("content").and_then(|c| c.as_str()) == Some("</s>"))
                        .and_then(|t| t.get("id")?.as_u64())
                });
                if vocab.is_empty() {
                    return None;
                }
                println!(
                    "  [ALB-130] Embedding tokenizer: {} vocab, {} merges",
                    vocab.len(),
                    merges.len()
                );
                Some((vocab, merges, bos_id, eos_id))
            });

        Box::new(move |path: &std::path::Path| {
            use aprender::serialization::apr::AprWriter;
            use serde_json::Value as Jv;

            let mut writer = AprWriter::new();

            // Metadata
            writer.set_metadata("model_name", Jv::String(name));
            writer.set_metadata("architecture", Jv::String(architecture));
            writer.set_metadata(
                "format",
                Jv::String(if is_delta_checkpoint {
                    "entrenar-delta-checkpoint".into()
                } else {
                    "entrenar-checkpoint".into()
                }),
            );
            writer.set_metadata("checkpoint_step", Jv::String(step.to_string()));
            writer.set_metadata("loss", Jv::String(format!("{loss:.6}")));
            writer.set_metadata("learning_rate", Jv::String(format!("{lr:.6e}")));
            writer.set_metadata("optimizer_step", Jv::String(embed_step.to_string()));
            if let Some(cfg) = model_config_json {
                writer.set_metadata("model_config", Jv::String(cfg));
            }

            // ALB-130: Embed tokenizer vocab + merges for standalone inference
            if let Some((vocab, merges, bos_id, eos_id)) = tokenizer_data {
                writer.set_metadata(
                    "tokenizer.vocabulary",
                    Jv::Array(vocab.into_iter().map(Jv::String).collect()),
                );
                writer.set_metadata(
                    "tokenizer.merges",
                    Jv::Array(merges.into_iter().map(Jv::String).collect()),
                );
                if let Some(bos) = bos_id {
                    writer.set_metadata("tokenizer.bos_token_id", Jv::Number(bos.into()));
                }
                if let Some(eos) = eos_id {
                    writer.set_metadata("tokenizer.eos_token_id", Jv::Number(eos.into()));
                }
            }

            // Find hidden_size from norm weights for shape inference
            let hidden_size = param_data
                .iter()
                .find(|(n, _)| n.ends_with("layernorm.weight") || n == "model.norm.weight")
                .map_or(0, |(_, d)| d.len());

            // Model weight tensors
            for (tensor_name, data) in &param_data {
                let shape = infer_tensor_shape(tensor_name, data.len(), hidden_size);
                writer.add_tensor_f32(tensor_name.clone(), shape, data);
            }

            // Optimizer state tensors
            for (i, m_data) in embed_m.iter().enumerate() {
                let len = m_data.len();
                writer.add_tensor_f32(
                    format!("__training__.embed_optimizer.m.{i}"),
                    vec![len],
                    m_data,
                );
            }
            for (i, v_data) in embed_v.iter().enumerate() {
                let len = v_data.len();
                writer.add_tensor_f32(
                    format!("__training__.embed_optimizer.v.{i}"),
                    vec![len],
                    v_data,
                );
            }

            // ALB-118: Save GPU block optimizer states (m/v moments for all 24 blocks)
            for (layer_idx, buffers) in block_optim_data.iter().enumerate() {
                for (suffix, data) in buffers {
                    let len = data.len();
                    writer.add_tensor_f32(
                        format!("__training__.block_optimizer.{layer_idx}.{suffix}"),
                        vec![len],
                        data,
                    );
                }
            }

            // ALB-118: Save LM head and final norm optimizer states
            if !lm_head_m_host.is_empty() {
                let len = lm_head_m_host.len();
                writer.add_tensor_f32(
                    "__training__.lm_head_optimizer.m".to_string(),
                    vec![len],
                    &lm_head_m_host,
                );
                let len = lm_head_v_host.len();
                writer.add_tensor_f32(
                    "__training__.lm_head_optimizer.v".to_string(),
                    vec![len],
                    &lm_head_v_host,
                );
            }
            if !final_norm_m_host.is_empty() {
                let len = final_norm_m_host.len();
                writer.add_tensor_f32(
                    "__training__.final_norm_optimizer.m".to_string(),
                    vec![len],
                    &final_norm_m_host,
                );
                let len = final_norm_v_host.len();
                writer.add_tensor_f32(
                    "__training__.final_norm_optimizer.v".to_string(),
                    vec![len],
                    &final_norm_v_host,
                );
            }

            // ENT-276: Save LoRA adapter weights (QLoRA checkpoint resume)
            for (layer_idx, a_q, b_q, a_v, b_v) in &lora_data {
                if !a_q.is_empty() {
                    writer.add_tensor_f32(
                        format!("lora.{layer_idx}.q_proj.lora_a"),
                        vec![a_q.len()],
                        a_q,
                    );
                    writer.add_tensor_f32(
                        format!("lora.{layer_idx}.q_proj.lora_b"),
                        vec![b_q.len()],
                        b_q,
                    );
                }
                if !a_v.is_empty() {
                    writer.add_tensor_f32(
                        format!("lora.{layer_idx}.v_proj.lora_a"),
                        vec![a_v.len()],
                        a_v,
                    );
                    writer.add_tensor_f32(
                        format!("lora.{layer_idx}.v_proj.lora_b"),
                        vec![b_v.len()],
                        b_v,
                    );
                }
            }

            // Write APR checkpoint to file
            writer
                .write(path)
                .map_err(|e| crate::error::Error::Serialization(format!("APR save failed: {e}")))?;

            Ok(())
        })
    }

    /// GPU device name.
    pub fn gpu_name(&self) -> String {
        self.cuda_trainer.device_name()
    }

    /// ENT-269: Save LoRA adapter weights as PEFT-compatible files.
    ///
    /// Downloads LoRA A/B matrices from GPU, un-scales B (divide by lora_scale),
    /// transposes to PEFT convention (A=[rank, d_in], B=[d_out, rank]),
    /// and writes `adapter_model.safetensors` + `adapter_config.json`.
    ///
    /// # Contract: C-QLORA-SAVE-001
    ///
    /// NF4 QLoRA training MUST produce `adapter_model.safetensors` in output_dir.
    pub fn save_cuda_lora_adapter(
        &self,
        output_dir: &std::path::Path,
        base_model_name: Option<&str>,
    ) -> crate::Result<()> {
        if !self.config.quantize_nf4 || !self.config.is_lora() {
            return Ok(()); // Not a QLoRA run, nothing to save
        }

        let lora_rank = self.config.lora_rank.unwrap_or(16);
        let lora_alpha = self.config.lora_alpha.unwrap_or(2.0 * lora_rank as f32);
        let lora_scale = lora_alpha / lora_rank as f32;
        let hidden_size = self.config.model_config.hidden_size;
        let head_dim = self.config.model_config.head_dim();
        let q_dim = self.config.model_config.num_attention_heads * head_dim;
        let kv_hidden = self.config.model_config.num_kv_heads * head_dim;

        let lora_config =
            crate::lora::LoRAConfig::new(lora_rank, lora_alpha).target_qv_projections();

        let mut adapters: Vec<(String, crate::lora::LoRALayer)> = Vec::new();

        for (i, block) in self.cuda_blocks.iter().enumerate() {
            let (a_q, b_q_scaled, a_v, b_v_scaled) = match block.download_lora_weights() {
                Ok(weights) => weights,
                Err(_) => continue, // Skip non-NF4 blocks
            };

            if a_q.is_empty() && a_v.is_empty() {
                continue;
            }

            // Q projection LoRA
            if !a_q.is_empty() {
                // GPU stores A_q as [hidden, rank] row-major, PEFT expects [rank, hidden]
                let mut a_transposed = vec![0.0f32; lora_rank * hidden_size];
                for r in 0..hidden_size {
                    for c in 0..lora_rank {
                        a_transposed[c * hidden_size + r] = a_q[r * lora_rank + c];
                    }
                }

                // GPU stores B_q as [rank, q_dim] pre-scaled by lora_scale
                // PEFT expects [q_dim, rank] un-scaled
                let inv_scale = if lora_scale.abs() > 1e-10 { 1.0 / lora_scale } else { 1.0 };
                let mut b_transposed = vec![0.0f32; q_dim * lora_rank];
                for r in 0..lora_rank {
                    for c in 0..q_dim {
                        b_transposed[c * lora_rank + r] = b_q_scaled[r * q_dim + c] * inv_scale;
                    }
                }

                let base_weight = crate::autograd::Tensor::zeros(q_dim * hidden_size, false);
                let mut layer = crate::lora::LoRALayer::new(
                    base_weight,
                    q_dim,
                    hidden_size,
                    lora_rank,
                    lora_alpha,
                );
                // Overwrite the A and B data with trained weights
                layer.lora_a_mut().data_mut().assign(&ndarray::Array1::from(a_transposed));
                layer.lora_b_mut().data_mut().assign(&ndarray::Array1::from(b_transposed));

                adapters.push((format!("model.layers.{i}.self_attn.q_proj"), layer));
            }

            // V projection LoRA
            if !a_v.is_empty() {
                let mut a_transposed = vec![0.0f32; lora_rank * hidden_size];
                for r in 0..hidden_size {
                    for c in 0..lora_rank {
                        a_transposed[c * hidden_size + r] = a_v[r * lora_rank + c];
                    }
                }

                let inv_scale = if lora_scale.abs() > 1e-10 { 1.0 / lora_scale } else { 1.0 };
                let mut b_transposed = vec![0.0f32; kv_hidden * lora_rank];
                for r in 0..lora_rank {
                    for c in 0..kv_hidden {
                        b_transposed[c * lora_rank + r] = b_v_scaled[r * kv_hidden + c] * inv_scale;
                    }
                }

                let base_weight = crate::autograd::Tensor::zeros(kv_hidden * hidden_size, false);
                let mut layer = crate::lora::LoRALayer::new(
                    base_weight,
                    kv_hidden,
                    hidden_size,
                    lora_rank,
                    lora_alpha,
                );
                layer.lora_a_mut().data_mut().assign(&ndarray::Array1::from(a_transposed));
                layer.lora_b_mut().data_mut().assign(&ndarray::Array1::from(b_transposed));

                adapters.push((format!("model.layers.{i}.self_attn.v_proj"), layer));
            }
        }

        if adapters.is_empty() {
            println!("  [WARN] No LoRA adapters found to save");
            return Ok(());
        }

        let adapter_refs: Vec<(&str, &crate::lora::LoRALayer)> =
            adapters.iter().map(|(name, layer)| (name.as_str(), layer)).collect();

        std::fs::create_dir_all(output_dir).ok();
        crate::lora::save_adapter_peft(&adapter_refs, &lora_config, base_model_name, output_dir)
            .map_err(|e| crate::error::Error::Io(format!("Failed to save PEFT adapter: {e}")))?;

        let adapter_path = output_dir.join("adapter_model.safetensors");
        let size_mb =
            std::fs::metadata(&adapter_path).map(|m| m.len()).unwrap_or(0) / (1024 * 1024);
        println!(
            "✓ LoRA adapter saved ({} layers, {} MB) to {}",
            adapters.len(),
            size_mb,
            output_dir.display()
        );

        Ok(())
    }

    /// R-001: Save CPU embedding optimizer state (m/v buffers + step counter).
    ///
    /// Writes `optimizer_state.json` to the given directory. GPU block optimizer
    /// states remain on-device (D2H for 20 buffers × N blocks is deferred).
    pub fn save_optimizer_state(&self, dir: &std::path::Path) -> crate::Result<()> {
        let path = dir.join("optimizer_state.json");
        let m_data: Vec<Option<Vec<f32>>> = self
            .embed_optimizer
            .first_moments()
            .iter()
            .map(|opt| opt.as_ref().map(ndarray::ArrayBase::to_vec))
            .collect();
        let v_data: Vec<Option<Vec<f32>>> = self
            .embed_optimizer
            .second_moments()
            .iter()
            .map(|opt| opt.as_ref().map(ndarray::ArrayBase::to_vec))
            .collect();
        let state = serde_json::json!({
            "type": "adamw_cpu_embed",
            "step": self.embed_optimizer.step_count(),
            "m": m_data,
            "v": v_data,
        });
        let json_str = serde_json::to_string(&state).map_err(|e| {
            crate::error::Error::ConfigError(format!("serialize optimizer state: {e}"))
        })?;
        std::fs::write(&path, json_str)
            .map_err(|e| crate::error::Error::ConfigError(format!("write optimizer state: {e}")))?;
        Ok(())
    }

    /// ENT-276: Restore LoRA adapter weights from APR checkpoint.
    ///
    /// Reads `lora.{layer}.{q,v}_proj.lora_{a,b}` tensors from the APR file
    /// and uploads them to the NF4 CUDA blocks, replacing the fresh random init.
    /// Returns (layers_restored, layers_total).
    pub fn restore_lora_from_apr(&mut self, apr_path: &std::path::Path) -> (usize, usize) {
        let reader = match aprender::serialization::apr::AprReader::open(apr_path) {
            Ok(r) => r,
            Err(_) => return (0, self.cuda_blocks.len()),
        };

        let mut restored = 0usize;
        for (i, block) in self.cuda_blocks.iter_mut().enumerate() {
            let a_q =
                reader.read_tensor_f32(&format!("lora.{i}.q_proj.lora_a")).unwrap_or_default();
            let b_q =
                reader.read_tensor_f32(&format!("lora.{i}.q_proj.lora_b")).unwrap_or_default();
            let a_v =
                reader.read_tensor_f32(&format!("lora.{i}.v_proj.lora_a")).unwrap_or_default();
            let b_v =
                reader.read_tensor_f32(&format!("lora.{i}.v_proj.lora_b")).unwrap_or_default();

            if a_q.is_empty() {
                continue; // No LoRA data for this layer in checkpoint
            }

            if let Err(e) = block.upload_lora_weights(&a_q, &b_q, &a_v, &b_v) {
                eprintln!("Warning: failed to restore LoRA for layer {i}: {e}");
                continue;
            }
            restored += 1;
        }

        (restored, self.cuda_blocks.len())
    }

    /// ALB-096: Load CPU embedding optimizer state from APR checkpoint.
    ///
    /// Reads `__training__.embed_optimizer.{m,v}.*` tensors from the APR file.
    /// Returns true if state was loaded.
    pub fn load_optimizer_state_apr(&mut self, apr_path: &std::path::Path) -> bool {
        let reader = match aprender::serialization::apr::AprReader::open(apr_path) {
            Ok(r) => r,
            Err(_) => return false,
        };

        // Restore step count from metadata
        if let Some(step_val) = reader.get_metadata("optimizer_step") {
            if let Some(step_str) = step_val.as_str() {
                if let Ok(step) = step_str.parse::<u64>() {
                    self.embed_optimizer.set_step_count(step);
                }
            }
        }

        // Restore first moments (m)
        for i in 0..128 {
            let name = format!("__training__.embed_optimizer.m.{i}");
            match reader.read_tensor_f32(&name) {
                Ok(data) if !data.is_empty() => {
                    self.embed_optimizer.set_first_moment(i, ndarray::Array1::from_vec(data));
                }
                _ => break,
            }
        }

        // Restore second moments (v)
        for i in 0..128 {
            let name = format!("__training__.embed_optimizer.v.{i}");
            match reader.read_tensor_f32(&name) {
                Ok(data) if !data.is_empty() => {
                    self.embed_optimizer.set_second_moment(i, ndarray::Array1::from_vec(data));
                }
                _ => break,
            }
        }

        // ALB-118: Restore GPU block optimizer states (m/v moments for all blocks)
        let suffixes = [
            "m.w_q",
            "v.w_q",
            "m.w_k",
            "v.w_k",
            "m.w_v",
            "v.w_v",
            "m.w_o",
            "v.w_o",
            "m.w_gate",
            "v.w_gate",
            "m.w_up",
            "v.w_up",
            "m.w_down",
            "v.w_down",
            "m.input_norm",
            "v.input_norm",
            "m.post_attn_norm",
            "v.post_attn_norm",
        ];
        let mut blocks_restored = 0usize;
        for (layer_idx, state) in self.gpu_training.optimizer_states.iter_mut().enumerate() {
            let mut data = std::collections::HashMap::new();
            for suffix in &suffixes {
                let name = format!("__training__.block_optimizer.{layer_idx}.{suffix}");
                if let Ok(tensor_data) = reader.read_tensor_f32(&name) {
                    if !tensor_data.is_empty() {
                        data.insert(suffix.to_string(), tensor_data);
                    }
                }
            }
            if !data.is_empty() {
                let _ = state.restore_from_host(&data);
                blocks_restored += 1;
            }
        }

        // ALB-118: Restore LM head optimizer state
        if let Ok(m_data) = reader.read_tensor_f32("__training__.lm_head_optimizer.m") {
            if m_data.len() == self.lm_head_m.len() {
                let _ = self.lm_head_m.copy_from_host(&m_data);
            }
        }
        if let Ok(v_data) = reader.read_tensor_f32("__training__.lm_head_optimizer.v") {
            if v_data.len() == self.lm_head_v.len() {
                let _ = self.lm_head_v.copy_from_host(&v_data);
            }
        }

        // ALB-118: Restore final norm optimizer state
        if let Ok(m_data) = reader.read_tensor_f32("__training__.final_norm_optimizer.m") {
            if m_data.len() == self.final_norm_m.len() {
                let _ = self.final_norm_m.copy_from_host(&m_data);
            }
        }
        if let Ok(v_data) = reader.read_tensor_f32("__training__.final_norm_optimizer.v") {
            if v_data.len() == self.final_norm_v.len() {
                let _ = self.final_norm_v.copy_from_host(&v_data);
            }
        }

        // ALB-132: Report restore results — don't silently swallow failures
        if blocks_restored > 0 {
            println!(
                "  ✓ GPU block optimizer states restored ({blocks_restored}/{} blocks)",
                self.gpu_training.optimizer_states.len()
            );
        } else if !self.gpu_training.optimizer_states.is_empty() {
            println!(
                "  [WARN] GPU block optimizer states NOT restored (0/{} blocks — zeroed m/v)",
                self.gpu_training.optimizer_states.len()
            );
        }

        true
    }

    /// R-001: Load CPU embedding optimizer state from `optimizer_state.json`.
    ///
    /// Returns true if state was loaded, false if file doesn't exist.
    pub fn load_optimizer_state(&mut self, dir: &std::path::Path) -> bool {
        let path = dir.join("optimizer_state.json");
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let state: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if let Some(step) = state["step"].as_u64() {
            self.embed_optimizer.set_step_count(step);
        }
        restore_moment_buffers(&state["m"], |idx, arr| {
            self.embed_optimizer.set_first_moment(idx, arr);
        });
        restore_moment_buffers(&state["v"], |idx, arr| {
            self.embed_optimizer.set_second_moment(idx, arr);
        });
        true
    }
}

/// ALB-096: Infer 2D tensor shape from name and element count.
///
/// Same logic as `infer_all_tensor_shapes` in `io/save.rs` but for a single tensor.
#[cfg(feature = "cuda")]
fn infer_tensor_shape(name: &str, numel: usize, hidden_size: usize) -> Vec<usize> {
    if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
        vec![numel]
    } else if hidden_size > 0 && numel.is_multiple_of(hidden_size) {
        let other_dim = numel / hidden_size;
        if name.ends_with("down_proj.weight") {
            vec![hidden_size, other_dim]
        } else {
            vec![other_dim, hidden_size]
        }
    } else {
        vec![numel]
    }
}

/// Parse a JSON array of moment buffers and apply each via callback.
#[cfg(feature = "cuda")]
fn restore_moment_buffers(
    json_arr: &serde_json::Value,
    mut set_fn: impl FnMut(usize, ndarray::Array1<f32>),
) {
    let Some(arr) = json_arr.as_array() else { return };
    for (idx, val) in arr.iter().enumerate() {
        let Some(inner) = val.as_array() else { continue };
        let floats: Vec<f32> = inner.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
        if !floats.is_empty() {
            set_fn(idx, ndarray::Array1::from_vec(floats));
        }
    }
}

// ── Non-CUDA stub ──

#[cfg(not(feature = "cuda"))]
pub struct CudaTransformerTrainer;

#[cfg(not(feature = "cuda"))]
impl CudaTransformerTrainer {
    pub fn new(_config: super::config::TransformerTrainConfig) -> crate::Result<Self> {
        Err(crate::error::Error::ConfigError(
            "CUDA not available (compiled without cuda feature)".into(),
        ))
    }

    pub fn with_model(
        _model: crate::transformer::Transformer,
        _config: super::config::TransformerTrainConfig,
    ) -> crate::Result<Self> {
        Err(crate::error::Error::ConfigError(
            "CUDA not available (compiled without cuda feature)".into(),
        ))
    }

    pub fn gpu_name(&self) -> String {
        unreachable!("CudaTransformerTrainer stub should never be instantiated")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_trainer_stub_returns_error() {
        use super::super::config::TransformerTrainConfig;
        use crate::transformer::TransformerConfig;

        let mc = TransformerConfig::tiny();
        let config = TransformerTrainConfig::new(mc);
        let result = super::CudaTransformerTrainer::new(config);
        assert!(result.is_err());
    }
}
