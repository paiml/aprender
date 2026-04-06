//! Instruction-following fine-tuning pipeline (GH-371)
//!
//! Wires Transformer + LoRA for causal language model fine-tuning on
//! instruction-response pairs.
//!
//! # Architecture
//!
//! ```text
//! [prompt_ids ++ response_ids] -> Transformer.forward() -> logits [seq_len, vocab_size]
//!   -> causal_lm_loss(logits[prompt_len..], response_ids) -> scalar loss
//! ```
//!
//! # Contract
//!
//! - F-INST-002: Loss computed only on response tokens (prompt tokens masked)
//! - F-INST-003: Perplexity = exp(avg_loss) reported per epoch
//! - F-INST-004: LoRA adapters saved in APR format

mod accessors;
mod backward;
mod constructors;
mod cuda_forward;
mod cuda_init;
mod generate;
mod training;
mod wgpu;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_cov3;
#[cfg(test)]
mod tests_cov3b;

use crate::lora::LoRALayer;
use crate::optim::{clip_grad_norm_refs, AdamW, Optimizer};
use crate::tokenizer::HfTokenizer;
use crate::train::transformer_trainer::step_profiler::StepProfiler;
use crate::transformer::{Transformer, TransformerConfig};
use crate::Tensor;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::CudaTrainer;
#[cfg(feature = "cuda")]
use crate::gpu::guard::VramGuard;
#[cfg(feature = "cuda")]
use crate::transformer::{
    CudaBlock, CudaBlockScratch, CudaLoraGradWorkspace, GpuLoraOptimizerState,
};
#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

/// Configuration for instruction fine-tuning.
#[derive(Debug, Clone)]
pub struct InstructConfig {
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Maximum sequence length (prompt + response)
    pub max_seq_len: usize,
    /// Maximum gradient norm for clipping
    pub gradient_clip_norm: Option<f32>,
    /// Quantize frozen weights to NF4 (4-bit) for QLoRA training (default: false).
    ///
    /// When enabled, uses `CudaNf4TransformerBlock` (~8x VRAM compression) instead
    /// of `CudaTransformerBlock`. GPU backward pass updates only LoRA adapters.
    pub quantize_nf4: bool,
}

impl Default for InstructConfig {
    fn default() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            learning_rate: 2e-4,
            epochs: 3,
            max_seq_len: 512,
            gradient_clip_norm: Some(1.0),
            quantize_nf4: false,
        }
    }
}

/// Result of processing one instruction-response pair.
#[derive(Debug, Clone)]
pub struct InstructStepResult {
    /// Cross-entropy loss on response tokens
    pub loss: f32,
    /// Number of response tokens
    pub num_response_tokens: usize,
    /// Perplexity = exp(loss)
    pub perplexity: f32,
}

/// Result of processing a mini-batch of instruction samples.
#[derive(Debug, Clone)]
pub struct InstructBatchResult {
    /// Average cross-entropy loss across the batch (response tokens only)
    pub avg_loss: f32,
    /// Total response tokens in batch
    pub total_response_tokens: usize,
    /// Perplexity = exp(avg_loss)
    pub perplexity: f32,
    /// Gradient norm before clipping
    pub grad_norm: f32,
}

/// Instruction fine-tuning pipeline.
///
/// Owns the transformer and LoRA adapters. Uses `Transformer::forward()`
/// for causal LM logits and computes loss on response tokens only.
/// GPU-resident training state for NF4 QLoRA backward pass.
///
/// Holds per-layer activation snapshots and scratch buffers needed for
/// activation checkpointing during NF4 backward.
#[cfg(feature = "cuda")]
pub(super) struct InstructGpuTrainingState {
    /// Saved input to each block during forward [num_layers][max_seq_len * hidden_size]
    layer_inputs: Vec<GpuBuffer<f32>>,
    /// Final RMSNorm weight uploaded to GPU [hidden_size]
    final_norm_weight: GpuBuffer<f32>,
    /// Blocks output saved on GPU for final norm backward [max_seq_len * hidden_size]
    blocks_output: GpuBuffer<f32>,
    /// Gradient scratch buffer A [max_seq_len * hidden_size]
    grad_buf_a: GpuBuffer<f32>,
    /// Gradient scratch buffer B [max_seq_len * hidden_size]
    grad_buf_b: GpuBuffer<f32>,
    /// Gradient for final RMSNorm weight [hidden_size]
    grad_final_norm_weight: GpuBuffer<f32>,
    embed_transposed: GpuBuffer<f32>, // [hidden*vocab] lm_head forward
    embed_original: GpuBuffer<f32>,   // [vocab*hidden] lm_head backward (KAIZEN-068)
    /// GPU scratch for logits [max_seq_len * vocab_size]
    logits_buf: GpuBuffer<f32>,
    /// GPU scratch for grad_hidden [max_seq_len * hidden_size]
    grad_hidden_buf: GpuBuffer<f32>,
    /// KAIZEN-045: Pre-allocated scratch buffer for activation checkpointing in backward
    output_scratch: GpuBuffer<f32>,
    /// KAIZEN-045: Pre-allocated upload buffer for gradient H2D transfer in backward
    grad_upload_buf: GpuBuffer<f32>,
    /// KAIZEN-062: Pre-allocated forward ping-pong buffer A
    fwd_scratch_a: GpuBuffer<f32>,
    /// KAIZEN-062: Pre-allocated forward ping-pong buffer B
    fwd_scratch_b: GpuBuffer<f32>,
    /// KAIZEN-062: Pre-allocated lm_head hidden input buffer
    lm_head_hidden_buf: GpuBuffer<f32>,
    /// PMAT-464: Cached CUDA graph for forward pass replay.
    forward_graph_exec: Option<trueno_gpu::driver::CudaGraphExec>,
    graph_cached_seq_len: usize,
    /// PMAT-488: Cached CUDA graph for backward pass replay.
    backward_graph_state: Option<super::backward_graph::BackwardGraphState>,
    /// PMAT-063: cuBLAS workspace buffer (must outlive CUDA graph)
    cublas_workspace: Option<GpuBuffer<f32>>,
    /// PMAT-483: Per-layer forward timing (microseconds per layer per step)
    profiler_layer_fwd_us: Vec<u64>,
    /// PMAT-483: Per-layer backward timing (microseconds per layer per step)
    profiler_layer_bwd_us: Vec<u64>,
    /// PMAT-483: Temporary layer start timestamp
    profiler_layer_start: Option<std::time::Instant>,
    /// PMAT-483/entrenar#328: Per-operation timing within layers (accumulated per step)
    /// Index matches StepProfiler::OP_* constants. Reset each step.
    profiler_op_us: [u64; 16],
    /// Per-operation start timestamp
    profiler_op_start: Option<std::time::Instant>,
}

pub struct InstructPipeline {
    /// Base transformer model
    pub model: Transformer,
    /// LoRA adapters applied to Q/V attention projections
    pub lora_layers: Vec<LoRALayer>,
    /// Pipeline configuration
    pub config: InstructConfig,
    /// AdamW optimizer for trainable parameters
    optimizer: AdamW,
    /// Optional BPE tokenizer
    tokenizer: Option<HfTokenizer>,
    /// Path to base model (for checkpoint provenance)
    model_dir: Option<PathBuf>,
    /// PMAT-483: Per-step profiler for scientific training measurement.
    /// Zero-overhead when disabled. Enable via --profile-interval N.
    pub profiler: StepProfiler,
    /// CUDA trainer for GPU memory management
    #[cfg(feature = "cuda")]
    cuda_trainer: Option<CudaTrainer>,
    /// CUDA-accelerated transformer blocks -- one per layer
    #[cfg(feature = "cuda")]
    cuda_blocks: Option<Vec<CudaBlock>>,
    /// Shared scratch buffers for NF4 forward pass
    #[cfg(feature = "cuda")]
    shared_scratch: Option<CudaBlockScratch>,
    /// Count of GPU forward passes that produced NaN/Inf
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    cuda_nan_count: usize,
    /// GPU training state for NF4 QLoRA backward pass
    #[cfg(feature = "cuda")]
    gpu_training: Option<InstructGpuTrainingState>,
    /// Shared LoRA gradient workspace for NF4 QLoRA backward
    #[cfg(feature = "cuda")]
    cuda_lora_grad_workspace: Option<CudaLoraGradWorkspace>,
    /// PMAT-477: Fused clip state -- zero D2H sync gradient clipping
    #[cfg(feature = "cuda")]
    lora_fused_clip: Option<crate::autograd::cuda_optim::FusedClipState>,
    /// Per-layer LoRA optimizer states for NF4 QLoRA training
    #[cfg(feature = "cuda")]
    cuda_lora_optimizer_states: Option<Vec<GpuLoraOptimizerState>>,
    /// NF4 LoRA optimizer step counter
    #[cfg(feature = "cuda")]
    nf4_lora_step: u32,
    /// VRAM reservation guard (GPU-SHARE-002). Releases ledger entry on Drop.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    vram_guard: Option<VramGuard>,
    /// wgpu training pipeline (zero unsafe alternative to CUDA)
    #[cfg(feature = "gpu")]
    wgpu_training: Option<WgpuTrainingState>,
}

/// State for wgpu-based training pipeline (WgpuTrainingPipeline)
#[cfg(feature = "gpu")]
struct WgpuTrainingState {
    /// GPU forward pass with persistent weight buffers + tiled GEMM
    fwd: trueno::backends::gpu::WgslForwardPass,
    cross_entropy: crate::autograd::wgpu_cross_entropy::WgslCrossEntropy,
    trainer: crate::autograd::wgpu_training::WgpuTrainer,
    // GPU buffers for logits, labels, losses, logsumexp
    logits_buf: trueno::backends::gpu::wgpu::Buffer,
    labels_buf: trueno::backends::gpu::wgpu::Buffer,
    losses_buf: trueno::backends::gpu::wgpu::Buffer,
    logsumexp_buf: trueno::backends::gpu::wgpu::Buffer,
    // Precomputed lm_head GPU buffers
    lm_head_gpu: trueno::backends::gpu::wgpu::Buffer,
    lm_head_t_gpu: trueno::backends::gpu::wgpu::Buffer,
    // Model config needed for forward pass
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
}

/// Configuration for autoregressive text generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of new tokens to generate (default: 256)
    pub max_new_tokens: usize,
    /// Sampling temperature (0.0 = greedy/argmax, >0 = stochastic)
    pub temperature: f32,
    /// Top-k filtering (0 = disabled, >0 = keep only top-k logits)
    pub top_k: usize,
    /// Additional stop token IDs (generation stops on EOS or any of these)
    pub stop_tokens: Vec<u32>,
}

/// Sample a token from logits with temperature and top-k filtering.
fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 || top_k == 1 {
        // Greedy: argmax
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32);
    }

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Top-k filtering
    let mut indices_and_logits: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indices_and_logits
        .sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let k = if top_k > 0 && top_k < indices_and_logits.len() {
        top_k
    } else {
        indices_and_logits.len()
    };
    let top = &indices_and_logits[..k];

    // Softmax over top-k
    let max_logit = top[0].1;
    let exps: Vec<f32> = top.iter().map(|(_, l)| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Sample from distribution (simple linear scan)
    let r: f32 = simple_random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return top[i].0 as u32;
        }
    }

    // Fallback to top-1
    top[0].0 as u32
}

/// Simple pseudo-random float in [0, 1) using thread-local state.
/// Not cryptographically secure but sufficient for sampling.
fn simple_random() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        );
    }
    STATE.with(|s| {
        // xorshift64
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}
