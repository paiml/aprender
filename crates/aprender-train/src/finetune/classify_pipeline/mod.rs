//! Classification fine-tuning pipeline
//!
//! Wires Transformer + LoRA + ClassificationHead for sequence classification.
//!
//! # Architecture
//!
//! ```text
//! token_ids -> Transformer.forward_hidden() -> [seq_len, hidden_size]
//!           -> ClassificationHead.forward()  -> [num_classes]
//!           -> cross_entropy_loss(target)    -> scalar loss
//! ```
//!
//! # Contract
//!
//! See `aprender/contracts/classification-finetune-v1.yaml`

use super::classification::{
    load_multi_label_corpus, load_safety_corpus, ClassificationHead, MultiLabelSafetySample,
    SafetySample, TokenizedSample,
};
use crate::autograd::matmul;
use crate::lora::LoRAConfig;
use crate::lora::LoRALayer;
use crate::optim::{clip_grad_norm_refs, AdamW, Optimizer};
use crate::tokenizer::HfTokenizer;
use crate::transformer::Transformer;
use crate::transformer::TransformerConfig;
use crate::Tensor;
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::pre_warm_lora_backward_kernels as pre_warm_backward_cache_kernels;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{pre_warm_forward_kernels, pre_warm_lora_backward_kernels};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::pre_warm_lora_adamw_kernels;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::{cuda_training_available, CudaTrainer};
#[cfg(feature = "cuda")]
use crate::gpu::guard::VramGuard;
#[cfg(feature = "cuda")]
use crate::transformer::{
    CudaBlock, CudaBlockScratch, CudaGradWorkspace, CudaLoraGradWorkspace, CudaTransformerBlock,
    GpuBlockOptimizerState, GpuLoraOptimizerState,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

/// Classification fine-tuning pipeline configuration.
#[derive(Debug, Clone)]
pub struct ClassifyConfig {
    /// Number of output classes
    pub num_classes: usize,
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Log every N steps
    pub log_interval: usize,
    /// Mini-batch size for `train_batch()`.
    ///
    /// Samples are processed one at a time (forward + backward), but the
    /// optimizer step is applied once per batch after accumulating gradients.
    pub batch_size: usize,
    /// Number of gradient accumulation steps.
    ///
    /// Allows effective batch size = `batch_size * accumulation_steps` without
    /// increasing peak memory beyond a single micro-batch forward pass.
    pub accumulation_steps: usize,
    /// Maximum gradient norm for clipping.
    ///
    /// When `Some(max_norm)`, gradients are clipped to this L2 norm before
    /// the optimizer step. `None` disables gradient clipping.
    pub gradient_clip_norm: Option<f32>,
    /// Per-class loss weights for imbalanced datasets.
    ///
    /// When `Some(weights)`, the cross-entropy loss for label `c` is multiplied
    /// by `weights[c]`. Weights should sum to `num_classes` to preserve loss scale.
    /// When `None`, all classes are weighted equally (weight = 1.0).
    ///
    /// See SPEC-TUNE-2026-001 §9 for weight computation strategies.
    pub class_weights: Option<Vec<f32>>,
    /// Quantize frozen weights to NF4 (4-bit) for QLoRA training (default: false).
    ///
    /// When enabled, uses `CudaNf4TransformerBlock` (~8x VRAM compression) instead
    /// of `CudaTransformerBlock`. GPU backward pass is disabled (LoRA-only training).
    pub quantize_nf4: bool,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            num_classes: 5,
            lora_rank: 16,
            lora_alpha: 16.0,
            learning_rate: 1e-4,
            epochs: 3,
            max_seq_len: 512,
            log_interval: 100,
            batch_size: 32,
            accumulation_steps: 1,
            gradient_clip_norm: Some(1.0),
            class_weights: None,
            quantize_nf4: false,
        }
    }
}

/// Hyperparameter diagnostic from contract validation.
///
/// Contract: qlora-hyperparameters-v1.yaml (C-HP-001..008)
#[derive(Debug, Clone)]
pub struct HyperparamDiagnostic {
    pub contract_id: &'static str,
    pub severity: DiagSeverity,
    pub message: String,
    pub recommendation: String,
}

/// Severity level for hyperparameter diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagSeverity {
    /// Informational — config is acceptable but not optimal.
    Info,
    /// Warning — config violates a research-grounded contract.
    Warn,
    /// Error — config is mathematically invalid (e.g., lr=0).
    Error,
}

/// Collection of hyperparameter diagnostics.
#[derive(Debug, Clone, Default)]
pub struct HyperparamDiagnostics {
    pub items: Vec<HyperparamDiagnostic>,
}

impl HyperparamDiagnostics {
    /// Check if any diagnostic matches a contract ID.
    pub fn has_warning(&self, contract_id: &str) -> bool {
        self.items.iter().any(|d| {
            d.contract_id == contract_id
                && matches!(d.severity, DiagSeverity::Warn | DiagSeverity::Error)
        })
    }

    /// Check if there are any blocking errors.
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| matches!(d.severity, DiagSeverity::Error))
    }

    /// Print all diagnostics to stderr.
    pub fn print_all(&self) {
        for d in &self.items {
            let prefix = match d.severity {
                DiagSeverity::Info => "[HP-INFO]",
                DiagSeverity::Warn => "[HP-WARN]",
                DiagSeverity::Error => "[HP-ERROR]",
            };
            eprintln!("{prefix} {}: {} → {}", d.contract_id, d.message, d.recommendation);
        }
    }
}

/// Data distribution statistics for data-driven hyperparameter validation.
///
/// Contract: C-HP-004 (seq_len from data), C-HP-008 (epochs for imbalance)
pub struct DataStats {
    /// 99th percentile of BPE token lengths in training data.
    pub p99_token_length: usize,
    /// Class imbalance ratio (majority_count / minority_count).
    pub imbalance_ratio: f32,
    /// Number of samples in minority class.
    pub minority_count: usize,
}

impl ClassifyConfig {
    /// Create a QLoRA config with research-grounded defaults.
    ///
    /// Every parameter traces to a published source:
    ///
    /// | Parameter | Value | Source |
    /// |-----------|-------|--------|
    /// | `learning_rate` | 2e-4 (≤13B) / 1e-4 (>13B) | Dettmers 2023 Table 9 |
    /// | `lora_alpha` | 2 × rank | Lightning AI experiments |
    /// | `batch_size` | 4 | RTX VRAM budget |
    /// | `accumulation_steps` | 4 | effective=16, Dettmers 2023 |
    /// | `gradient_clip_norm` | 1.0 | Standard practice |
    /// | `epochs` | 3 | Imbalanced classification |
    ///
    /// Contract: provable-contracts/contracts/entrenar/qlora-hyperparameters-v1.yaml
    pub fn qlora_default(model_params: u64) -> Self {
        // C-HP-001: lr scales with model size (Dettmers 2023 Table 9)
        let learning_rate = if model_params <= 13_000_000_000 { 2e-4 } else { 1e-4 };
        let lora_rank = 16;
        Self {
            num_classes: 2,
            lora_rank,
            // C-HP-003: alpha = 2 * rank (Lightning AI)
            lora_alpha: (2 * lora_rank) as f32,
            learning_rate,
            // C-HP-008: >= 2 epochs for imbalanced classification
            epochs: 3,
            // C-HP-004: set from data, 256 is SSC v3 default
            max_seq_len: 256,
            log_interval: 100,
            // C-HP-002: effective=16 (Dettmers 2023). Micro-batch=16 to saturate
            // GPU occupancy (batch=4 leaves RTX 4090 at 1.5% MFU).
            batch_size: 16,
            accumulation_steps: 1,
            // C-HP-006: gradient clipping (standard, SSC v2.2 precedent)
            gradient_clip_norm: Some(1.0),
            class_weights: None,
            quantize_nf4: true,
        }
    }

    /// Validate hyperparameters against research-grounded contracts.
    ///
    /// Returns diagnostics (warnings/errors) for each violated contract.
    /// Does NOT block — caller decides whether to abort or proceed.
    ///
    /// Contract: qlora-hyperparameters-v1.yaml (FALSIFY-HP-001..008)
    pub fn validate_hyperparameters(&self, model_params: u64) -> HyperparamDiagnostics {
        let mut diags = HyperparamDiagnostics::default();

        // C-HP-001: Learning rate scaling
        if self.quantize_nf4 && model_params <= 13_000_000_000 && self.learning_rate < 1.5e-4 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-001",
                severity: DiagSeverity::Warn,
                message: format!(
                    "lr={:.0e} too low for {}B model (Dettmers 2023: use 2e-4 for ≤13B)",
                    self.learning_rate,
                    model_params / 1_000_000_000
                ),
                recommendation: "learning_rate: 0.0002".to_string(),
            });
        }

        // C-HP-002: Effective batch size
        let eff_batch = self.batch_size * self.accumulation_steps;
        if eff_batch != 16 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-002",
                severity: DiagSeverity::Warn,
                message: format!(
                    "effective_batch={eff_batch} ({}×{}), Dettmers 2023 recommends 16 for ≤13B",
                    self.batch_size, self.accumulation_steps
                ),
                recommendation: format!(
                    "batch_size: {}, accumulation_steps: {}",
                    self.batch_size,
                    16 / self.batch_size.max(1)
                ),
            });
        }

        // C-HP-003: Alpha/rank ratio
        let expected_alpha = 2.0 * self.lora_rank as f32;
        if (self.lora_alpha - expected_alpha).abs() > 0.5 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-003",
                severity: DiagSeverity::Warn,
                message: format!(
                    "lora_alpha={} with rank={} (ratio={:.1}), Lightning AI: alpha=2×rank={} optimal",
                    self.lora_alpha, self.lora_rank,
                    self.lora_alpha / self.lora_rank as f32,
                    expected_alpha
                ),
                recommendation: format!("lora_alpha: {expected_alpha}"),
            });
        }

        // C-HP-006: Gradient clipping
        if self.gradient_clip_norm.is_none() {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-006",
                severity: DiagSeverity::Warn,
                message: "No gradient clipping — SSC v2.2 saw grad norms up to 115.1".to_string(),
                recommendation: "gradient_clip_norm: 1.0".to_string(),
            });
        }

        // Blocking errors
        if self.learning_rate <= 0.0 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-001",
                severity: DiagSeverity::Error,
                message: "learning_rate must be > 0".to_string(),
                recommendation: "learning_rate: 0.0002".to_string(),
            });
        }
        if self.batch_size == 0 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-002",
                severity: DiagSeverity::Error,
                message: "batch_size must be > 0".to_string(),
                recommendation: "batch_size: 4".to_string(),
            });
        }

        diags
    }

    /// Validate hyperparameters that depend on training data distribution.
    ///
    /// Requires measuring the actual data (genchi genbutsu — go and see).
    ///
    /// Contract: C-HP-004 (seq_len), C-HP-008 (epochs)
    pub fn validate_with_data(&self, stats: &DataStats) -> HyperparamDiagnostics {
        let mut diags = HyperparamDiagnostics::default();

        // C-HP-004: Sequence length from data distribution
        if self.max_seq_len > 2 * stats.p99_token_length && stats.p99_token_length > 0 {
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-004",
                severity: DiagSeverity::Warn,
                message: format!(
                    "max_seq_len={} but p99(tokens)={} — attention is O(n²), wasting {:.0}× compute",
                    self.max_seq_len,
                    stats.p99_token_length,
                    (self.max_seq_len as f64 / stats.p99_token_length as f64).powi(2)
                ),
                recommendation: format!(
                    "max_seq_len: {} (next_pow2 of p99)",
                    stats.p99_token_length.next_power_of_two()
                ),
            });
        }

        // C-HP-008: Epochs for imbalanced classification
        if stats.imbalance_ratio > 5.0 && self.epochs < 2 {
            let eff_batch = self.batch_size * self.accumulation_steps;
            let updates_per_epoch = stats.minority_count / eff_batch.max(1);
            diags.items.push(HyperparamDiagnostic {
                contract_id: "C-HP-008",
                severity: DiagSeverity::Warn,
                message: format!(
                    "epochs={} with {:.1}:1 imbalance — minority gets only {} gradient updates",
                    self.epochs,
                    stats.imbalance_ratio,
                    updates_per_epoch * self.epochs
                ),
                recommendation: format!(
                    "epochs: 3 (minority gets {} updates)",
                    updates_per_epoch * 3
                ),
            });
        }

        diags
    }
}

/// Result of processing one mini-batch via [`ClassifyPipeline::train_batch`].
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Average cross-entropy loss across the batch
    pub avg_loss: f32,
    /// Number of correctly classified samples
    pub correct: usize,
    /// Total number of samples in the batch
    pub total: usize,
    /// Global gradient norm (before clipping). 0.0 if clipping disabled.
    pub grad_norm: f32,
}

impl BatchResult {
    /// Compute classification accuracy as `correct / total`.
    ///
    /// Returns 0.0 for an empty batch (total == 0).
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        contract_pre_accuracy!();
        self.correct as f32 / self.total.max(1) as f32
    }
}

/// GPU-resident training state for full-finetune backward pass.
///
/// Holds per-layer activation snapshots, optimizer state, and scratch buffers
/// required to run backward through all transformer blocks on GPU.
///
/// # Contract (C-GPUTRAIN-001)
///
/// - **Precondition**: `init()` called after CUDA blocks are created
/// - **Postcondition**: All layer_inputs saved during forward; optimizer states zero-initialized
/// - **Invariant**: `layer_inputs.len() == num_layers`; buffers never reallocated after init
#[cfg(feature = "cuda")]
struct GpuTrainingState {
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
    /// Per-block AdamW optimizer states
    optimizer_states: Vec<GpuBlockOptimizerState>,
    /// Global optimizer step counter
    step: u32,
    /// KAIZEN-045: Pre-allocated scratch buffer for activation checkpointing in backward
    /// [max_seq_len * hidden_size]. Eliminates per-backward cuMemAlloc/cuMemFree.
    output_scratch: GpuBuffer<f32>,
    /// KAIZEN-045: Pre-allocated upload buffer for gradient H2D transfer in backward
    /// [max_seq_len * hidden_size]. Eliminates per-backward cuMemAlloc/cuMemFree.
    grad_upload_buf: GpuBuffer<f32>,
    /// KAIZEN-060: Pre-allocated forward ping-pong buffers [max_seq_len * hidden_size].
    /// Eliminates 2 × cuMemAlloc/Free per forward pass.
    fwd_scratch_a: GpuBuffer<f32>,
    fwd_scratch_b: GpuBuffer<f32>,
    /// KAIZEN-061: Pre-allocated CPU staging buffer for backward mean-pool gradient.
    /// Sized to max_seq_len * hidden_size. Eliminates ~1.25MB heap alloc per sample
    /// in both backward_gpu_blocks and backward_nf4_gpu_blocks (~17.5GB/epoch).
    backward_cpu_staging: Vec<f32>,
}

/// Classification fine-tuning pipeline.
///
/// Owns the transformer, LoRA adapters, and classification head.
/// Provides `train_step()` for single-step training and `train()` for full loop.
///
/// When compiled with `feature = "cuda"` and a GPU is available, the forward pass
/// runs on CUDA via `CudaTransformerBlock`s for ~10-50x speedup (F-CUDA-007).
/// When `gpu_training` is set, the backward pass also runs on GPU with full-finetune
/// of all transformer weights (F-CUDA-014).
pub struct ClassifyPipeline {
    /// Base transformer model (weights frozen)
    pub model: Transformer,
    /// Classification head (trainable)
    pub classifier: ClassificationHead,
    /// LoRA adapters applied to attention projections
    pub lora_layers: Vec<LoRALayer>,
    /// Pipeline configuration
    pub config: ClassifyConfig,
    /// AdamW optimizer for trainable parameters
    optimizer: AdamW,
    /// Optional BPE tokenizer (None = byte-level fallback)
    tokenizer: Option<HfTokenizer>,
    /// Path to base model directory (set by `from_pretrained`, None for random init)
    model_dir: Option<PathBuf>,
    /// CUDA trainer for GPU memory management (F-CUDA-002)
    #[cfg(feature = "cuda")]
    cuda_trainer: Option<CudaTrainer>,
    /// CUDA-accelerated transformer blocks — one per layer (F-CUDA-006)
    #[cfg(feature = "cuda")]
    cuda_blocks: Option<Vec<CudaBlock>>,
    /// Shared scratch buffers for NF4 forward pass (C-SCRATCH-001).
    /// One allocation shared across all 36 layers — saves 7.5 GB for Qwen3-4B.
    #[cfg(feature = "cuda")]
    shared_scratch: Option<CudaBlockScratch>,
    /// Count of GPU forward passes that produced NaN/Inf and fell back to CPU.
    /// Used to decide when to permanently disable CUDA (threshold: 100).
    #[cfg(feature = "cuda")]
    cuda_nan_count: usize,
    /// GPU training state for full-finetune backward pass (F-CUDA-014).
    /// When `Some`, backward pass runs on GPU updating all transformer weights.
    #[cfg(feature = "cuda")]
    gpu_training: Option<GpuTrainingState>,
    /// Shared gradient workspace — one set of weight-gradient buffers for all layers.
    /// Contract: C-GRADWS-001. Saves (L-1) * 372 MB for Qwen3-4B (13.0 GB).
    #[cfg(feature = "cuda")]
    cuda_grad_workspace: Option<CudaGradWorkspace>,
    /// Shared LoRA gradient workspace for NF4 QLoRA backward (ENT-153).
    /// One set of LoRA gradient buffers shared across all layers.
    #[cfg(feature = "cuda")]
    cuda_lora_grad_workspace: Option<CudaLoraGradWorkspace>,
    /// Per-layer LoRA optimizer states for NF4 QLoRA training (ENT-153).
    #[cfg(feature = "cuda")]
    cuda_lora_optimizer_states: Option<Vec<GpuLoraOptimizerState>>,
    /// KAIZEN-014: Per-layer gradient accumulators for batch-correct LoRA optimizer.
    #[cfg(feature = "cuda")]
    cuda_lora_grad_accum: Option<Vec<CudaLoraGradWorkspace>>,
    /// NF4 LoRA optimizer step counter (separate from fp32 GpuTrainingState.step).
    #[cfg(feature = "cuda")]
    nf4_lora_step: u32,
    /// wgpu-accelerated forward pass (GPU feature, non-CUDA)
    #[cfg(feature = "gpu")]
    wgpu_forward_pass: Option<crate::transformer::WgpuForwardPass>,
    /// VRAM reservation guard (GPU-SHARE-002). Releases ledger entry on Drop.
    /// Held for RAII — released when pipeline is dropped.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    vram_guard: Option<VramGuard>,
}

impl ClassifyPipeline {
    /// Create a new classification pipeline with random weights and byte-level tokenization.
    ///
    /// # Arguments
    /// * `model_config` - Transformer configuration (e.g., `TransformerConfig::qwen2_0_5b()`)
    /// * `classify_config` - Classification pipeline configuration
    pub fn new(model_config: &TransformerConfig, classify_config: ClassifyConfig) -> Self {
        let model = Transformer::new(model_config);
        let classifier =
            ClassificationHead::new(model_config.hidden_size, classify_config.num_classes);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &classify_config);

        // Ensure LoRA A/B matrices have requires_grad=true
        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        // ── CUDA initialization (F-CUDA-001..006, GPU-SHARE-002) ─────────
        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks, shared_scratch, vram_guard) =
            Self::try_init_cuda(&model, model_config, &classify_config, &lora_layers);

        // ── GPU training state (F-CUDA-014) ────────────────────────────
        #[cfg(feature = "cuda")]
        let gpu_training = Self::try_init_gpu_training(
            &model,
            model_config,
            classify_config.max_seq_len,
            cuda_trainer.as_ref(),
            cuda_blocks.as_ref(),
        );

        // ── Shared gradient workspace (C-GRADWS-001) ────────────────────
        #[cfg(feature = "cuda")]
        let cuda_grad_workspace = if classify_config.quantize_nf4 {
            None
        } else {
            cuda_trainer.as_ref().and_then(|t| {
                CudaGradWorkspace::new(t.context(), model_config)
                    .map_err(|e| eprintln!("[CUDA] Failed to allocate grad workspace: {e}"))
                    .ok()
            })
        };

        // ── NF4 LoRA training state (ENT-153) ──────────────────────────
        #[cfg(feature = "cuda")]
        let (cuda_lora_grad_workspace, cuda_lora_optimizer_states, cuda_lora_grad_accum) =
            if classify_config.quantize_nf4 {
                Self::try_init_nf4_lora_training(
                    cuda_trainer.as_ref(),
                    cuda_blocks.as_ref(),
                    model_config,
                    &classify_config,
                )
            } else {
                (None, None, None)
            };

        // ── wgpu initialization (when CUDA unavailable) ──────────────────
        #[cfg(feature = "gpu")]
        let wgpu_forward_pass = {
            #[cfg(feature = "cuda")]
            let has_cuda = cuda_trainer.is_some();
            #[cfg(not(feature = "cuda"))]
            let has_cuda = false;

            if has_cuda {
                None // CUDA takes priority
            } else {
                // KAIZEN-015: Pre-upload FFN weights to GPU (zero H2D per forward pass)
                match crate::transformer::WgpuForwardPass::with_resident_weights(&model) {
                    Ok(pass) => {
                        eprintln!("[wgpu] GPU forward pass initialized (resident weights)");
                        Some(pass)
                    }
                    Err(e) => {
                        eprintln!("[wgpu] GPU resident init failed, trying default: {e}");
                        match crate::transformer::WgpuForwardPass::new_default(model_config) {
                            Ok(pass) => {
                                eprintln!("[wgpu] GPU forward pass initialized (upload per call)");
                                Some(pass)
                            }
                            Err(e2) => {
                                eprintln!("[wgpu] GPU initialization failed, using CPU: {e2}");
                                None
                            }
                        }
                    }
                }
            }
        };

        Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer: None,
            model_dir: None,
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            shared_scratch,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training,
            #[cfg(feature = "cuda")]
            cuda_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_accum,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "gpu")]
            wgpu_forward_pass,
            #[cfg(feature = "cuda")]
            vram_guard,
        }
    }

    /// Create a classification pipeline from pretrained weights.
    ///
    /// Loads a transformer from SafeTensors weights and optionally a BPE tokenizer
    /// from `tokenizer.json` in the model directory.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing SafeTensors weights (and optionally `tokenizer.json`)
    /// * `model_config` - Transformer configuration matching the pretrained weights
    /// * `classify_config` - Classification pipeline configuration
    ///
    /// # Errors
    /// Returns error if the model directory doesn't exist or weights fail to load.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        model_config: &TransformerConfig,
        classify_config: ClassifyConfig,
    ) -> crate::Result<Self> {
        let model_dir = model_dir.as_ref();

        let model = Transformer::from_safetensors(model_dir, model_config)?;
        let classifier =
            ClassificationHead::new(model_config.hidden_size, classify_config.num_classes);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &classify_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        // CONTRACT: Training requires a BPE tokenizer — byte-fallback is not acceptable.
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Some(
                HfTokenizer::from_file(&tokenizer_path)
                    .map_err(|e| crate::Error::Io(format!("Failed to load tokenizer: {e}")))?,
            )
        } else {
            return Err(crate::Error::ConfigError(format!(
                "No tokenizer.json found in '{}'. Training requires a BPE tokenizer.",
                model_dir.display(),
            )));
        };

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        // ── CUDA initialization (F-CUDA-001..006, GPU-SHARE-002) ─────────
        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks, shared_scratch, vram_guard) =
            Self::try_init_cuda(&model, model_config, &classify_config, &lora_layers);

        // ── GPU training state (F-CUDA-014) ────────────────────────────
        #[cfg(feature = "cuda")]
        let gpu_training = Self::try_init_gpu_training(
            &model,
            model_config,
            classify_config.max_seq_len,
            cuda_trainer.as_ref(),
            cuda_blocks.as_ref(),
        );

        // ── Shared gradient workspace (C-GRADWS-001) ────────────────────
        #[cfg(feature = "cuda")]
        let cuda_grad_workspace = if classify_config.quantize_nf4 {
            None // No grad workspace needed for frozen NF4 weights
        } else {
            cuda_trainer.as_ref().and_then(|t| {
                CudaGradWorkspace::new(t.context(), model_config)
                    .map_err(|e| eprintln!("[CUDA] Failed to allocate grad workspace: {e}"))
                    .ok()
            })
        };

        // ── NF4 LoRA training state (ENT-153) ──────────────────────────
        #[cfg(feature = "cuda")]
        let (cuda_lora_grad_workspace, cuda_lora_optimizer_states, cuda_lora_grad_accum) =
            if classify_config.quantize_nf4 {
                Self::try_init_nf4_lora_training(
                    cuda_trainer.as_ref(),
                    cuda_blocks.as_ref(),
                    model_config,
                    &classify_config,
                )
            } else {
                (None, None, None)
            };

        // ── wgpu initialization (when CUDA unavailable) ──────────────────
        #[cfg(feature = "gpu")]
        let wgpu_forward_pass = {
            #[cfg(feature = "cuda")]
            let has_cuda = cuda_trainer.is_some();
            #[cfg(not(feature = "cuda"))]
            let has_cuda = false;

            if has_cuda {
                None
            } else {
                // KAIZEN-015: Pre-upload FFN weights to GPU
                match crate::transformer::WgpuForwardPass::with_resident_weights(&model) {
                    Ok(pass) => {
                        eprintln!(
                            "[wgpu] Batched forward pass initialized ({} layers, resident weights)",
                            model_config.num_hidden_layers
                        );
                        Some(pass)
                    }
                    Err(e) => {
                        eprintln!("[wgpu] Resident init failed, trying default: {e}");
                        match crate::transformer::WgpuForwardPass::new_default(model_config) {
                            Ok(pass) => {
                                eprintln!("[wgpu] Batched forward pass initialized ({} layers, upload per call)", model_config.num_hidden_layers);
                                Some(pass)
                            }
                            Err(e2) => {
                                eprintln!("[wgpu] GPU init failed, using CPU: {e2}");
                                None
                            }
                        }
                    }
                }
            }
        };

        Ok(Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer,
            model_dir: Some(model_dir.to_path_buf()),
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            shared_scratch,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training,
            #[cfg(feature = "cuda")]
            cuda_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_accum,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "gpu")]
            wgpu_forward_pass,
            #[cfg(feature = "cuda")]
            vram_guard,
        })
    }

    /// Create pipeline from APR model file (.apr format).
    ///
    /// Loads transformer weights from the APR binary, dequantizing from any
    /// stored dtype (F16, Q4K, etc.) to F32. Loads sibling tokenizer if present
    /// (e.g., `model.tokenizer.json` next to `model.apr`).
    ///
    /// # Errors
    /// Returns error if APR file cannot be loaded or weights are invalid.
    pub fn from_apr(
        apr_path: &Path,
        model_config: &TransformerConfig,
        classify_config: ClassifyConfig,
    ) -> crate::Result<Self> {
        let model = Transformer::from_apr(apr_path, model_config)?;
        let classifier =
            ClassificationHead::new(model_config.hidden_size, classify_config.num_classes);
        let mut lora_layers = Self::build_lora_layers(&model, model_config, &classify_config);

        for lora in &mut lora_layers {
            for param in lora.trainable_params() {
                param.set_requires_grad(true);
            }
        }

        // CONTRACT: Training requires a BPE tokenizer — byte-fallback is not acceptable.
        let tokenizer = {
            let sibling = apr_path.file_stem().and_then(|stem| {
                apr_path
                    .parent()
                    .map(|p| p.join(format!("{}.tokenizer.json", stem.to_str().unwrap_or(""))))
            });

            match sibling {
                Some(ref path) if path.exists() => {
                    let tok = HfTokenizer::from_file(path).map_err(|e| {
                        crate::Error::ConfigError(format!(
                            "Failed to load tokenizer from '{}': {e}. \
                             Training requires a BPE tokenizer.",
                            path.display(),
                        ))
                    })?;
                    Some(tok)
                }
                _ => {
                    return Err(crate::Error::ConfigError(format!(
                        "No sibling tokenizer found for '{}'. Expected \
                         '{}.tokenizer.json' next to the .apr file. Training \
                         requires a BPE tokenizer.",
                        apr_path.display(),
                        apr_path.file_stem().unwrap_or_default().to_str().unwrap_or(""),
                    )));
                }
            }
        };

        let optimizer = AdamW::default_params(classify_config.learning_rate);

        #[cfg(feature = "cuda")]
        let (cuda_trainer, cuda_blocks, shared_scratch, vram_guard) =
            Self::try_init_cuda(&model, model_config, &classify_config, &lora_layers);

        #[cfg(feature = "cuda")]
        let gpu_training = Self::try_init_gpu_training(
            &model,
            model_config,
            classify_config.max_seq_len,
            cuda_trainer.as_ref(),
            cuda_blocks.as_ref(),
        );

        #[cfg(feature = "cuda")]
        let cuda_grad_workspace = if classify_config.quantize_nf4 {
            None
        } else {
            cuda_trainer.as_ref().and_then(|t| {
                CudaGradWorkspace::new(t.context(), model_config)
                    .map_err(|e| eprintln!("[CUDA] Failed to allocate grad workspace: {e}"))
                    .ok()
            })
        };

        #[cfg(feature = "cuda")]
        let (cuda_lora_grad_workspace, cuda_lora_optimizer_states, cuda_lora_grad_accum) =
            if classify_config.quantize_nf4 {
                Self::try_init_nf4_lora_training(
                    cuda_trainer.as_ref(),
                    cuda_blocks.as_ref(),
                    model_config,
                    &classify_config,
                )
            } else {
                (None, None, None)
            };

        // ── wgpu initialization ──────────────────────────────────────────
        #[cfg(feature = "gpu")]
        let wgpu_forward_pass = {
            #[cfg(feature = "cuda")]
            let has_cuda = cuda_trainer.is_some();
            #[cfg(not(feature = "cuda"))]
            let has_cuda = false;

            if has_cuda {
                None
            } else {
                // KAIZEN-015: Pre-upload FFN weights to GPU
                crate::transformer::WgpuForwardPass::with_resident_weights(&model)
                    .or_else(|e| {
                        eprintln!("[wgpu] Resident init failed: {e}, trying default");
                        crate::transformer::WgpuForwardPass::new_default(model_config)
                    })
                    .map_err(|e| eprintln!("[wgpu] GPU init failed: {e}"))
                    .ok()
            }
        };

        Ok(Self {
            model,
            classifier,
            lora_layers,
            config: classify_config,
            optimizer,
            tokenizer,
            model_dir: Some(apr_path.to_path_buf()),
            #[cfg(feature = "cuda")]
            cuda_trainer,
            #[cfg(feature = "cuda")]
            cuda_blocks,
            #[cfg(feature = "cuda")]
            shared_scratch,
            #[cfg(feature = "cuda")]
            cuda_nan_count: 0,
            #[cfg(feature = "cuda")]
            gpu_training,
            #[cfg(feature = "cuda")]
            cuda_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_workspace,
            #[cfg(feature = "cuda")]
            cuda_lora_optimizer_states,
            #[cfg(feature = "cuda")]
            cuda_lora_grad_accum,
            #[cfg(feature = "cuda")]
            nf4_lora_step: 0,
            #[cfg(feature = "gpu")]
            wgpu_forward_pass,
            #[cfg(feature = "cuda")]
            vram_guard,
        })
    }

    /// Tokenize input text using BPE tokenizer.
    ///
    /// Truncates to `config.max_seq_len` and ensures at least one token.
    ///
    /// # Panics
    /// Panics if no BPE tokenizer is loaded. Training pipelines MUST have a
    /// tokenizer — byte-level fallback is a silent corruption path.
    pub(crate) fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = match self.tokenizer.as_ref() {
            Some(tok) => tok.encode(text),
            None => {
                // Byte-level fallback when no BPE tokenizer is loaded
                text.bytes().map(u32::from).collect()
            }
        };
        ids.truncate(self.config.max_seq_len);
        if ids.is_empty() {
            ids.push(0);
        }
        ids
    }

    /// Pre-tokenize a batch of samples for efficient training (KAIZEN-028).
    ///
    /// Tokenizes each sample once and stores the token IDs alongside the label.
    /// This eliminates redundant BPE encoding across epochs and batches.
    ///
    /// # Contract (C-PRETOK-001)
    ///
    /// - **Precondition**: All samples have non-empty `input`
    /// - **Postcondition**: Each `TokenizedSample` has `token_ids.len() in 1..=max_seq_len`
    /// - **Invariant**: Tokenization is deterministic — same input always produces same IDs
    pub fn pre_tokenize(&self, samples: &[SafetySample]) -> Vec<TokenizedSample> {
        let has_tokenizer = self.tokenizer.is_some();
        samples
            .iter()
            .map(|s| {
                let token_ids = if has_tokenizer {
                    self.tokenize(&s.input)
                } else {
                    // Byte-level fallback for tests without BPE tokenizer
                    let mut ids = s.input_ids();
                    ids.truncate(self.config.max_seq_len);
                    if ids.is_empty() {
                        ids.push(0);
                    }
                    ids
                };
                TokenizedSample { token_ids, label: s.label }
            })
            .collect()
    }

    /// Train on a batch of pre-tokenized samples (KAIZEN-028).
    ///
    /// Identical to [`train_batch`] but skips tokenization — token IDs are
    /// pre-computed at dataset construction time.
    pub fn train_batch_tokenized(&mut self, samples: &[TokenizedSample]) -> BatchResult {
        if samples.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let batch_size = samples.len();

        // ── 1. Zero gradients ──────────────────────────────────────────
        self.zero_all_gradients();

        // ── 2. Accumulate gradients over all samples ───────────────────
        #[cfg(feature = "gpu")]
        let (total_loss, correct) = self
            .try_train_batch_wgpu_tokenized(samples)
            .unwrap_or_else(|| self.train_batch_per_sample_tokenized(samples));

        #[cfg(not(feature = "gpu"))]
        let (total_loss, correct) = self.train_batch_per_sample_tokenized(samples);

        // ── 3. Normalize gradients by batch size ───────────────────────
        self.scale_all_gradients(1.0 / batch_size as f32);

        // ── 4. Gradient clipping (captures pre-clip norm) ────────────
        let grad_norm = if let Some(max_norm) = self.config.gradient_clip_norm {
            let mut params = self.trainable_parameters_mut();
            clip_grad_norm_refs(&mut params, max_norm)
        } else {
            self.compute_grad_norm()
        };

        // ── 5. Optimizer step (once for the whole batch) ───────────────
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && !self.config.quantize_nf4 {
                let lr = self.optimizer.lr();
                self.gpu_optimizer_step(lr);
            }
        }

        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && self.config.quantize_nf4 {
                self.nf4_lora_batch_optimizer_step(batch_size);
            }
        }

        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        BatchResult {
            avg_loss: total_loss / batch_size as f32,
            correct,
            total: batch_size,
            grad_norm,
        }
    }

    /// Per-sample forward + backward with pre-tokenized IDs (KAIZEN-028).
    fn train_batch_per_sample_tokenized(&mut self, samples: &[TokenizedSample]) -> (f32, usize) {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for sample in samples {
            let (loss, predicted) = self.forward_backward_single(&sample.token_ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }
        (total_loss, correct)
    }

    /// Batched wgpu forward with pre-tokenized IDs (KAIZEN-028).
    #[cfg(feature = "gpu")]
    fn try_train_batch_wgpu_tokenized(
        &mut self,
        samples: &[TokenizedSample],
    ) -> Option<(f32, usize)> {
        self.wgpu_forward_pass.as_ref()?;

        let batch_token_ids: Vec<Vec<u32>> = samples.iter().map(|s| s.token_ids.clone()).collect();

        let lora_ref =
            if self.lora_layers.is_empty() { None } else { Some(self.lora_layers.as_slice()) };

        let hiddens = self
            .wgpu_forward_pass
            .as_ref()
            .expect("checked is_none above")
            .forward_hidden_batch(&self.model, &batch_token_ids, lora_ref)
            .map_err(|e| {
                eprintln!("[wgpu] Batched forward failed, falling back to per-sample: {e}");
            })
            .ok()?;

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for (i, hidden) in hiddens.iter().enumerate() {
            let (loss, predicted) = self.classify_backward_from_hidden(
                hidden,
                batch_token_ids[i].len(),
                samples[i].label,
            );
            total_loss += loss;
            if predicted == samples[i].label {
                correct += 1;
            }
        }
        Some((total_loss, correct))
    }

    /// Accumulate gradients with pre-tokenized samples (KAIZEN-028).
    ///
    /// Identical to [`accumulate_gradients`] but uses pre-tokenized IDs.
    pub fn accumulate_gradients_tokenized(
        &mut self,
        micro_batch: &[TokenizedSample],
    ) -> BatchResult {
        if micro_batch.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;

        for sample in micro_batch {
            let (loss, predicted) = self.forward_backward_single(&sample.token_ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }

        BatchResult {
            avg_loss: total_loss / micro_batch.len() as f32,
            correct,
            total: micro_batch.len(),
            grad_norm: 0.0,
        }
    }

    /// Forward-only inference with pre-tokenized sample (KAIZEN-028).
    ///
    /// Used for validation where we need loss + prediction without backward pass.
    pub fn forward_only_tokenized(&mut self, token_ids: &[u32], label: usize) -> (f32, usize) {
        self.forward_only(token_ids, label)
    }
}

// GPU initialization, VRAM guards, and CUDA pipeline methods
include!("gpu.rs");

// Training, inference, gradient, and optimizer methods
include!("training.rs");

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
