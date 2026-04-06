//! Configuration for transformer training

use crate::autograd::{CheckpointConfig, MixedPrecisionConfig};
use crate::train::TrainConfig;
use crate::transformer::TransformerConfig;
use std::net::SocketAddr;

/// Role of a node in distributed pretraining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistributedRole {
    /// Coordinates training: AllReduces gradients, manages checkpoints
    #[default]
    Coordinator,
    /// Computes forward/backward on assigned shard
    Worker,
}

/// Compute backend for a distributed worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistributedBackend {
    /// NVIDIA CUDA
    Cuda,
    /// wgpu (cross-platform)
    Wgpu,
    /// Auto-detect best available
    #[default]
    Auto,
}

/// Configuration for distributed pretraining (DDP).
///
/// Specifies this worker's role, rank, and communication topology.
/// All workers must agree on `world_size`. The coordinator address
/// is where workers connect and where AllReduce is orchestrated.
///
/// # Contract
///
/// C-DDP-001: After AllReduce + optimizer step, all workers hold identical weights.
#[derive(Debug, Clone)]
pub struct DistributedTrainConfig {
    /// Total number of workers participating
    pub world_size: usize,
    /// This worker's global rank (0-indexed)
    pub rank: usize,
    /// This worker's local rank on its machine (for multi-GPU)
    pub local_rank: usize,
    /// Role: coordinator (rank 0) or worker
    pub role: DistributedRole,
    /// Address for coordinator to bind / workers to connect
    pub coordinator_addr: SocketAddr,
    /// Compute backend for this worker
    pub backend: DistributedBackend,
}

/// Configuration for transformer training
#[derive(Debug, Clone)]
pub struct TransformerTrainConfig {
    /// Base training configuration
    pub base: TrainConfig,
    /// Transformer architecture configuration
    pub model_config: TransformerConfig,
    /// Checkpoint configuration for memory efficiency
    pub checkpoint_config: CheckpointConfig,
    /// Mixed-precision configuration
    pub precision_config: MixedPrecisionConfig,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Accumulation steps for gradient accumulation
    pub accumulation_steps: usize,
    /// Warmup steps for learning rate scheduler
    pub warmup_steps: usize,
    /// Learning rate
    pub lr: f32,
    /// Maximum training steps (stop after this many optimizer steps)
    pub max_steps: Option<usize>,
    /// Use CUDA GPU training when available (default: true = auto-detect)
    pub use_cuda: bool,
    /// AdamW beta1 (default: 0.9)
    pub beta1: f32,
    /// AdamW beta2 (default: 0.999)
    pub beta2: f32,
    /// AdamW weight decay (default: 0.01)
    pub weight_decay: f32,
    /// Distributed training configuration (None = single-GPU)
    pub distributed: Option<DistributedTrainConfig>,
    /// Enable bitwise deterministic training (CUBLAS_WORKSPACE_CONFIG, cuDNN deterministic)
    /// Contract: C-DETERM-001
    pub deterministic: bool,
    /// Random seed for reproducibility
    pub seed: u64,
    /// KAIZEN-047: Step profiler report interval (0 = disabled, N = print every N steps)
    pub profile_interval: usize,
    /// LoRA rank (None = full fine-tuning, Some(r) = LoRA with rank r)
    pub lora_rank: Option<usize>,
    /// LoRA alpha scaling factor (default: 2 * rank)
    pub lora_alpha: Option<f32>,
    /// LoRA target modules (e.g., `q_proj`, `v_proj`)
    pub lora_target_modules: Option<Vec<String>>,
    /// LoRA+ ratio: LR multiplier for B matrices (ENT-LoRA-006)
    /// Default 1.0 = standard LoRA. 16.0 = LoRA+ (Hayou et al. ICML 2024)
    pub lora_plus_ratio: f32,
    /// Double quantization for QLoRA (ENT-LoRA-008)
    /// Quantizes FP32 absmax constants to 8-bit, saving ~0.37 bits/param
    pub double_quantize: bool,
    /// Quantize frozen base weights to NF4 (4-bit) for QLoRA pretraining (ENT-263)
    ///
    /// When enabled with LoRA, uses `CudaNf4TransformerBlock` instead of
    /// `CudaTransformerBlock`, achieving ~8x VRAM compression on frozen weights.
    /// Only LoRA adapters and norm weights remain trainable in fp32.
    pub quantize_nf4: bool,
}

impl TransformerTrainConfig {
    /// Create new config with defaults
    pub fn new(model_config: TransformerConfig) -> Self {
        Self {
            base: TrainConfig::default(),
            model_config,
            checkpoint_config: CheckpointConfig::disabled(),
            precision_config: MixedPrecisionConfig::fp32(),
            max_seq_len: 512,
            accumulation_steps: 1,
            warmup_steps: 0,
            lr: 0.001,
            max_steps: None,
            use_cuda: true,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            distributed: None,
            deterministic: false,
            seed: 42,
            profile_interval: 0,
            lora_rank: None,
            lora_alpha: None,
            lora_target_modules: None,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_nf4: false,
        }
    }

    /// Enable gradient checkpointing
    pub fn with_checkpointing(mut self, num_segments: usize) -> Self {
        self.checkpoint_config = CheckpointConfig::enabled(num_segments);
        self
    }

    /// Enable bf16 mixed precision
    pub fn with_bf16(mut self) -> Self {
        self.precision_config = MixedPrecisionConfig::bf16();
        self
    }

    /// Enable fp16 mixed precision with dynamic loss scaling
    pub fn with_fp16(mut self) -> Self {
        self.precision_config = MixedPrecisionConfig::fp16();
        self
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Set gradient accumulation steps
    pub fn with_accumulation_steps(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps.max(1);
        self
    }

    /// Set warmup steps
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set learning rate
    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set gradient clipping
    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.base.max_grad_norm = Some(clip);
        self
    }

    /// Set maximum training steps
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Enable or disable CUDA GPU training (default: true = auto-detect)
    pub fn with_use_cuda(mut self, use_cuda: bool) -> Self {
        self.use_cuda = use_cuda;
        self
    }

    /// Set AdamW beta2 (default: 0.999)
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set AdamW weight decay (default: 0.01)
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Enable bitwise deterministic training (C-DETERM-001)
    ///
    /// Sets CUBLAS_WORKSPACE_CONFIG, cuDNN deterministic mode, and disables
    /// cuDNN benchmark. May reduce throughput but guarantees reproducibility.
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply deterministic settings to the CUDA environment.
    ///
    /// Must be called before any cuBLAS/cuDNN operations.
    /// Uses `ReproducibilityConfig` from finetune infrastructure.
    ///
    /// # Contract (C-DETERM-001)
    ///
    /// After calling this, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
    /// `CUDNN_DETERMINISTIC=1` are guaranteed set in the process environment.
    pub fn apply_deterministic_settings(&self) {
        if self.deterministic {
            use crate::finetune::ReproducibilityConfig;
            let repro = ReproducibilityConfig::with_seed(self.seed);
            repro.apply();
        }
    }

    /// Set step profiler report interval (0 = disabled, N = print every N steps)
    pub fn with_profile_interval(mut self, interval: usize) -> Self {
        self.profile_interval = interval;
        self
    }

    /// Enable LoRA fine-tuning with rank, alpha, and target modules
    ///
    /// When LoRA is enabled, only LoRA adapter weights (A, B matrices) and
    /// layer norms are trainable. Base model weights are frozen.
    ///
    /// # Contract (ENT-LoRA-001)
    /// - Base weights frozen (requires_grad=false)
    /// - Only LoRA A/B + norms are optimizer targets
    /// - scale = alpha / rank
    pub fn with_lora(mut self, rank: usize, alpha: f32, target_modules: Vec<String>) -> Self {
        self.lora_rank = Some(rank);
        self.lora_alpha = Some(alpha);
        self.lora_target_modules = Some(target_modules);
        self
    }

    /// Set LoRA+ ratio (ENT-LoRA-006)
    ///
    /// LR multiplier for B matrices. Default 1.0 = standard LoRA.
    /// 16.0 = LoRA+ (Hayou et al. ICML 2024) — B learns 16x faster than A.
    pub fn with_lora_plus_ratio(mut self, ratio: f32) -> Self {
        self.lora_plus_ratio = ratio;
        self
    }

    /// Enable double quantization for QLoRA (ENT-LoRA-008)
    pub fn with_double_quantize(mut self, enabled: bool) -> Self {
        self.double_quantize = enabled;
        self
    }

    /// Enable NF4 quantization for QLoRA pretraining (ENT-263)
    ///
    /// When enabled with LoRA, frozen base weights are quantized to 4-bit NF4,
    /// achieving ~8x VRAM compression. Only LoRA adapters and norm weights are
    /// trainable. Requires `lora_rank` to be set.
    pub fn with_quantize_nf4(mut self, enabled: bool) -> Self {
        self.quantize_nf4 = enabled;
        self
    }

    /// Check if NF4 quantization is enabled for QLoRA
    #[must_use]
    pub fn is_nf4(&self) -> bool {
        self.quantize_nf4
    }

    /// Check if LoRA fine-tuning is enabled
    #[must_use]
    pub fn is_lora(&self) -> bool {
        self.lora_rank.is_some()
    }

    /// Enable distributed training with the given configuration
    pub fn with_distributed(mut self, config: DistributedTrainConfig) -> Self {
        self.distributed = Some(config);
        self
    }

    /// Check if distributed training is enabled
    #[must_use]
    pub fn is_distributed(&self) -> bool {
        self.distributed.is_some()
    }

    /// Get world size (1 for single-GPU)
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.distributed.as_ref().map_or(1, |d| d.world_size)
    }

    /// Get this worker's rank (0 for single-GPU)
    #[must_use]
    pub fn rank(&self) -> usize {
        self.distributed.as_ref().map_or(0, |d| d.rank)
    }
}
