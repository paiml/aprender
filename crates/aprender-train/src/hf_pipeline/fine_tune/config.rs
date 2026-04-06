//! Fine-tuning configuration
//!
//! Provides configuration options for fine-tuning HuggingFace models.

use std::path::PathBuf;

use crate::hf_pipeline::error::Result;
use crate::hf_pipeline::FetchError;
use crate::lora::LoRAConfig;

use super::memory::{MemoryRequirement, MixedPrecision};
use super::method::FineTuneMethod;

/// Default number of training steps between checkpoint saves.
const DEFAULT_SAVE_STEPS: usize = 500;

/// Fine-tuning configuration
#[derive(Debug, Clone)]
pub struct FineTuneConfig {
    /// Base model repository ID
    pub model_id: String,
    /// Fine-tuning method
    pub method: FineTuneMethod,
    /// Output directory for checkpoints
    pub output_dir: PathBuf,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Weight decay
    pub weight_decay: f64,
    /// Warmup ratio (fraction of total steps)
    pub warmup_ratio: f32,
    /// Save checkpoints every N steps
    pub save_steps: usize,
    /// Evaluate every N steps
    pub eval_steps: usize,
    /// Use gradient checkpointing (memory optimization)
    pub gradient_checkpointing: bool,
    /// Use mixed precision (fp16/bf16)
    pub mixed_precision: Option<MixedPrecision>,
}

impl Default for FineTuneConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            method: FineTuneMethod::default(),
            output_dir: PathBuf::from("./output"),
            learning_rate: 2e-4, // Recommended for LoRA
            epochs: 3,
            batch_size: 8,
            max_seq_length: 512,
            gradient_accumulation_steps: 4,
            weight_decay: 0.01,
            warmup_ratio: 0.03,
            save_steps: DEFAULT_SAVE_STEPS,
            eval_steps: 100,
            gradient_checkpointing: true,
            mixed_precision: Some(MixedPrecision::Bf16),
        }
    }
}

impl FineTuneConfig {
    /// Create new fine-tuning config for a model
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self { model_id: model_id.into(), ..Default::default() }
    }

    /// Use LoRA fine-tuning
    #[must_use]
    pub fn with_lora(mut self, config: LoRAConfig) -> Self {
        self.method = FineTuneMethod::LoRA(config);
        self
    }

    /// Use QLoRA fine-tuning
    #[must_use]
    pub fn with_qlora(mut self, lora_config: LoRAConfig, bits: u8) -> Self {
        self.method = FineTuneMethod::QLoRA { lora_config, bits };
        self
    }

    /// Use full fine-tuning
    #[must_use]
    pub fn full_fine_tune(mut self) -> Self {
        self.method = FineTuneMethod::Full;
        self
    }

    /// Set learning rate
    #[must_use]
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of epochs
    #[must_use]
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }

    /// Set batch size
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set output directory
    #[must_use]
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = path.into();
        self
    }

    /// Enable gradient checkpointing
    #[must_use]
    pub fn gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.gradient_checkpointing = enabled;
        self
    }

    /// Set mixed precision mode
    #[must_use]
    pub fn mixed_precision(mut self, mode: Option<MixedPrecision>) -> Self {
        self.mixed_precision = mode;
        self
    }

    /// Estimate trainable parameters based on fine-tuning method.
    ///
    /// N-06 (Meyer DbC): Derives hidden_size and num_layers from total_params
    /// using the approximation total ≈ 12 * L * d² (transformer scaling law).
    #[must_use]
    pub fn estimate_trainable_params(&self, total_params: u64) -> u64 {
        // Estimate hidden_size from total params: total ≈ 12 * L * d²
        // Rough: d ≈ sqrt(total / 384) for typical 32-layer model
        let d = ((total_params as f64 / 384.0).sqrt() as u64).max(64);
        let num_layers_est = (total_params / (12 * d * d)).clamp(1, 128);

        match &self.method {
            FineTuneMethod::Full => total_params,
            FineTuneMethod::LoRA(config) => {
                // LoRA params = 2 * rank * d * num_modules * num_layers
                let num_modules = config.num_target_modules().max(4);
                2 * (config.rank as u64) * d * (num_modules as u64) * num_layers_est
            }
            FineTuneMethod::QLoRA { lora_config, .. } => {
                let num_modules = lora_config.num_target_modules().max(4);
                2 * (lora_config.rank as u64) * d * (num_modules as u64) * num_layers_est
            }
            FineTuneMethod::PrefixTuning { prefix_length } => {
                // Prefix params = prefix_length * hidden_size * 2 * num_layers
                (*prefix_length as u64) * d * 2 * num_layers_est
            }
        }
    }

    /// Estimate memory requirements in bytes
    #[must_use]
    pub fn estimate_memory(&self, total_params: u64) -> MemoryRequirement {
        let trainable = self.estimate_trainable_params(total_params);

        // Model memory
        let model_bytes = match &self.method {
            FineTuneMethod::Full => total_params * 4,    // FP32
            FineTuneMethod::LoRA(_) => total_params * 2, // FP16 base + LoRA
            FineTuneMethod::QLoRA { bits, .. } => {
                // Quantized base + FP16 LoRA
                let base = match bits {
                    4 => total_params / 2,
                    2 | 3 | 5..=8 | 0 | 1 | 9.. => total_params,
                };
                base + trainable * 2
            }
            FineTuneMethod::PrefixTuning { .. } => total_params * 2 + trainable * 4,
        };

        // Optimizer states (Adam: 2x for momentum + variance)
        let optimizer_bytes = trainable * 4 * 2;

        // Gradients
        let gradient_bytes = trainable * 4;

        // Activations (rough estimate based on batch size and seq len)
        let activation_bytes = (self.batch_size * self.max_seq_length * 4096 * 4) as u64
            * if self.gradient_checkpointing { 1 } else { 4 };

        MemoryRequirement {
            model: model_bytes,
            optimizer: optimizer_bytes,
            gradients: gradient_bytes,
            activations: activation_bytes,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() {
            return Err(FetchError::InvalidRepoId { repo_id: String::new() });
        }

        if self.learning_rate <= 0.0 {
            return Err(FetchError::ConfigParseError {
                message: "Learning rate must be positive".into(),
            });
        }

        if self.batch_size == 0 {
            return Err(FetchError::ConfigParseError {
                message: "Batch size must be greater than 0".into(),
            });
        }

        if let FineTuneMethod::QLoRA { bits, .. } = &self.method {
            if *bits != 4 && *bits != 8 {
                return Err(FetchError::ConfigParseError {
                    message: format!("QLoRA bits must be 4 or 8, got {bits}"),
                });
            }
        }

        Ok(())
    }
}
