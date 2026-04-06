//! Fine-tuning method selection
//!
//! Different approaches for adapting pretrained models.

use crate::lora::LoRAConfig;

/// Fine-tuning method selection
#[derive(Debug, Clone)]
pub enum FineTuneMethod {
    /// Full fine-tuning (all parameters trainable)
    Full,
    /// LoRA: Low-rank adaptation (Hu et al. 2021)
    LoRA(LoRAConfig),
    /// QLoRA: Quantized LoRA (Dettmers et al. 2023)
    QLoRA {
        /// LoRA configuration
        lora_config: LoRAConfig,
        /// Quantization bits (4 or 8)
        bits: u8,
    },
    /// Prefix tuning (Li & Liang 2021)
    PrefixTuning {
        /// Number of prefix tokens
        prefix_length: usize,
    },
}

impl Default for FineTuneMethod {
    fn default() -> Self {
        Self::LoRA(LoRAConfig::default())
    }
}
