//! Model training/inference paradigm definitions.

use serde::{Deserialize, Serialize};

use super::FineTuneMethod;

/// Model training/inference paradigm
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ModelParadigm {
    /// Traditional ML (sklearn-style): logistic regression, random forests, etc.
    TraditionalMl,
    /// Deep learning: neural networks trained from scratch
    #[default]
    DeepLearning,
    /// Fine-tuning a pretrained model
    FineTuning(FineTuneMethod),
    /// Knowledge distillation (teacher → student)
    Distillation,
    /// Mixture of Experts
    MoE,
    /// Ensemble of multiple models
    Ensemble,
}

impl ModelParadigm {
    /// Create a LoRA fine-tuning paradigm
    pub fn lora(rank: u32, alpha: f32) -> Self {
        Self::FineTuning(FineTuneMethod::LoRA { rank, alpha })
    }

    /// Create a QLoRA fine-tuning paradigm
    pub fn qlora(rank: u32, bits: u8) -> Self {
        Self::FineTuning(FineTuneMethod::QLoRA { rank, bits })
    }

    /// Get typical memory multiplier relative to model size
    ///
    /// Returns the factor by which memory usage increases during training
    /// compared to inference-only memory.
    pub fn typical_memory_multiplier(&self) -> f64 {
        match self {
            Self::TraditionalMl => 1.5,
            Self::DeepLearning => 4.0,
            Self::FineTuning(method) => method.memory_multiplier(),
            Self::Distillation => 5.0,
            Self::MoE => 2.0,
            Self::Ensemble => 3.0,
        }
    }

    /// Get typical training speedup relative to full training
    ///
    /// Returns the factor by which training is faster compared to
    /// training from scratch.
    pub fn typical_training_speedup(&self) -> f64 {
        match self {
            Self::TraditionalMl => 10.0,
            Self::DeepLearning => 1.0,
            Self::FineTuning(method) => method.training_speedup(),
            Self::Distillation => 1.5,
            Self::MoE => 0.8,
            Self::Ensemble => 0.5,
        }
    }

    /// Get typical quality retention compared to full training
    ///
    /// Returns expected quality as a fraction of full fine-tuning quality.
    pub fn typical_quality_retention(&self) -> f64 {
        match self {
            Self::TraditionalMl => 0.7,
            Self::DeepLearning => 1.0,
            Self::FineTuning(method) => method.quality_retention(),
            Self::Distillation => 0.85,
            Self::MoE => 1.05,
            Self::Ensemble => 1.02,
        }
    }

    /// Check if this paradigm requires a pretrained model
    pub fn requires_pretrained(&self) -> bool {
        matches!(self, Self::FineTuning(_) | Self::Distillation)
    }

    /// Check if this paradigm is parameter-efficient
    pub fn is_parameter_efficient(&self) -> bool {
        matches!(
            self,
            Self::FineTuning(
                FineTuneMethod::LoRA { .. }
                    | FineTuneMethod::QLoRA { .. }
                    | FineTuneMethod::Adapter
                    | FineTuneMethod::Prefix
                    | FineTuneMethod::IA3
            )
        )
    }

    /// Get recommended batch size multiplier
    ///
    /// Parameter-efficient methods allow larger batch sizes due to lower memory.
    pub fn batch_size_multiplier(&self) -> f64 {
        match self {
            Self::TraditionalMl => 10.0,
            Self::DeepLearning => 1.0,
            Self::FineTuning(method) => method.batch_size_multiplier(),
            Self::Distillation => 0.5,
            Self::MoE => 1.2,
            Self::Ensemble => 0.3,
        }
    }
}

impl std::fmt::Display for ModelParadigm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TraditionalMl => write!(f, "Traditional ML"),
            Self::DeepLearning => write!(f, "Deep Learning"),
            Self::FineTuning(method) => write!(f, "Fine-tuning ({method})"),
            Self::Distillation => write!(f, "Knowledge Distillation"),
            Self::MoE => write!(f, "Mixture of Experts"),
            Self::Ensemble => write!(f, "Ensemble"),
        }
    }
}
