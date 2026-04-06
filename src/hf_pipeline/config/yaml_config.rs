//! Complete distillation YAML configuration

use crate::hf_pipeline::error::{FetchError, Result};
use crate::hf_pipeline::fine_tune::{FineTuneConfig, MixedPrecision};
use crate::hf_pipeline::trainer::TrainerConfig;
use crate::lora::LoRAConfig;
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::dataset::DatasetConfig;
use super::distillation::DistillationConfig;
use super::output::OutputConfig;
use super::student::StudentConfig;
use super::teacher::TeacherConfig;
use super::training::TrainingConfig;

/// Complete distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationYamlConfig {
    /// Teacher model config
    pub teacher: TeacherConfig,
    /// Student model config
    pub student: StudentConfig,
    /// Distillation loss config
    #[serde(default)]
    pub distillation: DistillationConfig,
    /// Training hyperparameters
    #[serde(default)]
    pub training: TrainingConfig,
    /// Dataset config
    pub dataset: DatasetConfig,
    /// Output config
    #[serde(default)]
    pub output: OutputConfig,
}

impl DistillationYamlConfig {
    /// Load configuration from YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            FetchError::ConfigParseError { message: format!("Failed to read config file: {e}") }
        })?;

        Self::from_yaml(&content)
    }

    /// Parse configuration from YAML string
    pub fn from_yaml(content: &str) -> Result<Self> {
        serde_yaml::from_str(content).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to parse YAML: {e}"),
        })
    }

    /// Save configuration to YAML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = self.to_yaml()?;
        std::fs::write(path, content).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to write config file: {e}"),
        })
    }

    /// Serialize to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to serialize YAML: {e}"),
        })
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate teacher
        if self.teacher.model_id.is_empty() {
            return Err(FetchError::ConfigParseError {
                message: "teacher.model_id cannot be empty".into(),
            });
        }

        // Validate student
        if self.student.model_id.is_empty() {
            return Err(FetchError::ConfigParseError {
                message: "student.model_id cannot be empty".into(),
            });
        }

        // Validate distillation
        if self.distillation.temperature <= 0.0 {
            return Err(FetchError::ConfigParseError {
                message: "distillation.temperature must be positive".into(),
            });
        }

        if !(0.0..=1.0).contains(&self.distillation.alpha) {
            return Err(FetchError::ConfigParseError {
                message: "distillation.alpha must be between 0 and 1".into(),
            });
        }

        // Validate training
        if self.training.batch_size == 0 {
            return Err(FetchError::ConfigParseError {
                message: "training.batch_size must be > 0".into(),
            });
        }

        if self.training.learning_rate <= 0.0 {
            return Err(FetchError::ConfigParseError {
                message: "training.learning_rate must be positive".into(),
            });
        }

        // Validate dataset
        if self.dataset.path.is_empty() {
            return Err(FetchError::ConfigParseError {
                message: "dataset.path cannot be empty".into(),
            });
        }

        Ok(())
    }

    /// Convert to TrainerConfig
    pub fn to_trainer_config(&self) -> Result<TrainerConfig> {
        self.validate()?;

        let mut config = TrainerConfig::new(&self.teacher.model_id, &self.student.model_id)
            .temperature(self.distillation.temperature)
            .alpha(self.distillation.alpha)
            .epochs(self.training.epochs)
            .output_dir(&self.output.dir);

        // Add progressive distillation
        if let Some(ref prog) = self.distillation.progressive {
            let mapping: Vec<(usize, usize)> =
                prog.layer_mapping.iter().map(|[s, t]| (*s, *t)).collect();
            config = config.with_progressive(mapping);
        }

        // Add attention transfer
        if let Some(ref at) = self.distillation.attention_transfer {
            config = config.with_attention_transfer(at.weight);
        }

        // Set up fine-tuning config
        let mut fine_tune = FineTuneConfig::new(&self.student.model_id)
            .learning_rate(self.training.learning_rate)
            .epochs(self.training.epochs)
            .batch_size(self.training.batch_size);

        // Set LoRA if configured
        if let Some(ref lora_yaml) = self.student.lora {
            let lora_config = LoRAConfig::from(lora_yaml);
            if self.student.load_in_4bit {
                fine_tune = fine_tune.with_qlora(lora_config, 4);
            } else {
                fine_tune = fine_tune.with_lora(lora_config);
            }
        } else if !self.student.load_in_4bit {
            fine_tune = fine_tune.full_fine_tune();
        }

        // Set mixed precision
        if let Some(ref mp) = self.training.mixed_precision {
            fine_tune = fine_tune.mixed_precision(match mp.as_str() {
                "fp16" => Some(MixedPrecision::Fp16),
                "bf16" => Some(MixedPrecision::Bf16),
                _ => None,
            });
        }

        fine_tune = fine_tune.gradient_checkpointing(self.training.gradient_checkpointing);

        config.fine_tune = fine_tune;
        config.max_grad_norm = self.training.max_grad_norm;
        config.seed = self.training.seed;
        config.log_every_n_steps = self.output.log_steps;
        config.save_every_n_steps = self.output.save_steps;
        config.eval_every_n_steps = self.output.eval_steps;

        Ok(config)
    }
}
