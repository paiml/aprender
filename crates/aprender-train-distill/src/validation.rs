//! Configuration validation (Jidoka - built-in quality).
//!
//! Validates configuration before training to catch errors early
//! and provide actionable feedback.

use crate::config::DistillConfig;
use entrenar_common::{EntrenarError, Result};

/// Configuration validator implementing Jidoka principle.
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate a distillation configuration.
    ///
    /// Returns `Ok(())` if valid, or an error with actionable suggestions.
    pub fn validate(config: &DistillConfig) -> Result<()> {
        Self::validate_teacher(&config.teacher)?;
        Self::validate_student(&config.student)?;
        Self::validate_distillation(&config.distillation)?;
        Self::validate_training(&config.training)?;
        Self::validate_dataset(&config.dataset)?;
        Self::validate_output(&config.output)?;
        Ok(())
    }

    fn validate_teacher(config: &crate::config::TeacherConfig) -> Result<()> {
        if config.model_id.is_empty() {
            return Err(EntrenarError::ConfigValue {
                field: "teacher.model_id".into(),
                message: "Teacher model ID cannot be empty".into(),
                suggestion: "Provide a HuggingFace model ID like 'meta-llama/Llama-2-7b'".into(),
            });
        }

        // Validate model ID format (should contain / for HF models)
        if !config.model_id.contains('/') && !std::path::Path::new(&config.model_id).exists() {
            return Err(EntrenarError::ConfigValue {
                field: "teacher.model_id".into(),
                message: format!("Invalid model ID: {}", config.model_id),
                suggestion: "Use format 'org/model-name' for HuggingFace or a valid local path"
                    .into(),
            });
        }

        Ok(())
    }

    fn validate_student(config: &crate::config::StudentConfig) -> Result<()> {
        if config.model_id.is_empty() {
            return Err(EntrenarError::ConfigValue {
                field: "student.model_id".into(),
                message: "Student model ID cannot be empty".into(),
                suggestion: "Provide a HuggingFace model ID like 'TinyLlama/TinyLlama-1.1B'".into(),
            });
        }

        // Validate LoRA config if present
        if let Some(lora) = &config.lora {
            if lora.rank == 0 {
                return Err(EntrenarError::ConfigValue {
                    field: "student.lora.rank".into(),
                    message: "LoRA rank must be positive".into(),
                    suggestion: "Use rank 16-256 (64 recommended for most cases)".into(),
                });
            }

            if lora.alpha <= 0.0 {
                return Err(EntrenarError::ConfigValue {
                    field: "student.lora.alpha".into(),
                    message: "LoRA alpha must be positive".into(),
                    suggestion: "Use alpha equal to rank/4 (e.g., 16 for rank 64)".into(),
                });
            }

            if lora.target_modules.is_empty() {
                return Err(EntrenarError::ConfigValue {
                    field: "student.lora.target_modules".into(),
                    message: "LoRA target modules cannot be empty".into(),
                    suggestion: "Use [q_proj, k_proj, v_proj, o_proj] for attention layers".into(),
                });
            }

            if lora.dropout < 0.0 || lora.dropout >= 1.0 {
                return Err(EntrenarError::ConfigValue {
                    field: "student.lora.dropout".into(),
                    message: format!("Invalid dropout: {}", lora.dropout),
                    suggestion: "Use dropout between 0.0 and 1.0 (0.1 recommended)".into(),
                });
            }
        }

        Ok(())
    }

    fn validate_distillation(config: &crate::config::DistillationParams) -> Result<()> {
        if config.temperature <= 0.0 {
            return Err(EntrenarError::ConfigValue {
                field: "distillation.temperature".into(),
                message: format!("Temperature must be positive, got {}", config.temperature),
                suggestion: "Use temperature 1.0-8.0 (4.0 recommended)".into(),
            });
        }

        if config.temperature > 20.0 {
            return Err(EntrenarError::ConfigValue {
                field: "distillation.temperature".into(),
                message: format!("Temperature too high: {}", config.temperature),
                suggestion: "Use temperature 1.0-8.0; higher values over-smooth distributions"
                    .into(),
            });
        }

        if config.alpha < 0.0 || config.alpha > 1.0 {
            return Err(EntrenarError::ConfigValue {
                field: "distillation.alpha".into(),
                message: format!("Alpha must be between 0 and 1, got {}", config.alpha),
                suggestion: "Use alpha 0.5-0.9 (0.7 recommended for most cases)".into(),
            });
        }

        // Validate progressive config
        if let Some(progressive) = &config.progressive {
            if progressive.enabled && progressive.layer_mapping.is_empty() {
                return Err(EntrenarError::ConfigValue {
                    field: "distillation.progressive.layer_mapping".into(),
                    message: "Progressive distillation enabled but no layer mapping provided"
                        .into(),
                    suggestion: "Add layer_mapping like [[0, 3], [1, 7], [2, 11]]".into(),
                });
            }

            if progressive.weight < 0.0 || progressive.weight > 1.0 {
                return Err(EntrenarError::ConfigValue {
                    field: "distillation.progressive.weight".into(),
                    message: format!("Progressive weight must be 0-1, got {}", progressive.weight),
                    suggestion: "Use weight 0.1-0.5 (0.3 recommended)".into(),
                });
            }
        }

        // Validate attention config
        if let Some(attention) = &config.attention {
            if attention.weight < 0.0 || attention.weight > 1.0 {
                return Err(EntrenarError::ConfigValue {
                    field: "distillation.attention.weight".into(),
                    message: format!("Attention weight must be 0-1, got {}", attention.weight),
                    suggestion: "Use weight 0.05-0.2 (0.1 recommended)".into(),
                });
            }
        }

        Ok(())
    }

    fn validate_training(config: &crate::config::TrainingConfig) -> Result<()> {
        if config.epochs == 0 {
            return Err(EntrenarError::ConfigValue {
                field: "training.epochs".into(),
                message: "Number of epochs must be positive".into(),
                suggestion: "Use 3-20 epochs (10 recommended for most tasks)".into(),
            });
        }

        if config.batch_size == 0 {
            return Err(EntrenarError::ConfigValue {
                field: "training.batch_size".into(),
                message: "Batch size must be positive".into(),
                suggestion: "Use batch_size 8-64 depending on memory".into(),
            });
        }

        if config.learning_rate <= 0.0 {
            return Err(EntrenarError::ConfigValue {
                field: "training.learning_rate".into(),
                message: "Learning rate must be positive".into(),
                suggestion: "Use 1e-5 to 1e-3 (1e-4 recommended for distillation)".into(),
            });
        }

        if config.learning_rate > 0.1 {
            return Err(EntrenarError::ConfigValue {
                field: "training.learning_rate".into(),
                message: format!("Learning rate {} is too high", config.learning_rate),
                suggestion: "Use learning rate < 0.01 to avoid training instability".into(),
            });
        }

        if config.gradient_accumulation == 0 {
            return Err(EntrenarError::ConfigValue {
                field: "training.gradient_accumulation".into(),
                message: "Gradient accumulation steps must be positive".into(),
                suggestion: "Use 1 (no accumulation) or 2-8 for larger effective batch size".into(),
            });
        }

        Ok(())
    }

    fn validate_dataset(config: &crate::config::DatasetConfig) -> Result<()> {
        if config.max_length == 0 {
            return Err(EntrenarError::ConfigValue {
                field: "dataset.max_length".into(),
                message: "Max sequence length must be positive".into(),
                suggestion: "Use 256-2048 (512 recommended for most tasks)".into(),
            });
        }

        if config.max_length > 8192 {
            return Err(EntrenarError::ConfigValue {
                field: "dataset.max_length".into(),
                message: format!("Max length {} may cause memory issues", config.max_length),
                suggestion: "Use max_length <= 4096 unless you have >48GB VRAM".into(),
            });
        }

        Ok(())
    }

    fn validate_output(config: &crate::config::OutputConfig) -> Result<()> {
        // Check if output directory's parent exists
        if let Some(parent) = config.dir.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                return Err(EntrenarError::ConfigValue {
                    field: "output.dir".into(),
                    message: format!("Parent directory does not exist: {}", parent.display()),
                    suggestion: "Create the parent directory or use a different output path".into(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DistillConfig, LoraConfig};

    #[test]
    fn test_valid_minimal_config() {
        let config = DistillConfig::minimal("org/teacher", "org/student");
        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok(), "Validation failed: {:?}", result.err());
    }

    #[test]
    fn test_empty_teacher_model_id() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.teacher.model_id = String::new();

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("teacher.model_id"));
    }

    #[test]
    fn test_invalid_temperature() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.distillation.temperature = -1.0;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));
    }

    #[test]
    fn test_temperature_too_high() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.distillation.temperature = 50.0;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_alpha() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.distillation.alpha = 1.5;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("alpha"));
    }

    #[test]
    fn test_invalid_lora_rank() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.student.lora = Some(LoraConfig {
            rank: 0,
            ..LoraConfig::default()
        });

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rank"));
    }

    #[test]
    fn test_invalid_learning_rate() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.training.learning_rate = 0.5;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("learning_rate"));
    }

    #[test]
    fn test_zero_epochs() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.training.epochs = 0;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_messages_are_actionable() {
        let mut config = DistillConfig::minimal("org/teacher", "org/student");
        config.distillation.temperature = -1.0;

        let result = ConfigValidator::validate(&config);
        let error_msg = result.unwrap_err().to_string();

        // Should contain the field name
        assert!(error_msg.contains("temperature"));
        // Should contain a suggestion
        assert!(error_msg.contains("1.0-8.0") || error_msg.contains("recommended"));
    }
}
