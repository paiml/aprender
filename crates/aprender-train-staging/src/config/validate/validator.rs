//! Configuration validation logic
//!
//! Validates training specifications for correctness before execution.

use super::error::ValidationError;
use crate::config::schema::TrainSpec;

/// Validate a training specification
///
/// Checks:
/// - File paths exist
/// - Numeric values are in valid ranges
/// - Enums match allowed values
pub fn validate_config(spec: &TrainSpec) -> Result<(), ValidationError> {
    validate_model_path(spec)?;
    validate_data_paths(spec)?;
    validate_batch_size(spec)?;
    validate_learning_rate(spec)?;
    validate_optimizer(spec)?;
    validate_epochs(spec)?;
    validate_training_params(spec)?;
    validate_lora(spec)?;
    validate_quantization(spec)?;
    validate_merge(spec)?;
    validate_publish(spec)?;
    Ok(())
}

/// Validate model path exists (or is a valid HuggingFace repo ID)
#[cfg(not(test))]
fn validate_model_path(spec: &TrainSpec) -> Result<(), ValidationError> {
    // Accept HF repo IDs — they'll be resolved at training time
    if spec.model.is_hf_repo_id() {
        return Ok(());
    }
    if !spec.model.path.exists() {
        return Err(ValidationError::ModelPathNotFound(spec.model.path.display().to_string()));
    }
    Ok(())
}

#[cfg(test)]
fn validate_model_path(_spec: &TrainSpec) -> Result<(), ValidationError> {
    Ok(())
}

/// Validate data paths exist
#[cfg(not(test))]
fn validate_data_paths(spec: &TrainSpec) -> Result<(), ValidationError> {
    if !spec.data.train.exists() {
        return Err(ValidationError::TrainDataNotFound(spec.data.train.display().to_string()));
    }

    if let Some(val_path) = &spec.data.val {
        if !val_path.exists() {
            return Err(ValidationError::ValDataNotFound(val_path.display().to_string()));
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_data_paths(_spec: &TrainSpec) -> Result<(), ValidationError> {
    Ok(())
}

/// Validate batch size is non-zero
fn validate_batch_size(spec: &TrainSpec) -> Result<(), ValidationError> {
    if spec.data.batch_size == 0 {
        return Err(ValidationError::InvalidBatchSize(spec.data.batch_size));
    }
    Ok(())
}

/// Validate learning rate is positive and reasonable
fn validate_learning_rate(spec: &TrainSpec) -> Result<(), ValidationError> {
    if spec.optimizer.lr <= 0.0 || spec.optimizer.lr > 1.0 {
        return Err(ValidationError::InvalidLearningRate(spec.optimizer.lr));
    }
    Ok(())
}

/// Validate optimizer name is supported
fn validate_optimizer(spec: &TrainSpec) -> Result<(), ValidationError> {
    const VALID_OPTIMIZERS: [&str; 6] = ["adam", "adamw", "sgd", "rmsprop", "adagrad", "lamb"];
    if !VALID_OPTIMIZERS.contains(&spec.optimizer.name.as_str()) {
        return Err(ValidationError::InvalidOptimizer(spec.optimizer.name.clone()));
    }
    Ok(())
}

/// Validate epochs is non-zero
fn validate_epochs(spec: &TrainSpec) -> Result<(), ValidationError> {
    if spec.training.epochs == 0 {
        return Err(ValidationError::InvalidEpochs(spec.training.epochs));
    }
    Ok(())
}

/// Validate training parameters (grad_clip, seq_len, save_interval, lr_scheduler)
fn validate_training_params(spec: &TrainSpec) -> Result<(), ValidationError> {
    validate_grad_clip(spec)?;
    validate_seq_len(spec)?;
    validate_save_interval(spec)?;
    validate_lr_scheduler(spec)?;
    Ok(())
}

/// Validate gradient clipping value
fn validate_grad_clip(spec: &TrainSpec) -> Result<(), ValidationError> {
    if let Some(grad_clip) = spec.training.grad_clip {
        if grad_clip <= 0.0 {
            return Err(ValidationError::InvalidGradClip(grad_clip));
        }
    }
    Ok(())
}

/// Validate sequence length if specified
fn validate_seq_len(spec: &TrainSpec) -> Result<(), ValidationError> {
    if let Some(seq_len) = spec.data.seq_len {
        if seq_len == 0 {
            return Err(ValidationError::InvalidSeqLen(seq_len));
        }
    }
    Ok(())
}

/// Validate save interval
fn validate_save_interval(spec: &TrainSpec) -> Result<(), ValidationError> {
    if spec.training.save_interval == 0 {
        return Err(ValidationError::InvalidSaveInterval(spec.training.save_interval));
    }
    Ok(())
}

/// Validate LR scheduler if specified
fn validate_lr_scheduler(spec: &TrainSpec) -> Result<(), ValidationError> {
    if let Some(scheduler) = &spec.training.lr_scheduler {
        const VALID_SCHEDULERS: [&str; 7] =
            ["cosine", "linear", "constant", "step", "exponential", "one_cycle", "plateau"];
        if !VALID_SCHEDULERS.contains(&scheduler.as_str()) {
            return Err(ValidationError::InvalidLRScheduler(scheduler.clone()));
        }
    }
    Ok(())
}

/// Validate LoRA configuration if present
fn validate_lora(spec: &TrainSpec) -> Result<(), ValidationError> {
    let Some(lora) = &spec.lora else {
        return Ok(());
    };

    validate_lora_rank(lora.rank)?;
    validate_lora_alpha(lora.alpha)?;
    validate_lora_dropout(lora.dropout)?;
    validate_lora_targets(&lora.target_modules)?;
    Ok(())
}

/// Validate LoRA rank (1-1024)
fn validate_lora_rank(rank: usize) -> Result<(), ValidationError> {
    if rank == 0 || rank > 1024 {
        return Err(ValidationError::InvalidLoRARank(rank));
    }
    Ok(())
}

/// Validate LoRA alpha (must be positive)
fn validate_lora_alpha(alpha: f32) -> Result<(), ValidationError> {
    if alpha <= 0.0 {
        return Err(ValidationError::InvalidLoRAAlpha(alpha));
    }
    Ok(())
}

/// Validate LoRA dropout (0.0 to <1.0)
fn validate_lora_dropout(dropout: f32) -> Result<(), ValidationError> {
    if !(0.0..1.0).contains(&dropout) {
        return Err(ValidationError::InvalidLoRADropout(dropout));
    }
    Ok(())
}

/// Validate LoRA target modules are not empty
fn validate_lora_targets(targets: &[String]) -> Result<(), ValidationError> {
    if targets.is_empty() {
        return Err(ValidationError::EmptyLoRATargets);
    }
    Ok(())
}

/// Validate quantization configuration if present
fn validate_quantization(spec: &TrainSpec) -> Result<(), ValidationError> {
    let Some(quant) = &spec.quantize else {
        return Ok(());
    };

    if quant.bits != 4 && quant.bits != 8 {
        return Err(ValidationError::InvalidQuantBits(quant.bits));
    }
    Ok(())
}

/// Validate merge configuration if present
fn validate_merge(spec: &TrainSpec) -> Result<(), ValidationError> {
    let Some(merge) = &spec.merge else {
        return Ok(());
    };

    const VALID_METHODS: [&str; 3] = ["ties", "dare", "slerp"];
    if !VALID_METHODS.contains(&merge.method.as_str()) {
        return Err(ValidationError::InvalidMergeMethod(merge.method.clone()));
    }
    Ok(())
}

/// Validate publish configuration if present
fn validate_publish(spec: &TrainSpec) -> Result<(), ValidationError> {
    let Some(publish) = &spec.publish else {
        return Ok(());
    };

    // Repo ID must contain exactly one `/`
    let parts: Vec<&str> = publish.repo.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(ValidationError::InvalidPublishRepo(publish.repo.clone()));
    }

    // Format must be safetensors or gguf
    if publish.format != "safetensors" && publish.format != "gguf" {
        return Err(ValidationError::InvalidPublishFormat(publish.format.clone()));
    }

    Ok(())
}
