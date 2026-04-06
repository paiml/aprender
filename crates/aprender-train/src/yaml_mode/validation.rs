//! Manifest Validation (Poka-yoke)
//!
//! Schema validation catches errors at parse time, not runtime.
//! Implements the Toyota Way's poka-yoke principle of defect prevention at source.

use super::manifest::TrainingManifest;
use thiserror::Error;

/// Validation result type
pub type ValidationResult<T> = Result<T, ManifestError>;

/// Manifest validation errors
#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("Unsupported entrenar version: {0}. Supported versions: 1.0")]
    UnsupportedVersion(String),

    #[error("Empty required field: {0}")]
    EmptyRequiredField(String),

    #[error("Invalid range for {field}: {value} (expected {constraint})")]
    InvalidRange { field: String, value: String, constraint: String },

    #[error("Mutually exclusive fields specified: {field1} and {field2}")]
    MutuallyExclusive { field1: String, field2: String },

    #[error("Invalid split ratios: sum is {sum} (expected 1.0)")]
    InvalidSplitRatios { sum: f64 },

    #[error("Invalid quantization bits: {bits}. Valid values: 2, 4, 8")]
    InvalidQuantBits { bits: u8 },

    #[error("Dependency error: {0}")]
    DependencyError(String),

    #[error("Invalid optimizer: {0}")]
    InvalidOptimizer(String),

    #[error("Invalid scheduler: {0}")]
    InvalidScheduler(String),
}

/// Supported entrenar specification versions
const SUPPORTED_VERSIONS: &[&str] = &["1.0"];

/// Valid optimizer names
const VALID_OPTIMIZERS: &[&str] = &["sgd", "adam", "adamw", "rmsprop", "adagrad", "lamb"];

/// Valid scheduler names
const VALID_SCHEDULERS: &[&str] =
    &["step", "cosine", "cosine_annealing", "linear", "exponential", "plateau", "one_cycle"];

/// Valid quantization bit widths
const VALID_QUANT_BITS: &[u8] = &[2, 4, 8];

/// Validate a training manifest
///
/// Performs comprehensive validation including:
/// 1. Version compatibility
/// 2. Required fields presence
/// 3. Type constraints
/// 4. Range constraints
/// 5. Mutual exclusivity
/// 6. Cross-field dependencies
pub fn validate_manifest(manifest: &TrainingManifest) -> ValidationResult<()> {
    // 1. Version validation
    validate_version(&manifest.entrenar)?;

    // 2. Required field validation
    validate_required_fields(manifest)?;

    // 3. Optimizer validation
    if let Some(ref optim) = manifest.optimizer {
        validate_optimizer(optim)?;
    }

    // 4. Scheduler validation
    if let Some(ref sched) = manifest.scheduler {
        validate_scheduler(sched)?;
    }

    // 5. Training config validation
    if let Some(ref training) = manifest.training {
        validate_training(training)?;
    }

    // 6. Data config validation
    if let Some(ref data) = manifest.data {
        validate_data(data)?;
    }

    // 7. LoRA validation
    if let Some(ref lora) = manifest.lora {
        validate_lora(lora)?;
    }

    // 8. Quantization validation
    if let Some(ref quant) = manifest.quantize {
        validate_quantize(quant)?;
    }

    Ok(())
}

/// Validate specification version
fn validate_version(version: &str) -> ValidationResult<()> {
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(ManifestError::UnsupportedVersion(version.to_string()));
    }
    Ok(())
}

/// Validate required fields
fn validate_required_fields(manifest: &TrainingManifest) -> ValidationResult<()> {
    if manifest.name.is_empty() {
        return Err(ManifestError::EmptyRequiredField("name".to_string()));
    }

    if manifest.version.is_empty() {
        return Err(ManifestError::EmptyRequiredField("version".to_string()));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared range-check helpers (reduce nesting in callers)
// ---------------------------------------------------------------------------

/// Validate that a required f64 is strictly positive
fn validate_positive_f64(value: f64, field: &str, constraint: &str) -> ValidationResult<()> {
    if value <= 0.0 {
        return Err(ManifestError::InvalidRange {
            field: field.to_string(),
            value: value.to_string(),
            constraint: constraint.to_string(),
        });
    }
    Ok(())
}

/// Validate that an optional usize, if present, is non-zero (>= 1)
fn validate_nonzero_usize(value: Option<usize>, field: &str) -> ValidationResult<()> {
    if let Some(v) = value {
        if v == 0 {
            return Err(ManifestError::InvalidRange {
                field: field.to_string(),
                value: v.to_string(),
                constraint: ">= 1".to_string(),
            });
        }
    }
    Ok(())
}

/// Validate that an optional f64, if present, is non-negative (>= 0)
fn validate_nonneg_f64(value: Option<f64>, field: &str) -> ValidationResult<()> {
    if let Some(v) = value {
        if v < 0.0 {
            return Err(ManifestError::InvalidRange {
                field: field.to_string(),
                value: v.to_string(),
                constraint: ">= 0".to_string(),
            });
        }
    }
    Ok(())
}

/// Validate that an optional f64, if present, lies within the half-open range [0, 1)
fn validate_dropout_range(value: Option<f64>, field: &str) -> ValidationResult<()> {
    if let Some(v) = value {
        if !(0.0..1.0).contains(&v) {
            return Err(ManifestError::InvalidRange {
                field: field.to_string(),
                value: v.to_string(),
                constraint: "in [0, 1)".to_string(),
            });
        }
    }
    Ok(())
}

/// Validate that an optional u8, if present, is a valid quantization bit width
fn validate_quant_bits(bits: Option<u8>) -> ValidationResult<()> {
    if let Some(b) = bits {
        if !VALID_QUANT_BITS.contains(&b) {
            return Err(ManifestError::InvalidQuantBits { bits: b });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Optimizer validation
// ---------------------------------------------------------------------------

/// Validate optimizer configuration
fn validate_optimizer(optim: &super::manifest::OptimizerConfig) -> ValidationResult<()> {
    validate_optimizer_name(&optim.name)?;
    validate_positive_f64(optim.lr, "optimizer.lr", "> 0")?;
    validate_nonneg_f64(optim.weight_decay, "optimizer.weight_decay")?;
    validate_optimizer_betas(optim.betas.as_deref())?;
    Ok(())
}

/// Validate optimizer name against the allow-list
fn validate_optimizer_name(name: &str) -> ValidationResult<()> {
    let name_lower = name.to_lowercase();
    if !VALID_OPTIMIZERS.contains(&name_lower.as_str()) {
        return Err(ManifestError::InvalidOptimizer(format!(
            "Unknown optimizer '{name}'. Valid options: {VALID_OPTIMIZERS:?}",
        )));
    }
    Ok(())
}

/// Validate that each beta value is in the open interval (0, 1)
fn validate_optimizer_betas(betas: Option<&[f64]>) -> ValidationResult<()> {
    let Some(betas) = betas else {
        return Ok(());
    };
    for (i, beta) in betas.iter().enumerate() {
        if *beta <= 0.0 || *beta >= 1.0 {
            return Err(ManifestError::InvalidRange {
                field: format!("optimizer.betas[{i}]"),
                value: beta.to_string(),
                constraint: "in (0, 1)".to_string(),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scheduler validation
// ---------------------------------------------------------------------------

/// Validate scheduler configuration
fn validate_scheduler(sched: &super::manifest::SchedulerConfig) -> ValidationResult<()> {
    let name_lower = sched.name.to_lowercase();
    if !VALID_SCHEDULERS.contains(&name_lower.as_str()) {
        return Err(ManifestError::InvalidScheduler(format!(
            "Unknown scheduler '{}'. Valid options: {:?}",
            sched.name, VALID_SCHEDULERS
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Training validation
// ---------------------------------------------------------------------------

/// Validate training configuration
fn validate_training(training: &super::manifest::TrainingConfig) -> ValidationResult<()> {
    validate_duration_exclusivity(training)?;
    validate_nonzero_usize(training.epochs, "training.epochs")?;
    validate_gradient_config(training.gradient.as_ref())?;
    Ok(())
}

/// Ensure at most one of epochs / max_steps / duration is specified
fn validate_duration_exclusivity(
    training: &super::manifest::TrainingConfig,
) -> ValidationResult<()> {
    let has_epochs = training.epochs.is_some();
    let has_max_steps = training.max_steps.is_some();
    let has_duration = training.duration.is_some();

    if let Some((f1, f2)) = first_duration_conflict(has_epochs, has_max_steps, has_duration) {
        return Err(ManifestError::MutuallyExclusive {
            field1: f1.to_string(),
            field2: f2.to_string(),
        });
    }
    Ok(())
}

/// Return the first pair of conflicting duration fields, if any
fn first_duration_conflict(
    has_epochs: bool,
    has_max_steps: bool,
    has_duration: bool,
) -> Option<(&'static str, &'static str)> {
    if has_epochs && has_max_steps {
        return Some(("training.epochs", "training.max_steps"));
    }
    if has_epochs && has_duration {
        return Some(("training.epochs", "training.duration"));
    }
    if has_max_steps && has_duration {
        return Some(("training.max_steps", "training.duration"));
    }
    None
}

/// Validate gradient accumulation steps if present
fn validate_gradient_config(
    gradient: Option<&super::manifest::GradientConfig>,
) -> ValidationResult<()> {
    let Some(grad) = gradient else {
        return Ok(());
    };
    validate_nonzero_usize(grad.accumulation_steps, "training.gradient.accumulation_steps")
}

// ---------------------------------------------------------------------------
// Data validation
// ---------------------------------------------------------------------------

/// Validate data configuration
fn validate_data(data: &super::manifest::DataConfig) -> ValidationResult<()> {
    validate_loader_batch_size(data.loader.as_ref())?;
    validate_split_ratios(data.split.as_ref())
}

/// Validate that loader batch_size > 0
fn validate_loader_batch_size(
    loader: Option<&super::manifest::DataLoader>,
) -> ValidationResult<()> {
    let Some(loader) = loader else {
        return Ok(());
    };
    if loader.batch_size == 0 {
        return Err(ManifestError::InvalidRange {
            field: "data.loader.batch_size".to_string(),
            value: "0".to_string(),
            constraint: ">= 1".to_string(),
        });
    }
    Ok(())
}

/// Validate data split ratios sum to 1.0 and train ratio is in [0, 1]
fn validate_split_ratios(split: Option<&super::manifest::DataSplit>) -> ValidationResult<()> {
    let Some(split) = split else {
        return Ok(());
    };

    let sum = split.train + split.val.unwrap_or(0.0) + split.test.unwrap_or(0.0);

    // Allow small tolerance for floating point
    if (sum - 1.0).abs() > 0.001 {
        return Err(ManifestError::InvalidSplitRatios { sum });
    }

    // Validate individual ratios in [0, 1]
    if split.train < 0.0 || split.train > 1.0 {
        return Err(ManifestError::InvalidRange {
            field: "data.split.train".to_string(),
            value: split.train.to_string(),
            constraint: "in [0, 1]".to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LoRA validation
// ---------------------------------------------------------------------------

/// Validate LoRA configuration
fn validate_lora(lora: &super::manifest::LoraConfig) -> ValidationResult<()> {
    // Only validate if enabled
    if !lora.enabled {
        return Ok(());
    }

    validate_lora_target_modules(lora)?;
    validate_lora_rank(lora.rank)?;
    validate_positive_f64(lora.alpha, "lora.alpha", "> 0")?;
    validate_dropout_range(lora.dropout, "lora.dropout")?;
    validate_quant_bits(lora.quantize_bits)
}

/// Validate that at least one of target_modules or target_modules_pattern is provided
fn validate_lora_target_modules(lora: &super::manifest::LoraConfig) -> ValidationResult<()> {
    if lora.target_modules.is_empty() && lora.target_modules_pattern.is_none() {
        return Err(ManifestError::EmptyRequiredField("lora.target_modules".to_string()));
    }
    Ok(())
}

/// Validate LoRA rank is at least 1
fn validate_lora_rank(rank: usize) -> ValidationResult<()> {
    if rank == 0 {
        return Err(ManifestError::InvalidRange {
            field: "lora.rank".to_string(),
            value: "0".to_string(),
            constraint: ">= 1".to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Quantization validation
// ---------------------------------------------------------------------------

/// Validate quantization configuration
fn validate_quantize(quant: &super::manifest::QuantizeConfig) -> ValidationResult<()> {
    // Only validate if enabled
    if !quant.enabled {
        return Ok(());
    }

    // Validate bits
    if !VALID_QUANT_BITS.contains(&quant.bits) {
        return Err(ManifestError::InvalidQuantBits { bits: quant.bits });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_version() {
        assert!(validate_version("1.0").is_ok());
        assert!(validate_version("2.0").is_err());
    }

    #[test]
    fn test_valid_optimizers() {
        for opt in VALID_OPTIMIZERS {
            let optim = super::super::manifest::OptimizerConfig {
                name: opt.to_string(),
                lr: 0.001,
                weight_decay: None,
                betas: None,
                eps: None,
                amsgrad: None,
                momentum: None,
                nesterov: None,
                dampening: None,
                alpha: None,
                centered: None,
                param_groups: None,
            };
            assert!(validate_optimizer(&optim).is_ok(), "Optimizer {opt} should be valid");
        }
    }

    #[test]
    fn test_valid_quant_bits() {
        for bits in VALID_QUANT_BITS {
            let quant = super::super::manifest::QuantizeConfig {
                enabled: true,
                bits: *bits,
                scheme: None,
                granularity: None,
                group_size: None,
                qat: None,
                calibration: None,
                exclude: None,
            };
            assert!(validate_quantize(&quant).is_ok(), "Quant bits {bits} should be valid");
        }
    }
}
