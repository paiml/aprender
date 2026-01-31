//! APR Converter - Model Merging (APR-SPEC §4.9)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::serialization::safetensors::save_safetensors;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// Import shared function from parent module
use super::load_model_tensors;

/// Merge strategy options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Average weights (simple ensemble)
    Average,
    /// Weighted average by performance
    Weighted,
    /// TIES merging (trim, elect, sign) - advanced
    Ties,
    /// DARE merging (drop and rescale) - advanced
    Dare,
    /// Spherical linear interpolation - advanced
    Slerp,
}

impl std::str::FromStr for MergeStrategy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "avg" => Ok(Self::Average),
            "weighted" => Ok(Self::Weighted),
            "ties" => Ok(Self::Ties),
            "dare" => Ok(Self::Dare),
            "slerp" => Ok(Self::Slerp),
            _ => Err(format!("Unknown merge strategy: {s}")),
        }
    }
}

impl MergeStrategy {
    /// Check if strategy is currently supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Average | Self::Weighted)
    }
}

/// Options for model merging
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Merge strategy to use
    pub strategy: MergeStrategy,
    /// Weights for weighted merging (must match number of models)
    pub weights: Option<Vec<f32>>,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Average,
            weights: None,
        }
    }
}

/// Report from merge operation
#[derive(Debug, Clone)]
pub struct MergeReport {
    /// Number of models merged
    pub model_count: usize,
    /// Number of tensors in merged model
    pub tensor_count: usize,
    /// Output file size in bytes
    pub output_size: usize,
    /// Strategy used
    pub strategy: MergeStrategy,
    /// Weights used (if weighted merge)
    pub weights_used: Option<Vec<f32>>,
}

// ============================================================================
// MERGE HELPER FUNCTIONS (Refactored for reduced complexity)
// ============================================================================

/// Validate merge options and input count.
fn validate_merge_options<P: AsRef<Path>>(inputs: &[P], options: &MergeOptions) -> Result<()> {
    if inputs.len() < 2 {
        return Err(AprenderError::FormatError {
            message: "Merge requires at least 2 input models".to_string(),
        });
    }

    if !options.strategy.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Merge strategy {:?} is not yet supported. Use 'average' or 'weighted'.",
                options.strategy
            ),
        });
    }

    if options.strategy == MergeStrategy::Weighted {
        match &options.weights {
            Some(weights) if weights.len() != inputs.len() => {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Weighted merge requires {} weights, got {}",
                        inputs.len(),
                        weights.len()
                    ),
                });
            }
            None => {
                return Err(AprenderError::FormatError {
                    message: "Weighted merge requires weights to be specified".to_string(),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Load all model tensors from input files.
fn load_all_models<P: AsRef<Path>>(
    inputs: &[P],
) -> Result<Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>>> {
    let mut all_tensors = Vec::new();
    for input_path in inputs {
        let path = input_path.as_ref();
        if !path.exists() {
            return Err(AprenderError::FormatError {
                message: format!("Input file not found: {}", path.display()),
            });
        }
        all_tensors.push(load_model_tensors(path)?);
    }
    Ok(all_tensors)
}

/// Verify all models have compatible tensor structures.
fn verify_tensor_compatibility(
    all_tensors: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
) -> Result<()> {
    let reference = &all_tensors[0];
    for (i, tensors) in all_tensors.iter().enumerate().skip(1) {
        if tensors.len() != reference.len() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Model {} has {} tensors, but model 0 has {}",
                    i,
                    tensors.len(),
                    reference.len()
                ),
            });
        }
        verify_single_model_tensors(reference, tensors, i)?;
    }
    Ok(())
}

/// Verify tensor compatibility for a single model against reference.
fn verify_single_model_tensors(
    reference: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    model_idx: usize,
) -> Result<()> {
    for (name, (_, shape)) in reference {
        match tensors.get(name) {
            None => {
                return Err(AprenderError::FormatError {
                    message: format!("Model {} is missing tensor '{}'", model_idx, name),
                });
            }
            Some((_, other_shape)) if other_shape != shape => {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Tensor '{}' has shape {:?} in model 0 but {:?} in model {}",
                        name, shape, other_shape, model_idx
                    ),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Calculate normalized merge weights based on strategy.
pub(crate) fn calculate_merge_weights(
    input_count: usize,
    options: &MergeOptions,
) -> Result<Vec<f32>> {
    match options.strategy {
        MergeStrategy::Average => {
            let w = 1.0 / input_count as f32;
            Ok(vec![w; input_count])
        }
        MergeStrategy::Weighted => {
            let raw_weights = options.weights.as_ref().expect("validated above");
            let sum: f32 = raw_weights.iter().sum();
            if sum <= 0.0 {
                return Err(AprenderError::FormatError {
                    message: "Weights must sum to a positive value".to_string(),
                });
            }
            Ok(raw_weights.iter().map(|w| w / sum).collect())
        }
        _ => unreachable!("unsupported strategies filtered above"),
    }
}

/// Merge tensors from multiple models using given weights.
fn merge_tensors(
    all_tensors: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    weights: &[f32],
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let reference = &all_tensors[0];
    let mut merged = BTreeMap::new();

    for (name, (_, shape)) in reference {
        let data_len = all_tensors[0].get(name).map(|(d, _)| d.len()).unwrap_or(0);
        let mut merged_data = vec![0.0f32; data_len];

        for (model_idx, model_tensors) in all_tensors.iter().enumerate() {
            let (data, _) = model_tensors.get(name).expect("validated above");
            let weight = weights[model_idx];
            for (i, &val) in data.iter().enumerate() {
                merged_data[i] += val * weight;
            }
        }

        merged.insert(name.clone(), (merged_data, shape.clone()));
    }
    merged
}

/// Merge multiple models into one
///
/// # Arguments
///
/// * `inputs` - Input model paths (.apr or .safetensors)
/// * `output` - Output file path
/// * `options` - Merge options
///
/// # Returns
///
/// Merge report with statistics
///
/// # Errors
///
/// Returns error if:
/// - Less than 2 input files
/// - Input files don't exist
/// - Models have incompatible tensor shapes
/// - Strategy not supported
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{apr_merge, MergeOptions, MergeStrategy};
///
/// let options = MergeOptions {
///     strategy: MergeStrategy::Average,
///     weights: None,
/// };
/// let report = apr_merge(&["model1.apr", "model2.apr"], "merged.apr", options)?;
/// ```
pub fn apr_merge<P: AsRef<Path>>(
    inputs: &[P],
    output: P,
    options: MergeOptions,
) -> Result<MergeReport> {
    // Validate inputs and options
    validate_merge_options(inputs, &options)?;

    // Load all models
    let all_tensors = load_all_models(inputs)?;

    // Verify tensor compatibility
    verify_tensor_compatibility(&all_tensors)?;

    // Calculate weights
    let weights = calculate_merge_weights(inputs.len(), &options)?;

    // Merge tensors
    let merged = merge_tensors(&all_tensors, &weights);

    // Save merged model
    let output_path = output.as_ref();
    save_safetensors(output_path, &merged).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to save merged model: {e}"),
    })?;

    // Get output file size
    let output_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    Ok(MergeReport {
        model_count: inputs.len(),
        tensor_count: merged.len(),
        output_size,
        strategy: options.strategy,
        weights_used: Some(weights),
    })
}

// ============================================================================
