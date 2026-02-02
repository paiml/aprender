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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialization::safetensors::save_safetensors;
    use tempfile::tempdir;

    /// Create a test model file with given tensors
    fn create_test_model(
        path: &Path,
        tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<()> {
        save_safetensors(path, tensors).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to create test model: {e}"),
        })
    }

    #[test]
    fn test_merge_strategy_from_str_average() {
        assert_eq!(
            "average".parse::<MergeStrategy>().unwrap(),
            MergeStrategy::Average
        );
        assert_eq!(
            "avg".parse::<MergeStrategy>().unwrap(),
            MergeStrategy::Average
        );
        assert_eq!(
            "AVERAGE".parse::<MergeStrategy>().unwrap(),
            MergeStrategy::Average
        );
    }

    #[test]
    fn test_merge_strategy_from_str_weighted() {
        assert_eq!(
            "weighted".parse::<MergeStrategy>().unwrap(),
            MergeStrategy::Weighted
        );
    }

    #[test]
    fn test_merge_strategy_from_str_advanced() {
        assert_eq!(
            "ties".parse::<MergeStrategy>().unwrap(),
            MergeStrategy::Ties
        );
        assert_eq!(
            "dare".parse::<MergeStrategy>().unwrap(),
            MergeStrategy::Dare
        );
        assert_eq!(
            "slerp".parse::<MergeStrategy>().unwrap(),
            MergeStrategy::Slerp
        );
    }

    #[test]
    fn test_merge_strategy_from_str_unknown() {
        let result = "unknown_strategy".parse::<MergeStrategy>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown merge strategy"));
    }

    #[test]
    fn test_merge_strategy_is_supported() {
        assert!(MergeStrategy::Average.is_supported());
        assert!(MergeStrategy::Weighted.is_supported());
        assert!(!MergeStrategy::Ties.is_supported());
        assert!(!MergeStrategy::Dare.is_supported());
        assert!(!MergeStrategy::Slerp.is_supported());
    }

    #[test]
    fn test_merge_options_default() {
        let options = MergeOptions::default();
        assert_eq!(options.strategy, MergeStrategy::Average);
        assert!(options.weights.is_none());
    }

    #[test]
    fn test_validate_merge_options_less_than_2_inputs() {
        let options = MergeOptions::default();
        let result = validate_merge_options(&["single.safetensors"], &options);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("at least 2"));
    }

    #[test]
    fn test_validate_merge_options_unsupported_strategy() {
        let options = MergeOptions {
            strategy: MergeStrategy::Ties,
            weights: None,
        };
        let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("not yet supported"));
    }

    #[test]
    fn test_validate_merge_options_weighted_missing_weights() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: None,
        };
        let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("requires weights"));
    }

    #[test]
    fn test_validate_merge_options_weighted_wrong_count() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.5, 0.3, 0.2]), // 3 weights but 2 inputs
        };
        let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("requires 2 weights"));
    }

    #[test]
    fn test_validate_merge_options_weighted_valid() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.6, 0.4]),
        };
        let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_all_models_file_not_found() {
        let result = load_all_models(&["/nonexistent/path/model.safetensors"]);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_calculate_merge_weights_average() {
        let options = MergeOptions::default();
        let weights = calculate_merge_weights(4, &options).unwrap();
        assert_eq!(weights.len(), 4);
        for w in &weights {
            assert!((*w - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_calculate_merge_weights_weighted_normalized() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![2.0, 3.0]),
        };
        let weights = calculate_merge_weights(2, &options).unwrap();
        assert!((weights[0] - 0.4).abs() < 1e-6);
        assert!((weights[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_merge_weights_zero_sum() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.0, 0.0]),
        };
        let result = calculate_merge_weights(2, &options);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("positive value"));
    }

    #[test]
    fn test_calculate_merge_weights_negative_sum() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![-1.0, 0.5]),
        };
        let result = calculate_merge_weights(2, &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_tensor_compatibility_different_tensor_count() {
        let mut tensors1 = BTreeMap::new();
        tensors1.insert("a".to_string(), (vec![1.0], vec![1]));
        tensors1.insert("b".to_string(), (vec![2.0], vec![1]));

        let mut tensors2 = BTreeMap::new();
        tensors2.insert("a".to_string(), (vec![1.0], vec![1]));
        // Missing tensor "b"

        let all_tensors = vec![tensors1, tensors2];
        let result = verify_tensor_compatibility(&all_tensors);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("tensors"));
    }

    #[test]
    fn test_verify_single_model_tensors_missing() {
        let mut reference = BTreeMap::new();
        reference.insert("weight".to_string(), (vec![1.0, 2.0], vec![2]));
        reference.insert("bias".to_string(), (vec![0.5], vec![1]));

        let mut other = BTreeMap::new();
        other.insert("weight".to_string(), (vec![1.0, 2.0], vec![2]));
        // Missing "bias"

        let result = verify_single_model_tensors(&reference, &other, 1);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("missing tensor"));
    }

    #[test]
    fn test_verify_single_model_tensors_shape_mismatch() {
        let mut reference = BTreeMap::new();
        reference.insert("weight".to_string(), (vec![1.0, 2.0], vec![2]));

        let mut other = BTreeMap::new();
        other.insert("weight".to_string(), (vec![1.0, 2.0, 3.0], vec![3]));

        let result = verify_single_model_tensors(&reference, &other, 1);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("shape"));
    }

    #[test]
    fn test_merge_tensors_basic() {
        let mut tensors1 = BTreeMap::new();
        tensors1.insert("w".to_string(), (vec![1.0, 2.0], vec![2]));

        let mut tensors2 = BTreeMap::new();
        tensors2.insert("w".to_string(), (vec![3.0, 4.0], vec![2]));

        let all_tensors = vec![tensors1, tensors2];
        let weights = vec![0.5, 0.5];

        let merged = merge_tensors(&all_tensors, &weights);
        let (data, shape) = merged.get("w").unwrap();
        assert_eq!(shape, &vec![2]);
        assert!((data[0] - 2.0).abs() < 1e-6); // (1.0*0.5 + 3.0*0.5)
        assert!((data[1] - 3.0).abs() < 1e-6); // (2.0*0.5 + 4.0*0.5)
    }

    #[test]
    fn test_apr_merge_average() {
        let dir = tempdir().unwrap();
        let model1_path = dir.path().join("model1.safetensors");
        let model2_path = dir.path().join("model2.safetensors");
        let output_path = dir.path().join("merged.safetensors");

        // Create test models
        let mut tensors1 = BTreeMap::new();
        tensors1.insert(
            "layer.weight".to_string(),
            (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        let mut tensors2 = BTreeMap::new();
        tensors2.insert(
            "layer.weight".to_string(),
            (vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
        );

        create_test_model(&model1_path, &tensors1).unwrap();
        create_test_model(&model2_path, &tensors2).unwrap();

        let options = MergeOptions::default();
        let report = apr_merge(&[&model1_path, &model2_path], &output_path, options).unwrap();

        assert_eq!(report.model_count, 2);
        assert_eq!(report.tensor_count, 1);
        assert!(report.output_size > 0);
        assert_eq!(report.strategy, MergeStrategy::Average);
        assert!(report.weights_used.is_some());
    }

    #[test]
    fn test_apr_merge_weighted() {
        let dir = tempdir().unwrap();
        let model1_path = dir.path().join("model1.safetensors");
        let model2_path = dir.path().join("model2.safetensors");
        let output_path = dir.path().join("merged.safetensors");

        let mut tensors1 = BTreeMap::new();
        tensors1.insert("w".to_string(), (vec![0.0], vec![1]));

        let mut tensors2 = BTreeMap::new();
        tensors2.insert("w".to_string(), (vec![10.0], vec![1]));

        create_test_model(&model1_path, &tensors1).unwrap();
        create_test_model(&model2_path, &tensors2).unwrap();

        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.3, 0.7]),
        };
        let report = apr_merge(&[&model1_path, &model2_path], &output_path, options).unwrap();

        assert_eq!(report.strategy, MergeStrategy::Weighted);
        // Verify the merged output exists
        assert!(output_path.exists());
    }

    #[test]
    fn test_merge_report_debug_clone() {
        let report = MergeReport {
            model_count: 2,
            tensor_count: 5,
            output_size: 1024,
            strategy: MergeStrategy::Average,
            weights_used: Some(vec![0.5, 0.5]),
        };
        let debug = format!("{:?}", report);
        assert!(debug.contains("MergeReport"));
        let cloned = report.clone();
        assert_eq!(cloned.model_count, report.model_count);
    }

    #[test]
    fn test_merge_strategy_debug_clone() {
        let strategy = MergeStrategy::Ties;
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("Ties"));
        let cloned = strategy;
        assert_eq!(cloned, MergeStrategy::Ties);
    }

    #[test]
    fn test_merge_options_debug_clone() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.6, 0.4]),
        };
        let debug = format!("{:?}", options);
        assert!(debug.contains("MergeOptions"));
        let cloned = options.clone();
        assert_eq!(cloned.strategy, MergeStrategy::Weighted);
    }
}

// ============================================================================
