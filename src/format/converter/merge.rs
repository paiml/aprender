//! APR Converter - Model Merging (APR-SPEC §4.9)
//! PMAT-197: Extracted from mod.rs for file size reduction
//!
//! Supports 5 merge strategies:
//! - Average: Simple average of weights
//! - Weighted: Weighted average by specified weights
//! - SLERP: Spherical linear interpolation (2 models only)
//! - TIES: Trim, Elect Sign, Merge (requires base model)
//! - DARE: Drop And Rescale (requires base model)

use crate::error::{AprenderError, Result};
use crate::serialization::safetensors::save_safetensors;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

// Import shared function from parent module
use super::load_model_tensors;

/// Merge strategy options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Average weights (simple ensemble)
    Average,
    /// Weighted average by performance
    Weighted,
    /// TIES merging (trim, elect, sign)
    Ties,
    /// DARE merging (drop and rescale)
    Dare,
    /// Spherical linear interpolation
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
        matches!(
            self,
            Self::Average | Self::Weighted | Self::Slerp | Self::Ties | Self::Dare
        )
    }
}

/// Options for model merging
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Merge strategy to use
    pub strategy: MergeStrategy,
    /// Weights for weighted merging (must match number of models)
    pub weights: Option<Vec<f32>>,
    /// Base model path for TIES/DARE (task vectors computed as delta from base)
    pub base_model: Option<PathBuf>,
    /// DARE drop probability — fraction of delta elements to zero out (default 0.9)
    pub drop_rate: f32,
    /// TIES trim density — elements with |delta| below density * max(|delta|) are trimmed (default 0.2)
    pub density: f32,
    /// RNG seed for DARE deterministic dropping (default 42)
    pub seed: u64,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Average,
            weights: None,
            base_model: None,
            drop_rate: 0.9,
            density: 0.2,
            seed: 42,
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

    validate_strategy_specific(inputs, options)
}

/// Validate strategy-specific constraints.
fn validate_strategy_specific<P: AsRef<Path>>(
    inputs: &[P],
    options: &MergeOptions,
) -> Result<()> {
    match options.strategy {
        MergeStrategy::Weighted => validate_weighted_options(inputs, options),
        MergeStrategy::Slerp => validate_slerp_options(inputs),
        MergeStrategy::Ties => validate_ties_dare_options(options, "TIES"),
        MergeStrategy::Dare => validate_ties_dare_options(options, "DARE"),
        MergeStrategy::Average => Ok(()),
    }
}

/// Validate weighted merge options.
fn validate_weighted_options<P: AsRef<Path>>(
    inputs: &[P],
    options: &MergeOptions,
) -> Result<()> {
    match &options.weights {
        Some(weights) if weights.len() != inputs.len() => Err(AprenderError::FormatError {
            message: format!(
                "Weighted merge requires {} weights, got {}",
                inputs.len(),
                weights.len()
            ),
        }),
        None => Err(AprenderError::FormatError {
            message: "Weighted merge requires weights to be specified".to_string(),
        }),
        _ => Ok(()),
    }
}

/// Validate SLERP requires exactly 2 models.
fn validate_slerp_options<P: AsRef<Path>>(inputs: &[P]) -> Result<()> {
    if inputs.len() != 2 {
        return Err(AprenderError::FormatError {
            message: format!(
                "SLERP merge requires exactly 2 input models, got {}",
                inputs.len()
            ),
        });
    }
    Ok(())
}

/// Validate TIES/DARE require base model and valid parameters.
fn validate_ties_dare_options(options: &MergeOptions, name: &str) -> Result<()> {
    if options.base_model.is_none() {
        return Err(AprenderError::FormatError {
            message: format!(
                "{name} merge requires --base-model to compute task vectors"
            ),
        });
    }
    let base = options.base_model.as_ref().expect("checked above");
    if !base.exists() {
        return Err(AprenderError::FormatError {
            message: format!("Base model not found: {}", base.display()),
        });
    }
    if name == "DARE" && (options.drop_rate <= 0.0 || options.drop_rate >= 1.0) {
        return Err(AprenderError::FormatError {
            message: format!(
                "DARE drop_rate must be in (0, 1), got {}",
                options.drop_rate
            ),
        });
    }
    if name == "TIES" && (options.density <= 0.0 || options.density >= 1.0) {
        return Err(AprenderError::FormatError {
            message: format!(
                "TIES density must be in (0, 1), got {}",
                options.density
            ),
        });
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

            // BUG-MERGE-006 FIX: Validate each weight is finite before summing.
            for (i, &w) in raw_weights.iter().enumerate() {
                if !w.is_finite() {
                    return Err(AprenderError::FormatError {
                        message: format!(
                            "Weight {} is not finite ({}). All weights must be finite values.",
                            i, w
                        ),
                    });
                }
            }

            let sum: f32 = raw_weights.iter().sum();
            // BUG-MERGE-006 FIX: Also check sum is finite (overflow protection)
            if sum <= 0.0 || !sum.is_finite() {
                return Err(AprenderError::FormatError {
                    message: "Weights must sum to a finite positive value".to_string(),
                });
            }
            Ok(raw_weights.iter().map(|w| w / sum).collect())
        }
        // TIES/DARE/SLERP handle weights internally
        _ => Ok(vec![1.0 / input_count as f32; input_count]),
    }
}

/// Merge tensors from multiple models using given weights (average/weighted).
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

// ============================================================================
// SLERP (Spherical Linear Interpolation)
// ============================================================================

/// SLERP interpolation between two tensors.
///
/// For each tensor, treats the flattened data as a vector and computes
/// slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
/// where omega = acos(dot(norm_a, norm_b)).
///
/// Falls back to lerp when vectors are nearly parallel (omega < epsilon).
fn slerp_tensors(
    model_a: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    model_b: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    t: f32,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut merged = BTreeMap::new();

    for (name, (data_a, shape)) in model_a {
        let (data_b, _) = model_b.get(name).expect("validated above");
        let merged_data = slerp_vectors(data_a, data_b, t);
        merged.insert(name.clone(), (merged_data, shape.clone()));
    }

    merged
}

/// SLERP between two flat f32 vectors.
fn slerp_vectors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let norm_a = vector_norm(a);
    let norm_b = vector_norm(b);

    // Degenerate: zero-norm vector — fall back to lerp
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return lerp_vectors(a, b, t);
    }

    // Compute cosine similarity via normalized dot product
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let cos_omega = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    // Nearly parallel — fall back to lerp to avoid division by ~zero
    if omega.abs() < 1e-6 {
        return lerp_vectors(a, b, t);
    }

    let sin_omega = omega.sin();
    let t64 = f64::from(t);
    let coeff_a = ((1.0 - t64) * omega).sin() / sin_omega;
    let coeff_b = (t64 * omega).sin() / sin_omega;

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (coeff_a * f64::from(x) + coeff_b * f64::from(y)) as f32)
        .collect()
}

/// Linear interpolation between two vectors: (1-t)*a + t*b.
fn lerp_vectors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * (1.0 - t) + y * t)
        .collect()
}

/// L2 norm of a vector (computed in f64 for precision).
fn vector_norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

include!("ties_merge.rs");
