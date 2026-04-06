//! Model merging methods (TIES, DARE, SLERP)
//!
//! This module provides three model merging algorithms for combining
//! multiple fine-tuned models:
//!
//! - **TIES**: Task Inference via Elimination and Sign voting
//! - **DARE**: Drop And REscale for stochastic merging
//! - **SLERP**: Spherical Linear intERPolation for smooth blending

mod dare;
mod ensemble;
mod slerp;
mod ties;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod commutativity;

pub use dare::{dare_merge, DareConfig};
pub use ensemble::{ensemble_merge, EnsembleConfig, EnsembleStrategy};
pub use slerp::{slerp_merge, SlerpConfig};
pub use ties::{ties_merge, TiesConfig};

use crate::autograd::Tensor;
use std::collections::HashMap;

/// A model represented as a collection of named tensors
pub type Model = HashMap<String, Tensor>;

/// Error types for model merging operations
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    #[error("Models have incompatible architectures: {0}")]
    IncompatibleArchitectures(String),

    #[error("Parameter {0} has mismatched shapes")]
    ShapeMismatch(String),

    #[error("Invalid merge configuration: {0}")]
    InvalidConfig(String),

    #[error("Insufficient models provided: need at least {min}, got {got}")]
    InsufficientModels { min: usize, got: usize },
}

/// Compute delta weights (model - base) for each model
pub(crate) fn compute_deltas(models: &[Model], base: &Model) -> Result<Vec<Model>, MergeError> {
    models
        .iter()
        .map(|model| {
            let mut delta = HashMap::new();
            for (name, tensor) in model {
                let base_tensor = base.get(name).ok_or_else(|| {
                    MergeError::IncompatibleArchitectures(format!(
                        "Base model missing parameter: {name}"
                    ))
                })?;

                if tensor.len() != base_tensor.len() {
                    return Err(MergeError::ShapeMismatch(name.clone()));
                }

                // Delta = model - base
                let delta_data = tensor.data() - base_tensor.data();
                delta.insert(name.clone(), Tensor::new(delta_data, false));
            }
            Ok(delta)
        })
        .collect()
}

/// Merge deltas back with base model
pub(crate) fn merge_with_base(base: &Model, delta: Model) -> Model {
    let mut merged = HashMap::new();
    for (name, base_tensor) in base {
        if let Some(delta_tensor) = delta.get(name) {
            let merged_data = base_tensor.data() + delta_tensor.data();
            merged.insert(name.clone(), Tensor::new(merged_data, false));
        } else {
            merged.insert(name.clone(), base_tensor.clone());
        }
    }
    merged
}

/// Validate that all models have compatible architectures
pub(crate) fn validate_models(models: &[Model]) -> Result<(), MergeError> {
    if models.is_empty() {
        return Err(MergeError::InsufficientModels { min: 1, got: 0 });
    }

    let reference = &models[0];
    for (i, model) in models.iter().enumerate().skip(1) {
        validate_model_keys(reference, model, i)?;
        validate_model_shapes(reference, model, i)?;
    }

    Ok(())
}

/// Check all reference parameters exist in the model.
fn validate_model_keys(reference: &Model, model: &Model, idx: usize) -> Result<(), MergeError> {
    for name in reference.keys() {
        if !model.contains_key(name) {
            return Err(MergeError::IncompatibleArchitectures(format!(
                "Model {idx} missing parameter: {name}"
            )));
        }
    }
    Ok(())
}

/// Check all parameter shapes match between reference and model.
fn validate_model_shapes(reference: &Model, model: &Model, idx: usize) -> Result<(), MergeError> {
    for (name, ref_tensor) in reference {
        let model_tensor = &model[name];
        if ref_tensor.len() != model_tensor.len() {
            return Err(MergeError::ShapeMismatch(format!(
                "{} (model 0: {}, model {}: {})",
                name,
                ref_tensor.len(),
                idx,
                model_tensor.len()
            )));
        }
    }
    Ok(())
}
