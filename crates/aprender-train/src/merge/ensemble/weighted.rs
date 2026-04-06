//! ENT-032: Weighted average merging

use super::{MergeError, Model};
use std::collections::HashMap;

/// Weighted average of multiple models
pub fn weighted_average_merge(models: &[Model], weights: &[f32]) -> Result<Model, MergeError> {
    let weights = normalize_weights(weights, models.len())?;
    let reference = &models[0];
    let mut merged = HashMap::new();

    for name in reference.keys() {
        let tensor = weighted_sum_param(name, models, &weights)?;
        merged.insert(name.clone(), tensor);
    }

    Ok(merged)
}

/// Normalize weights or create uniform weights.
fn normalize_weights(weights: &[f32], n: usize) -> Result<Vec<f32>, MergeError> {
    if weights.is_empty() {
        return Ok(vec![1.0 / n as f32; n]);
    }
    if weights.len() != n {
        return Err(MergeError::InvalidConfig(format!(
            "Weights length {} doesn't match models length {n}",
            weights.len(),
        )));
    }
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 {
        return Err(MergeError::InvalidConfig("Weights must sum to positive value".to_string()));
    }
    Ok(weights.iter().map(|w| w / sum).collect())
}

/// Compute weighted sum for a single parameter across models.
fn weighted_sum_param(
    name: &str,
    models: &[Model],
    weights: &[f32],
) -> Result<crate::autograd::Tensor, MergeError> {
    let param_len = models[0][name].len();
    let mut weighted_sum = ndarray::Array1::<f32>::zeros(param_len);

    for (model, weight) in models.iter().zip(weights.iter()) {
        let param = model
            .get(name)
            .ok_or_else(|| MergeError::IncompatibleArchitectures(format!("Missing {name}")))?;
        if param.len() != param_len {
            return Err(MergeError::ShapeMismatch(name.to_string()));
        }
        weighted_sum = weighted_sum + param.data() * *weight;
    }

    Ok(crate::autograd::Tensor::new(weighted_sum, false))
}
