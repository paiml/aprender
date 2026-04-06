//! TIES (Task Inference via Elimination and Sign) merge algorithm
//!
//! TIES merges multiple fine-tuned models by:
//! 1. **Trim**: Keep only top-k% magnitude parameters (eliminate noise)
//! 2. **Elect Sign**: Majority vote per parameter to resolve conflicts
//! 3. **Merge**: Average parameters with elected sign

use super::{compute_deltas, merge_with_base, validate_models, MergeError, Model};
use crate::autograd::Tensor;
use ndarray::Array1;
use std::collections::HashMap;

/// Configuration for TIES merge
#[derive(Clone, Debug)]
pub struct TiesConfig {
    /// Density parameter: fraction of parameters to keep (0.0 to 1.0)
    /// Higher density = keep more parameters = less aggressive trimming
    /// Typical values: 0.2 (20% kept) to 0.5 (50% kept)
    pub density: f32,
}

impl Default for TiesConfig {
    fn default() -> Self {
        Self { density: 0.2 }
    }
}

impl TiesConfig {
    pub fn new(density: f32) -> Result<Self, MergeError> {
        if !(0.0..=1.0).contains(&density) {
            return Err(MergeError::InvalidConfig(format!(
                "Density must be in [0.0, 1.0], got {density}"
            )));
        }
        Ok(Self { density })
    }
}

/// TIES merge: trim, elect sign, merge same-sign parameters
///
/// # Arguments
/// * `models` - Fine-tuned models to merge
/// * `base` - Base model (pre-fine-tuning checkpoint)
/// * `config` - TIES configuration (density parameter)
///
/// # Returns
/// Merged model combining all input models
///
/// # Algorithm
/// 1. Compute deltas: Δᵢ = model_i - base
/// 2. Trim: Keep only top-k% magnitude values per delta
/// 3. Elect sign: For each parameter, take sign of majority
/// 4. Merge: Average trimmed deltas with elected sign
/// 5. Add back to base: merged = base + averaged_delta
pub fn ties_merge(
    models: &[Model],
    base: &Model,
    config: &TiesConfig,
) -> Result<Model, MergeError> {
    if models.len() < 2 {
        return Err(MergeError::InsufficientModels { min: 2, got: models.len() });
    }

    validate_models(models)?;

    // Step 1: Compute deltas (model - base)
    let deltas = compute_deltas(models, base)?;

    // Step 2: Trim deltas (keep top-k% magnitude)
    let trimmed_deltas = trim_deltas(&deltas, config.density);

    // Step 3 & 4: Elect sign and merge
    let merged_delta = elect_and_merge(&trimmed_deltas);

    // Step 5: Add back to base
    Ok(merge_with_base(base, merged_delta))
}

/// Trim each delta to keep only top-k% magnitude parameters
fn trim_deltas(deltas: &[Model], density: f32) -> Vec<Model> {
    deltas
        .iter()
        .map(|delta| {
            let mut trimmed = HashMap::new();
            for (name, tensor) in delta {
                trimmed.insert(name.clone(), trim_tensor(tensor, density));
            }
            trimmed
        })
        .collect()
}

/// Trim a single tensor to keep only top-k% magnitude values
fn trim_tensor(tensor: &Tensor, density: f32) -> Tensor {
    let data = tensor.data();
    let n = data.len();
    let k = ((n as f32 * density).ceil() as usize).max(1).min(n);

    // Get magnitude-sorted indices
    let mut indices_and_magnitudes: Vec<(usize, f32)> =
        data.iter().enumerate().map(|(i, &val)| (i, val.abs())).collect();

    indices_and_magnitudes.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Keep top-k magnitude values, zero out rest
    let mut trimmed_data = Array1::zeros(n);
    for (idx, _) in indices_and_magnitudes.iter().take(k) {
        trimmed_data[*idx] = data[*idx];
    }

    Tensor::new(trimmed_data, false)
}

/// Elect sign per parameter and merge same-sign values
fn elect_and_merge(trimmed_deltas: &[Model]) -> Model {
    if trimmed_deltas.is_empty() {
        return HashMap::new();
    }

    let reference = &trimmed_deltas[0];
    let mut merged = HashMap::new();

    for name in reference.keys() {
        // Collect all delta values for this parameter
        let all_values: Vec<&Array1<f32>> =
            trimmed_deltas.iter().map(|delta| delta[name].data()).collect();

        let merged_tensor = elect_and_merge_parameter(&all_values);
        merged.insert(name.clone(), merged_tensor);
    }

    merged
}

/// Elect sign and merge for a single parameter across all models
fn elect_and_merge_parameter(values: &[&Array1<f32>]) -> Tensor {
    let n = values[0].len();
    let mut merged_data = Array1::zeros(n);

    for i in 0..n {
        // Count positive and negative non-zero values
        let (pos_sum, pos_count, neg_sum, neg_count) = values.iter().fold(
            (0.0f32, 0usize, 0.0f32, 0usize),
            |(pos_sum, pos_count, neg_sum, neg_count), arr| {
                let val = arr[i];
                if val > 0.0 {
                    (pos_sum + val, pos_count + 1, neg_sum, neg_count)
                } else if val < 0.0 {
                    (pos_sum, pos_count, neg_sum + val, neg_count + 1)
                } else {
                    (pos_sum, pos_count, neg_sum, neg_count)
                }
            },
        );

        // Elect sign by majority vote (most non-zero contributors)
        // Merge by averaging same-sign values
        merged_data[i] = if pos_count > neg_count {
            // Positive wins: average positive values only
            if pos_count > 0 {
                pos_sum / pos_count as f32
            } else {
                0.0
            }
        } else if neg_count > pos_count {
            // Negative wins: average negative values only
            if neg_count > 0 {
                neg_sum / neg_count as f32
            } else {
                0.0
            }
        } else {
            // Tie or all zero: take overall average
            let total = pos_sum + neg_sum;
            let total_count = pos_count + neg_count;
            if total_count > 0 {
                total / total_count as f32
            } else {
                0.0
            }
        };
    }

    Tensor::new(merged_data, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_trim_tensor_keeps_top_k() {
        let tensor = Tensor::from_vec(vec![1.0, -5.0, 2.0, -0.1, 3.0], false);
        let trimmed = trim_tensor(&tensor, 0.4); // Keep top 40% = 2 values

        // Should keep -5.0 and 3.0 (highest magnitudes)
        let data = trimmed.data();
        assert_eq!(data[0], 0.0); // 1.0 trimmed
        assert_eq!(data[1], -5.0); // kept
        assert_eq!(data[2], 0.0); // 2.0 trimmed
        assert_eq!(data[3], 0.0); // -0.1 trimmed
        assert_eq!(data[4], 3.0); // kept
    }

    #[test]
    fn test_elect_and_merge_parameter_majority_positive() {
        let v1 = Array1::from(vec![1.0, -1.0, 0.0]);
        let v2 = Array1::from(vec![2.0, 0.0, 1.0]);
        let v3 = Array1::from(vec![3.0, -2.0, 0.0]);

        let result = elect_and_merge_parameter(&[&v1, &v2, &v3]);

        // Index 0: 3 positive votes -> average (1+2+3)/3 = 2.0
        assert!((result.data()[0] - 2.0).abs() < 1e-6);

        // Index 1: 2 negative votes -> average (-1-2)/2 = -1.5
        assert!((result.data()[1] - (-1.5)).abs() < 1e-6);

        // Index 2: 1 positive vote -> 1.0
        assert!((result.data()[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ties_config_validation() {
        assert!(TiesConfig::new(0.5).is_ok());
        assert!(TiesConfig::new(0.0).is_ok());
        assert!(TiesConfig::new(1.0).is_ok());
        assert!(TiesConfig::new(-0.1).is_err());
        assert!(TiesConfig::new(1.1).is_err());
    }

    #[test]
    fn test_ties_merge_insufficient_models() {
        let mut base = HashMap::new();
        base.insert("w".to_string(), Tensor::from_vec(vec![0.0], false));

        let models = vec![base.clone()];
        let config = TiesConfig::default();

        let result = ties_merge(&models, &base, &config);
        assert!(matches!(result, Err(MergeError::InsufficientModels { min: 2, got: 1 })));
    }

    #[test]
    fn test_trim_tensor_density_zero() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
        let trimmed = trim_tensor(&tensor, 0.0);

        // Density 0 should still keep at least 1 value (the maximum)
        let data = trimmed.data();
        let non_zero_count = data.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count >= 1);
    }

    #[test]
    fn test_trim_tensor_density_one() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
        let trimmed = trim_tensor(&tensor, 1.0);

        // Density 1.0 should keep all values
        let data = trimmed.data();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 5.0);
    }

    #[test]
    fn test_elect_sign_tie_breaker() {
        // Test tie case: equal positive and negative votes
        let v1 = Array1::from(vec![1.0]);
        let v2 = Array1::from(vec![-1.0]);

        let result = elect_and_merge_parameter(&[&v1, &v2]);

        // Tie: should average all values (1 + -1) / 2 = 0
        assert!((result.data()[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_elect_sign_all_zeros() {
        let v1 = Array1::from(vec![0.0, 0.0]);
        let v2 = Array1::from(vec![0.0, 0.0]);

        let result = elect_and_merge_parameter(&[&v1, &v2]);

        assert_eq!(result.data()[0], 0.0);
        assert_eq!(result.data()[1], 0.0);
    }

    #[test]
    fn test_ties_merge_two_models() {
        let mut base = HashMap::new();
        base.insert("w".to_string(), Tensor::from_vec(vec![0.0, 0.0, 0.0], false));

        let mut model1 = HashMap::new();
        model1.insert("w".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], false));

        let mut model2 = HashMap::new();
        model2.insert("w".to_string(), Tensor::from_vec(vec![2.0, -1.0, 4.0], false));

        let config = TiesConfig::new(1.0).expect("config should be valid"); // Keep all
        let result = ties_merge(&[model1, model2], &base, &config).expect("config should be valid");

        // Both positive at index 0: average (1+2)/2 = 1.5
        // Mixed at index 1: pos=2, neg=-1, pos wins -> 2.0
        // Both positive at index 2: average (3+4)/2 = 3.5
        let w = result.get("w").expect("key should exist");
        assert!((w.data()[0] - 1.5).abs() < 1e-6);
        assert!((w.data()[2] - 3.5).abs() < 1e-6);
    }

    // Property tests

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_trim_preserves_top_k_count(
            values in proptest::collection::vec(-100.0f32..100.0, 10..50),
            density in 0.1f32..1.0
        ) {
            let tensor = Tensor::from_vec(values.clone(), false);
            let trimmed = trim_tensor(&tensor, density);

            let expected_k = ((values.len() as f32 * density).ceil() as usize).max(1).min(values.len());
            let actual_nonzero = trimmed.data().iter().filter(|&&x| x != 0.0).count();

            // Should keep approximately expected_k values (exact for non-zero inputs)
            prop_assert!(actual_nonzero <= expected_k + 1);
        }

        #[test]
        fn prop_trim_keeps_highest_magnitudes(
            values in proptest::collection::vec(-100.0f32..100.0, 5..20),
            density in 0.3f32..0.7
        ) {
            let tensor = Tensor::from_vec(values.clone(), false);
            let trimmed = trim_tensor(&tensor, density);

            // Find the minimum magnitude among kept values
            let kept_magnitudes: Vec<f32> = trimmed.data()
                .iter()
                .filter(|&&x| x != 0.0)
                .map(|x| x.abs())
                .collect();

            if !kept_magnitudes.is_empty() {
                let min_kept = kept_magnitudes.iter().copied().fold(f32::INFINITY, f32::min);

                // All trimmed values should have magnitude <= min_kept
                for (orig, trim) in values.iter().zip(trimmed.data().iter()) {
                    if *trim == 0.0 && *orig != 0.0 {
                        prop_assert!(
                            orig.abs() <= min_kept + 1e-6,
                            "Trimmed value {} has higher magnitude than kept minimum {}",
                            orig.abs(),
                            min_kept
                        );
                    }
                }
            }
        }

        #[test]
        fn prop_elect_sign_follows_majority(
            pos_count in 1usize..5,
            neg_count in 1usize..5,
            pos_val in 0.1f32..10.0,
            neg_val in -10.0f32..-0.1
        ) {
            let mut arrays: Vec<Array1<f32>> = Vec::new();

            for _ in 0..pos_count {
                arrays.push(Array1::from(vec![pos_val]));
            }
            for _ in 0..neg_count {
                arrays.push(Array1::from(vec![neg_val]));
            }

            let refs: Vec<&Array1<f32>> = arrays.iter().collect();
            let result = elect_and_merge_parameter(&refs);

            if pos_count > neg_count {
                // Positive majority: result should be positive
                prop_assert!(result.data()[0] > 0.0, "Expected positive, got {}", result.data()[0]);
            } else if neg_count > pos_count {
                // Negative majority: result should be negative
                prop_assert!(result.data()[0] < 0.0, "Expected negative, got {}", result.data()[0]);
            }
            // Tie case: could be either, don't assert
        }

        #[test]
        fn prop_ties_config_density_valid(density in 0.0f32..=1.0) {
            let config = TiesConfig::new(density);
            prop_assert!(config.is_ok());
        }

        #[test]
        fn prop_ties_config_density_invalid_negative(density in -10.0f32..-0.01) {
            let config = TiesConfig::new(density);
            prop_assert!(config.is_err());
        }

        #[test]
        fn prop_ties_config_density_invalid_above_one(density in 1.01f32..10.0) {
            let config = TiesConfig::new(density);
            prop_assert!(config.is_err());
        }

        #[test]
        fn prop_trim_idempotent_at_density_one(
            values in proptest::collection::vec(-100.0f32..100.0, 5..20)
        ) {
            let tensor = Tensor::from_vec(values.clone(), false);
            let trimmed = trim_tensor(&tensor, 1.0);

            // At density 1.0, all values should be preserved
            for (orig, trim) in values.iter().zip(trimmed.data().iter()) {
                prop_assert!(
                    (orig - trim).abs() < 1e-6,
                    "Value changed at density 1.0: {} -> {}",
                    orig,
                    trim
                );
            }
        }

        #[test]
        fn prop_elect_preserves_magnitude_order(
            values in proptest::collection::vec(1.0f32..10.0, 3..6)
        ) {
            // All same sign: result should be average
            let arrays: Vec<Array1<f32>> = values.iter().map(|&v| Array1::from(vec![v])).collect();
            let refs: Vec<&Array1<f32>> = arrays.iter().collect();

            let result = elect_and_merge_parameter(&refs);
            let expected_avg: f32 = values.iter().sum::<f32>() / values.len() as f32;

            prop_assert!(
                (result.data()[0] - expected_avg).abs() < 1e-5,
                "Expected average {}, got {}",
                expected_avg,
                result.data()[0]
            );
        }
    }
}
