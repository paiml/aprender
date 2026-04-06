//! DARE (Drop And REscale) merge algorithm
//!
//! DARE merges models by randomly dropping delta parameters with probability p,
//! then rescaling the remaining values to maintain expected magnitude.

use super::{compute_deltas, merge_with_base, validate_models, MergeError, Model};
use crate::autograd::Tensor;
use ndarray::Array1;
use rand::Rng;
use std::collections::HashMap;

/// Configuration for DARE merge
#[derive(Clone, Debug)]
pub struct DareConfig {
    /// Drop probability: probability of zeroing out each delta parameter
    /// Higher p = more aggressive dropping = sparser merged model
    /// Typical values: 0.3 to 0.7
    pub drop_prob: f32,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for DareConfig {
    fn default() -> Self {
        Self { drop_prob: 0.5, seed: None }
    }
}

impl DareConfig {
    pub fn new(drop_prob: f32) -> Result<Self, MergeError> {
        if !(0.0..=1.0).contains(&drop_prob) {
            return Err(MergeError::InvalidConfig(format!(
                "Drop probability must be in [0.0, 1.0], got {drop_prob}"
            )));
        }
        Ok(Self { drop_prob, seed: None })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// DARE merge: drop and rescale delta parameters
///
/// # Arguments
/// * `models` - Fine-tuned models to merge
/// * `base` - Base model (pre-fine-tuning checkpoint)
/// * `config` - DARE configuration (drop probability)
///
/// # Returns
/// Merged model with sparsified deltas
///
/// # Algorithm
/// 1. Compute deltas: Δᵢ = model_i - base
/// 2. Drop: Apply Bernoulli(1-p) mask to each delta
/// 3. Rescale: Multiply kept values by 1/(1-p) to maintain expected value
/// 4. Average: Take mean across all masked deltas
/// 5. Add back to base: merged = base + averaged_delta
pub fn dare_merge(
    models: &[Model],
    base: &Model,
    config: &DareConfig,
) -> Result<Model, MergeError> {
    if models.is_empty() {
        return Err(MergeError::InsufficientModels { min: 1, got: 0 });
    }

    validate_models(models)?;

    // Step 1: Compute deltas
    let deltas = compute_deltas(models, base)?;

    // Step 2 & 3: Drop and rescale
    let masked_deltas = if let Some(_seed) = config.seed {
        // For deterministic merging (testing), use seeded RNG
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(_seed);
        drop_and_rescale_deltas(&deltas, config.drop_prob, &mut rng)
    } else {
        // For normal use, use thread-local RNG
        let mut rng = rand::rng();
        drop_and_rescale_deltas(&deltas, config.drop_prob, &mut rng)
    };

    // Step 4: Average masked deltas
    let averaged_delta = average_deltas(&masked_deltas);

    // Step 5: Add back to base
    Ok(merge_with_base(base, averaged_delta))
}

/// Drop parameters with probability p and rescale by 1/(1-p)
fn drop_and_rescale_deltas<R: Rng>(deltas: &[Model], drop_prob: f32, rng: &mut R) -> Vec<Model> {
    let keep_prob = 1.0 - drop_prob;
    let scale = if keep_prob > 0.0 { 1.0 / keep_prob } else { 1.0 };

    deltas
        .iter()
        .map(|delta| {
            let mut masked = HashMap::new();
            for (name, tensor) in delta {
                masked.insert(name.clone(), drop_and_rescale_tensor(tensor, drop_prob, scale, rng));
            }
            masked
        })
        .collect()
}

/// Apply Bernoulli dropout mask to a single tensor
fn drop_and_rescale_tensor<R: Rng>(
    tensor: &Tensor,
    drop_prob: f32,
    scale: f32,
    rng: &mut R,
) -> Tensor {
    let data = tensor.data();
    let masked_data: Array1<f32> = data
        .iter()
        .map(|&val| {
            if rng.random::<f32>() < drop_prob {
                0.0 // Drop
            } else {
                val * scale // Keep and rescale
            }
        })
        .collect();

    Tensor::new(masked_data, false)
}

/// Average multiple delta models
fn average_deltas(deltas: &[Model]) -> Model {
    if deltas.is_empty() {
        return HashMap::new();
    }

    let n = deltas.len() as f32;
    let reference = &deltas[0];
    let mut averaged = HashMap::new();

    for name in reference.keys() {
        let sum_data: Array1<f32> = deltas
            .iter()
            .map(|delta| delta[name].data())
            .fold(Array1::zeros(reference[name].len()), |acc, data| &acc + data);

        let avg_data = sum_data / n;
        averaged.insert(name.clone(), Tensor::new(avg_data, false));
    }

    averaged
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    #[test]
    fn test_drop_and_rescale_tensor_deterministic() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let masked = drop_and_rescale_tensor(&tensor, 0.5, 2.0, &mut rng);

        // With drop_prob=0.5, scale=2.0:
        // - Dropped values -> 0.0
        // - Kept values -> original * 2.0
        let data = masked.data();
        for &val in data {
            assert!(val == 0.0 || val % 2.0 == 0.0);
        }
    }

    #[test]
    fn test_average_deltas() {
        let mut delta1 = HashMap::new();
        delta1.insert("w".to_string(), Tensor::from_vec(vec![1.0, 2.0], false));

        let mut delta2 = HashMap::new();
        delta2.insert("w".to_string(), Tensor::from_vec(vec![3.0, 4.0], false));

        let averaged = average_deltas(&[delta1, delta2]);

        let expected = [2.0, 3.0]; // (1+3)/2, (2+4)/2
        let actual = averaged["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dare_config_validation() {
        assert!(DareConfig::new(0.5).is_ok());
        assert!(DareConfig::new(0.0).is_ok());
        assert!(DareConfig::new(1.0).is_ok());
        assert!(DareConfig::new(-0.1).is_err());
        assert!(DareConfig::new(1.1).is_err());
    }

    #[test]
    fn test_dare_merge_with_seed_is_deterministic() {
        let mut base = HashMap::new();
        base.insert("w".to_string(), Tensor::from_vec(vec![0.0, 0.0], false));

        let mut model1 = base.clone();
        model1.insert("w".to_string(), Tensor::from_vec(vec![1.0, 2.0], false));

        let mut model2 = base.clone();
        model2.insert("w".to_string(), Tensor::from_vec(vec![3.0, 4.0], false));

        let models = vec![model1, model2];
        let config = DareConfig::new(0.5).expect("config should be valid").with_seed(42);

        let result1 = dare_merge(&models, &base, &config).expect("config should be valid");
        let result2 = dare_merge(&models, &base, &config).expect("config should be valid");

        // Same seed should produce same results
        let r1_data = result1["w"].data();
        let r2_data = result2["w"].data();
        for (a, b) in r1_data.iter().zip(r2_data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_drop_prob_zero_keeps_all() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // drop_prob=0 means keep all, scale=1
        let masked = drop_and_rescale_tensor(&tensor, 0.0, 1.0, &mut rng);

        let data = masked.data();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 5.0);
    }

    #[test]
    fn test_drop_prob_one_drops_all() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // drop_prob=1.0 means drop all
        let masked = drop_and_rescale_tensor(&tensor, 1.0, 1.0, &mut rng);

        let data = masked.data();
        for &val in data {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_dare_merge_empty_models() {
        let mut base = HashMap::new();
        base.insert("w".to_string(), Tensor::from_vec(vec![0.0], false));

        let models: Vec<Model> = vec![];
        let config = DareConfig::default();

        let result = dare_merge(&models, &base, &config);
        assert!(matches!(result, Err(MergeError::InsufficientModels { min: 1, got: 0 })));
    }

    #[test]
    fn test_dare_merge_single_model() {
        let mut base = HashMap::new();
        base.insert("w".to_string(), Tensor::from_vec(vec![0.0, 0.0], false));

        let mut model1 = HashMap::new();
        model1.insert("w".to_string(), Tensor::from_vec(vec![1.0, 2.0], false));

        let models = vec![model1];
        let config = DareConfig::new(0.0).expect("config should be valid").with_seed(42); // Keep all

        let result = dare_merge(&models, &base, &config).expect("config should be valid");

        // With drop_prob=0, should get model1 back
        let w = result.get("w").expect("key should exist");
        assert!((w.data()[0] - 1.0).abs() < 1e-6);
        assert!((w.data()[1] - 2.0).abs() < 1e-6);
    }

    // Property tests

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_dare_config_valid_range(drop_prob in 0.0f32..=1.0) {
            let config = DareConfig::new(drop_prob);
            prop_assert!(config.is_ok());
        }

        #[test]
        fn prop_dare_config_invalid_negative(drop_prob in -10.0f32..-0.01) {
            let config = DareConfig::new(drop_prob);
            prop_assert!(config.is_err());
        }

        #[test]
        fn prop_dare_config_invalid_above_one(drop_prob in 1.01f32..10.0) {
            let config = DareConfig::new(drop_prob);
            prop_assert!(config.is_err());
        }

        #[test]
        fn prop_drop_and_rescale_output_values(
            values in proptest::collection::vec(1.0f32..10.0, 10..50),
            drop_prob in 0.0f32..1.0,
            seed in 0u64..1000
        ) {
            let tensor = Tensor::from_vec(values.clone(), false);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let keep_prob = 1.0 - drop_prob;
            let scale = if keep_prob > 0.0 { 1.0 / keep_prob } else { 1.0 };

            let masked = drop_and_rescale_tensor(&tensor, drop_prob, scale, &mut rng);

            // Each value should be either 0 (dropped) or original * scale (kept)
            for (orig, result) in values.iter().zip(masked.data().iter()) {
                if *result != 0.0 {
                    let expected = orig * scale;
                    prop_assert!(
                        (result - expected).abs() < 1e-4,
                        "Expected {} * {} = {}, got {}",
                        orig,
                        scale,
                        expected,
                        result
                    );
                }
            }
        }

        #[test]
        fn prop_drop_prob_zero_preserves_values(
            values in proptest::collection::vec(-100.0f32..100.0, 5..20),
            seed in 0u64..1000
        ) {
            let tensor = Tensor::from_vec(values.clone(), false);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

            // drop_prob=0 with scale=1 should preserve all values
            let masked = drop_and_rescale_tensor(&tensor, 0.0, 1.0, &mut rng);

            for (orig, result) in values.iter().zip(masked.data().iter()) {
                prop_assert!(
                    (orig - result).abs() < 1e-6,
                    "Value not preserved: {} -> {}",
                    orig,
                    result
                );
            }
        }

        #[test]
        fn prop_drop_prob_one_zeros_all(
            values in proptest::collection::vec(-100.0f32..100.0, 5..20),
            seed in 0u64..1000
        ) {
            let tensor = Tensor::from_vec(values, false);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

            // drop_prob=1.0 should zero all values
            let masked = drop_and_rescale_tensor(&tensor, 1.0, 1.0, &mut rng);

            for &val in masked.data() {
                prop_assert_eq!(val, 0.0);
            }
        }

        #[test]
        fn prop_average_deltas_is_mean(
            v1 in proptest::collection::vec(-100.0f32..100.0, 5..10),
            v2 in proptest::collection::vec(-100.0f32..100.0, 5..10)
        ) {
            // Ensure same length
            let len = v1.len().min(v2.len());
            let v1: Vec<f32> = v1.into_iter().take(len).collect();
            let v2: Vec<f32> = v2.into_iter().take(len).collect();

            let mut delta1 = HashMap::new();
            delta1.insert("w".to_string(), Tensor::from_vec(v1.clone(), false));

            let mut delta2 = HashMap::new();
            delta2.insert("w".to_string(), Tensor::from_vec(v2.clone(), false));

            let averaged = average_deltas(&[delta1, delta2]);
            let avg_data = averaged["w"].data();

            for i in 0..len {
                let expected = f32::midpoint(v1[i], v2[i]);
                prop_assert!(
                    (avg_data[i] - expected).abs() < 1e-5,
                    "Average mismatch at {}: expected {}, got {}",
                    i,
                    expected,
                    avg_data[i]
                );
            }
        }

        #[test]
        fn prop_dare_deterministic_with_same_seed(
            delta_values in proptest::collection::vec(-10.0f32..10.0, 5..15),
            seed in 0u64..1000,
            drop_prob in 0.1f32..0.9
        ) {
            let mut base = HashMap::new();
            base.insert("w".to_string(), Tensor::from_vec(vec![0.0; delta_values.len()], false));

            let mut model1 = HashMap::new();
            model1.insert("w".to_string(), Tensor::from_vec(delta_values, false));

            let models = vec![model1];
            let config = DareConfig::new(drop_prob).expect("config should be valid").with_seed(seed);

            let result1 = dare_merge(&models, &base, &config).expect("config should be valid");
            let result2 = dare_merge(&models, &base, &config).expect("config should be valid");

            // Same seed should produce identical results
            for (a, b) in result1["w"].data().iter().zip(result2["w"].data().iter()) {
                prop_assert!(
                    (a - b).abs() < 1e-6,
                    "Non-deterministic result: {} vs {}",
                    a,
                    b
                );
            }
        }
    }
}
