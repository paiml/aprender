//! Property tests for model merging algorithms

use super::*;
use crate::autograd::Tensor;
use std::collections::HashMap;

/// Helper to create a simple model with one parameter
fn create_model(name: &str, values: Vec<f32>) -> Model {
    let mut model = HashMap::new();
    model.insert(name.to_string(), Tensor::from_vec(values, false));
    model
}

/// Helper to create multiple models
fn create_models(values_per_model: Vec<Vec<f32>>) -> Vec<Model> {
    values_per_model.into_iter().map(|values| create_model("w", values)).collect()
}

#[cfg(test)]
mod ties_properties {
    use super::*;
    use crate::merge::{ties_merge, TiesConfig};

    #[test]
    fn ties_is_permutation_invariant() {
        // Property: TIES merge result should be independent of model ordering
        let base = create_model("w", vec![0.0, 0.0, 0.0, 0.0]);
        let models = create_models(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, 3.0, 4.0],
            vec![1.0, -2.0, -3.0, 4.0],
        ]);

        let config = TiesConfig::new(0.5).expect("config should be valid");

        let result1 = ties_merge(&models, &base, &config).expect("config should be valid");

        // Permute models
        let permuted = vec![models[2].clone(), models[0].clone(), models[1].clone()];
        let result2 = ties_merge(&permuted, &base, &config).expect("config should be valid");

        // Results should be identical
        let r1_data = result1["w"].data();
        let r2_data = result2["w"].data();
        for (a, b) in r1_data.iter().zip(r2_data.iter()) {
            assert!((a - b).abs() < 1e-5, "TIES should be permutation-invariant: {a} != {b}");
        }
    }

    #[test]
    fn ties_with_identical_models_has_same_deltas() {
        // Property: Merging identical models should preserve the delta direction
        // Note: Due to trimming, exact values may differ, but non-zero elements should align
        let base = create_model("w", vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let model = create_model("w", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let models = vec![model.clone(), model.clone()];

        // Use high density to preserve most values
        let config = TiesConfig::new(0.8).expect("config should be valid");
        let result = ties_merge(&models, &base, &config).expect("config should be valid");

        // Result should be close to the model (after trimming)
        // Since both models are identical, all votes agree, so kept values equal original
        let expected = model["w"].data();
        let actual = result["w"].data();

        // Check that non-zero elements have correct sign and magnitude is reasonable
        for (a, e) in actual.iter().zip(expected.iter()) {
            if a.abs() > 1e-6 {
                // Non-zero result should match expected sign
                assert!(a * e > 0.0, "Sign mismatch: {a} vs {e}");
            }
        }
    }

    #[test]
    fn ties_preserves_zero_deltas() {
        // Property: If all models equal base, output equals base
        let base = create_model("w", vec![5.0, 10.0]);
        let models = vec![base.clone(), base.clone()];

        let config = TiesConfig::default();
        let result = ties_merge(&models, &base, &config).expect("config should be valid");

        let expected = base["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod dare_properties {
    use super::*;
    use crate::merge::{dare_merge, DareConfig};

    #[test]
    fn dare_is_deterministic_with_seed() {
        // Property: Same seed produces same results
        let base = create_model("w", vec![0.0, 0.0]);
        let models = create_models(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let config = DareConfig::new(0.5).expect("config should be valid").with_seed(42);

        let result1 = dare_merge(&models, &base, &config).expect("config should be valid");
        let result2 = dare_merge(&models, &base, &config).expect("config should be valid");

        let r1_data = result1["w"].data();
        let r2_data = result2["w"].data();
        for (a, b) in r1_data.iter().zip(r2_data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn dare_with_zero_drop_is_average() {
        // Property: drop_prob=0 should equal simple average
        let base = create_model("w", vec![0.0, 0.0]);
        let models = create_models(vec![vec![2.0, 4.0], vec![4.0, 6.0]]);

        let config = DareConfig::new(0.0).expect("config should be valid");
        let result = dare_merge(&models, &base, &config).expect("config should be valid");

        // Expected: (2+4)/2 = 3.0, (4+6)/2 = 5.0
        assert!((result["w"].data()[0] - 3.0).abs() < 1e-5);
        assert!((result["w"].data()[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn dare_preserves_zero_deltas() {
        // Property: If all models equal base, output equals base
        let base = create_model("w", vec![7.0, 14.0]);
        let models = vec![base.clone(), base.clone()];

        let config = DareConfig::default();
        let result = dare_merge(&models, &base, &config).expect("config should be valid");

        let expected = base["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod slerp_properties {
    use super::*;
    use crate::merge::{slerp_merge, SlerpConfig};

    #[test]
    fn slerp_at_t0_returns_model1() {
        // Property: t=0 should return first model exactly
        let model1 = create_model("w", vec![1.0, 2.0, 3.0]);
        let model2 = create_model("w", vec![4.0, 5.0, 6.0]);

        let config = SlerpConfig::new(0.0).expect("slerp config creation should succeed");
        let result = slerp_merge(&model1, &model2, &config).expect("config should be valid");

        let expected = model1["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn slerp_at_t1_returns_model2() {
        // Property: t=1 should return second model exactly
        let model1 = create_model("w", vec![1.0, 2.0, 3.0]);
        let model2 = create_model("w", vec![4.0, 5.0, 6.0]);

        let config = SlerpConfig::new(1.0).expect("slerp config creation should succeed");
        let result = slerp_merge(&model1, &model2, &config).expect("config should be valid");

        let expected = model2["w"].data();
        let actual = result["w"].data();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    #[test]
    fn slerp_is_continuous() {
        // Property: Small changes in t produce small changes in output
        let model1 = create_model("w", vec![1.0, 0.0]);
        let model2 = create_model("w", vec![0.0, 1.0]);

        let config1 = SlerpConfig::new(0.5).expect("slerp config creation should succeed");
        let config2 = SlerpConfig::new(0.51).expect("slerp config creation should succeed");

        let result1 = slerp_merge(&model1, &model2, &config1).expect("config should be valid");
        let result2 = slerp_merge(&model1, &model2, &config2).expect("config should be valid");

        // Results should be very close for nearby t values
        let r1_data = result1["w"].data();
        let r2_data = result2["w"].data();
        for (a, b) in r1_data.iter().zip(r2_data.iter()) {
            assert!((a - b).abs() < 0.1); // Generous tolerance for continuity
        }
    }

    #[test]
    fn slerp_symmetric_models_at_midpoint() {
        // Property: For symmetric models, t=0.5 should be exactly midway
        let model1 = create_model("w", vec![1.0]);
        let model2 = create_model("w", vec![-1.0]);

        let config = SlerpConfig::new(0.5).expect("slerp config creation should succeed");
        let result = slerp_merge(&model1, &model2, &config).expect("config should be valid");

        // For anti-parallel vectors, SLERP at t=0.5 should be near zero
        assert!(result["w"].data()[0].abs() < 0.1);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::merge::{dare_merge, slerp_merge, ties_merge, DareConfig, SlerpConfig, TiesConfig};

    #[test]
    fn test_three_way_merge_comparison() {
        // Compare all three methods on same inputs
        let base = create_model("w", vec![0.0, 0.0, 0.0]);
        let models = create_models(vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);

        // TIES
        let ties_config = TiesConfig::default();
        let ties_result = ties_merge(&models, &base, &ties_config).expect("config should be valid");

        // DARE with zero drop (equivalent to average)
        let dare_config = DareConfig::new(0.0).expect("config should be valid");
        let dare_result = dare_merge(&models, &base, &dare_config).expect("config should be valid");

        // SLERP at midpoint
        let slerp_config = SlerpConfig::new(0.5).expect("slerp config creation should succeed");
        let slerp_result =
            slerp_merge(&models[0], &models[1], &slerp_config).expect("config should be valid");

        // All should produce reasonable results (no NaN/Inf)
        for val in ties_result["w"].data() {
            assert!(val.is_finite());
        }
        for val in dare_result["w"].data() {
            assert!(val.is_finite());
        }
        for val in slerp_result["w"].data() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_incompatible_shapes_rejected() {
        let base = create_model("w", vec![0.0, 0.0]);
        let model1 = create_model("w", vec![1.0, 2.0]);
        let model2 = create_model("w", vec![3.0, 4.0, 5.0]); // Wrong shape!

        let models = vec![model1, model2];

        // TIES should reject
        let ties_result = ties_merge(&models, &base, &TiesConfig::default());
        assert!(ties_result.is_err());

        // DARE should reject
        let dare_result = dare_merge(&models, &base, &DareConfig::default());
        assert!(dare_result.is_err());
    }
}

// =============================================================================
// Additional coverage tests for mod.rs functions
// =============================================================================

#[cfg(test)]
mod mod_coverage_tests {
    use super::*;
    use crate::merge::{compute_deltas, merge_with_base, validate_models, MergeError};

    #[test]
    fn test_compute_deltas_basic() {
        let base = create_model("w", vec![0.0, 0.0, 0.0]);
        let model1 = create_model("w", vec![1.0, 2.0, 3.0]);
        let model2 = create_model("w", vec![4.0, 5.0, 6.0]);

        let models = vec![model1, model2];
        let deltas = compute_deltas(&models, &base).expect("operation should succeed");

        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0]["w"].data()[0], 1.0);
        assert_eq!(deltas[1]["w"].data()[0], 4.0);
    }

    #[test]
    fn test_compute_deltas_missing_param() {
        let base = create_model("w", vec![0.0]);
        let mut model = HashMap::new();
        model.insert("other".to_string(), Tensor::from_vec(vec![1.0], false));

        let result = compute_deltas(&[model], &base);
        assert!(matches!(result, Err(MergeError::IncompatibleArchitectures(_))));
    }

    #[test]
    fn test_compute_deltas_shape_mismatch() {
        let base = create_model("w", vec![0.0, 0.0]);
        let model = create_model("w", vec![1.0, 2.0, 3.0]); // Wrong shape

        let result = compute_deltas(&[model], &base);
        assert!(matches!(result, Err(MergeError::ShapeMismatch(_))));
    }

    #[test]
    fn test_merge_with_base() {
        let base = create_model("w", vec![10.0, 20.0]);
        let delta = create_model("w", vec![1.0, 2.0]);

        let merged = merge_with_base(&base, delta);
        assert_eq!(merged["w"].data()[0], 11.0);
        assert_eq!(merged["w"].data()[1], 22.0);
    }

    #[test]
    fn test_merge_with_base_missing_delta() {
        let mut base = HashMap::new();
        base.insert("w".to_string(), Tensor::from_vec(vec![1.0], false));
        base.insert("b".to_string(), Tensor::from_vec(vec![2.0], false));

        // Delta only has 'w'
        let delta = create_model("w", vec![0.5]);

        let merged = merge_with_base(&base, delta);
        // 'w' should be merged, 'b' should be unchanged
        assert_eq!(merged["w"].data()[0], 1.5);
        assert_eq!(merged["b"].data()[0], 2.0);
    }

    #[test]
    fn test_validate_models_empty() {
        let models: Vec<Model> = vec![];
        let result = validate_models(&models);
        assert!(matches!(result, Err(MergeError::InsufficientModels { min: 1, got: 0 })));
    }

    #[test]
    fn test_validate_models_missing_param() {
        let model1 = create_model("w", vec![1.0]);
        let model2 = create_model("b", vec![2.0]); // Different param name

        let models = vec![model1, model2];
        let result = validate_models(&models);
        assert!(matches!(result, Err(MergeError::IncompatibleArchitectures(_))));
    }

    #[test]
    fn test_validate_models_shape_mismatch() {
        let model1 = create_model("w", vec![1.0, 2.0]);
        let model2 = create_model("w", vec![3.0, 4.0, 5.0]);

        let models = vec![model1, model2];
        let result = validate_models(&models);
        assert!(matches!(result, Err(MergeError::ShapeMismatch(_))));
    }

    #[test]
    fn test_validate_models_valid() {
        let model1 = create_model("w", vec![1.0, 2.0]);
        let model2 = create_model("w", vec![3.0, 4.0]);

        let models = vec![model1, model2];
        assert!(validate_models(&models).is_ok());
    }

    #[test]
    fn test_merge_error_display() {
        let err1 = MergeError::IncompatibleArchitectures("test".to_string());
        assert!(err1.to_string().contains("incompatible"));

        let err2 = MergeError::ShapeMismatch("param".to_string());
        assert!(err2.to_string().contains("shape"));

        let err3 = MergeError::InvalidConfig("bad config".to_string());
        assert!(err3.to_string().contains("Invalid"));

        let err4 = MergeError::InsufficientModels { min: 2, got: 1 };
        assert!(err4.to_string().contains("Insufficient"));
    }
}
