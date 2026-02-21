pub(crate) use super::*;
pub(crate) use crate::serialization::safetensors::save_safetensors;
pub(crate) use tempfile::tempdir;

/// Create a test model file with given tensors
pub(super) fn create_test_model(
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
    assert!(MergeStrategy::Ties.is_supported());
    assert!(MergeStrategy::Dare.is_supported());
    assert!(MergeStrategy::Slerp.is_supported());
}

#[test]
fn test_merge_options_default() {
    let options = MergeOptions::default();
    assert_eq!(options.strategy, MergeStrategy::Average);
    assert!(options.weights.is_none());
    assert!(options.base_model.is_none());
    assert!((options.drop_rate - 0.9).abs() < 1e-6);
    assert!((options.density - 0.2).abs() < 1e-6);
    assert_eq!(options.seed, 42);
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
fn test_validate_merge_options_ties_requires_base_model() {
    let options = MergeOptions {
        strategy: MergeStrategy::Ties,
        ..Default::default()
    };
    let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("base-model"));
}

#[test]
fn test_validate_merge_options_weighted_missing_weights() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: None,
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
    };
    let result = calculate_merge_weights(2, &options);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("finite positive value"));
}

#[test]
fn test_calculate_merge_weights_negative_sum() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![-1.0, 0.5]),
        ..Default::default()
    };
    let result = calculate_merge_weights(2, &options);
    assert!(result.is_err());
}

// ========================================================================
// BUG-MERGE-006 Falsification Tests: NaN/Inf Weight Handling
// ========================================================================

/// BUG-MERGE-006 FIX: NaN weights must be rejected.
#[test]
fn test_bug_merge_006_nan_weight_rejected() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![f32::NAN, 0.5]),
        ..Default::default()
    };
    let result = calculate_merge_weights(2, &options);
    assert!(
        result.is_err(),
        "FALSIFIED: NaN weight should be rejected but was accepted"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("not finite"),
        "Error message should mention 'not finite': {err}"
    );
}

/// BUG-MERGE-006 FIX: Infinity weights must be rejected.
#[test]
fn test_bug_merge_006_infinity_weight_rejected() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![f32::INFINITY, 1.0]),
        ..Default::default()
    };
    let result = calculate_merge_weights(2, &options);
    assert!(
        result.is_err(),
        "FALSIFIED: Infinity weight should be rejected but was accepted"
    );
}

/// BUG-MERGE-006 FIX: Negative infinity weights must be rejected.
#[test]
fn test_bug_merge_006_neg_infinity_weight_rejected() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![f32::NEG_INFINITY, 1.0]),
        ..Default::default()
    };
    let result = calculate_merge_weights(2, &options);
    assert!(
        result.is_err(),
        "FALSIFIED: Negative infinity weight should be rejected"
    );
}

/// BUG-MERGE-006 FIX: Weights that overflow to infinity when summed must be rejected.
#[test]
fn test_bug_merge_006_overflow_sum_rejected() {
    let options = MergeOptions {
        strategy: MergeStrategy::Weighted,
        weights: Some(vec![f32::MAX, f32::MAX]), // Sum overflows to Inf
        ..Default::default()
    };
    let result = calculate_merge_weights(2, &options);
    assert!(
        result.is_err(),
        "FALSIFIED: Overflow to infinity should be rejected"
    );
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
        ..Default::default()
    };
    let report = apr_merge(&[&model1_path, &model2_path], &output_path, options).unwrap();

    assert_eq!(report.strategy, MergeStrategy::Weighted);
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
        ..Default::default()
    };
    let debug = format!("{:?}", options);
    assert!(debug.contains("MergeOptions"));
    let cloned = options.clone();
    assert_eq!(cloned.strategy, MergeStrategy::Weighted);
}

// ========================================================================
// SLERP Tests
// ========================================================================

#[test]
fn test_slerp_vectors_at_t0() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let result = slerp_vectors(&a, &b, 0.0);
    for (r, expected) in result.iter().zip(a.iter()) {
        assert!((r - expected).abs() < 1e-5, "t=0 should return vector a");
    }
}

#[test]
fn test_slerp_vectors_at_t1() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let result = slerp_vectors(&a, &b, 1.0);
    for (r, expected) in result.iter().zip(b.iter()) {
        assert!((r - expected).abs() < 1e-5, "t=1 should return vector b");
    }
}

#[path = "merge_tests_slerp.rs"]
mod merge_tests_slerp;
