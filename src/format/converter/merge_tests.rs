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

#[test]
fn test_slerp_vectors_at_t05_orthogonal() {
    // Orthogonal unit vectors: slerp at t=0.5 should give normalized midpoint
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let result = slerp_vectors(&a, &b, 0.5);
    // omega = pi/2, slerp(0.5) = sin(pi/4)/sin(pi/2) * (a+b) = 0.707*[1,1]
    let expected = std::f32::consts::FRAC_1_SQRT_2;
    assert!((result[0] - expected).abs() < 1e-5);
    assert!((result[1] - expected).abs() < 1e-5);
}

#[test]
fn test_slerp_vectors_nearly_parallel_falls_back_to_lerp() {
    // Nearly parallel vectors — omega ~0, should use lerp fallback
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 1e-8, 0.0]; // almost the same direction
    let result = slerp_vectors(&a, &b, 0.5);
    // Lerp: (1+1)/2, (0+1e-8)/2, 0
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!(result[1].abs() < 1e-5);
}

#[test]
fn test_slerp_vectors_zero_norm_falls_back_to_lerp() {
    let a = vec![0.0, 0.0];
    let b = vec![2.0, 4.0];
    let result = slerp_vectors(&a, &b, 0.5);
    // lerp: 0.5*[0,0] + 0.5*[2,4] = [1, 2]
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 2.0).abs() < 1e-5);
}

#[test]
fn test_slerp_reject_3_models() {
    let options = MergeOptions {
        strategy: MergeStrategy::Slerp,
        ..Default::default()
    };
    let result = validate_merge_options(
        &["a.safetensors", "b.safetensors", "c.safetensors"],
        &options,
    );
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("exactly 2"));
}

#[test]
fn test_slerp_full_merge() {
    let dir = tempdir().unwrap();
    let model_a_path = dir.path().join("model_a.safetensors");
    let model_b_path = dir.path().join("model_b.safetensors");
    let output_path = dir.path().join("slerp_merged.safetensors");

    let mut tensors_a = BTreeMap::new();
    tensors_a.insert("w".to_string(), (vec![1.0, 0.0], vec![2]));

    let mut tensors_b = BTreeMap::new();
    tensors_b.insert("w".to_string(), (vec![0.0, 1.0], vec![2]));

    create_test_model(&model_a_path, &tensors_a).unwrap();
    create_test_model(&model_b_path, &tensors_b).unwrap();

    let options = MergeOptions {
        strategy: MergeStrategy::Slerp,
        weights: Some(vec![0.5]),
        ..Default::default()
    };
    let report =
        apr_merge(&[&model_a_path, &model_b_path], &output_path, options).unwrap();

    assert_eq!(report.strategy, MergeStrategy::Slerp);
    assert_eq!(report.model_count, 2);
    assert!(output_path.exists());

    // Verify merged output: slerp at t=0.5 of orthogonal unit vectors
    let merged = load_model_tensors(&output_path).unwrap();
    let (data, _) = merged.get("w").unwrap();
    let expected = std::f32::consts::FRAC_1_SQRT_2;
    assert!((data[0] - expected).abs() < 1e-4);
    assert!((data[1] - expected).abs() < 1e-4);
}

#[test]
fn test_slerp_default_weight_is_half() {
    let dir = tempdir().unwrap();
    let model_a_path = dir.path().join("model_a.safetensors");
    let model_b_path = dir.path().join("model_b.safetensors");
    let output_path = dir.path().join("slerp_default.safetensors");

    let mut tensors_a = BTreeMap::new();
    tensors_a.insert("w".to_string(), (vec![1.0, 0.0], vec![2]));

    let mut tensors_b = BTreeMap::new();
    tensors_b.insert("w".to_string(), (vec![0.0, 1.0], vec![2]));

    create_test_model(&model_a_path, &tensors_a).unwrap();
    create_test_model(&model_b_path, &tensors_b).unwrap();

    // No weights specified — should default to t=0.5
    let options = MergeOptions {
        strategy: MergeStrategy::Slerp,
        ..Default::default()
    };
    let report =
        apr_merge(&[&model_a_path, &model_b_path], &output_path, options).unwrap();
    assert_eq!(report.strategy, MergeStrategy::Slerp);
    assert!(output_path.exists());
}

// ========================================================================
// TIES Tests
// ========================================================================

#[test]
fn test_ties_trim_zeros_small_deltas() {
    // density=0.5, max_abs=10, threshold=5. Values with |x|<5 should be zeroed.
    let delta = vec![1.0, -2.0, 10.0, -8.0, 3.0];
    let trimmed = ties_trim(&delta, 0.5);
    assert!((trimmed[0]).abs() < 1e-6); // |1| < 5
    assert!((trimmed[1]).abs() < 1e-6); // |2| < 5
    assert!((trimmed[2] - 10.0).abs() < 1e-6); // |10| >= 5
    assert!((trimmed[3] - (-8.0)).abs() < 1e-6); // |8| >= 5
    assert!((trimmed[4]).abs() < 1e-6); // |3| < 5
}

#[test]
fn test_ties_trim_all_zeros() {
    let delta = vec![0.0, 0.0, 0.0];
    let trimmed = ties_trim(&delta, 0.5);
    for v in &trimmed {
        assert!(v.abs() < 1e-12);
    }
}

#[test]
fn test_ties_elect_and_merge_majority_positive() {
    // 3 deltas: [+5, +3, -2] at position 0 → majority positive (2 vs 1)
    let trimmed = vec![vec![5.0], vec![3.0], vec![-2.0]];
    let result = ties_elect_and_merge(&trimmed, 1);
    // Elected sign: positive. Agreeing values: 5, 3. Sum=8, count=2.
    // Average = 8 / 3 (num_models) = 2.667
    assert!((result[0] - 8.0 / 3.0).abs() < 1e-5);
}

#[test]
fn test_ties_elect_and_merge_majority_negative() {
    // 3 deltas: [-5, -3, +2] at position 0 → majority negative (2 vs 1)
    let trimmed = vec![vec![-5.0], vec![-3.0], vec![2.0]];
    let result = ties_elect_and_merge(&trimmed, 1);
    // Elected sign: negative. Agreeing values: -5, -3. Sum=-8.
    // Average = -8 / 3 = -2.667
    assert!((result[0] - (-8.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_ties_full_merge() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let task1_path = dir.path().join("task1.safetensors");
    let task2_path = dir.path().join("task2.safetensors");
    let output_path = dir.path().join("ties_merged.safetensors");

    // Base model
    let mut base = BTreeMap::new();
    base.insert("w".to_string(), (vec![0.0, 0.0, 0.0, 0.0], vec![4]));

    // Task models with deltas from base
    let mut task1 = BTreeMap::new();
    task1.insert("w".to_string(), (vec![10.0, -5.0, 1.0, -1.0], vec![4]));

    let mut task2 = BTreeMap::new();
    task2.insert("w".to_string(), (vec![8.0, -3.0, -1.0, 1.0], vec![4]));

    create_test_model(&base_path, &base).unwrap();
    create_test_model(&task1_path, &task1).unwrap();
    create_test_model(&task2_path, &task2).unwrap();

    let options = MergeOptions {
        strategy: MergeStrategy::Ties,
        base_model: Some(base_path),
        density: 0.2,
        ..Default::default()
    };
    let report =
        apr_merge(&[&task1_path, &task2_path], &output_path, options).unwrap();

    assert_eq!(report.strategy, MergeStrategy::Ties);
    assert_eq!(report.model_count, 2);
    assert!(output_path.exists());

    // Verify output has same tensor structure
    let merged = load_model_tensors(&output_path).unwrap();
    let (data, shape) = merged.get("w").unwrap();
    assert_eq!(shape, &vec![4]);
    assert_eq!(data.len(), 4);
}

#[test]
fn test_ties_validate_requires_base() {
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
fn test_ties_validate_density_range() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let mut base = BTreeMap::new();
    base.insert("w".to_string(), (vec![1.0], vec![1]));
    create_test_model(&base_path, &base).unwrap();

    // density = 0.0 is out of (0, 1) range
    let options = MergeOptions {
        strategy: MergeStrategy::Ties,
        base_model: Some(base_path.clone()),
        density: 0.0,
        ..Default::default()
    };
    let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("density"));

    // density = 1.0 is out of (0, 1) range
    let options = MergeOptions {
        strategy: MergeStrategy::Ties,
        base_model: Some(base_path),
        density: 1.0,
        ..Default::default()
    };
    let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
    assert!(result.is_err());
}

// ========================================================================
// DARE Tests
// ========================================================================

#[test]
fn test_dare_deterministic_with_seed() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let task_path = dir.path().join("task.safetensors");
    let output1_path = dir.path().join("dare1.safetensors");
    let output2_path = dir.path().join("dare2.safetensors");

    let mut base = BTreeMap::new();
    base.insert("w".to_string(), (vec![0.0; 100], vec![100]));

    let mut task = BTreeMap::new();
    task.insert("w".to_string(), (vec![1.0; 100], vec![100]));

    // Need 2 task models for merge (at least 2 inputs required)
    let task2_path = dir.path().join("task2.safetensors");
    let mut task2 = BTreeMap::new();
    task2.insert("w".to_string(), (vec![2.0; 100], vec![100]));

    create_test_model(&base_path, &base).unwrap();
    create_test_model(&task_path, &task).unwrap();
    create_test_model(&task2_path, &task2).unwrap();

    // Run twice with same seed
    for output_path in [&output1_path, &output2_path] {
        let options = MergeOptions {
            strategy: MergeStrategy::Dare,
            base_model: Some(base_path.clone()),
            drop_rate: 0.5,
            seed: 42,
            ..Default::default()
        };
        apr_merge(
            &[&task_path, &task2_path],
            output_path,
            options,
        )
        .unwrap();
    }

    let merged1 = load_model_tensors(&output1_path).unwrap();
    let merged2 = load_model_tensors(&output2_path).unwrap();
    let (data1, _) = merged1.get("w").unwrap();
    let (data2, _) = merged2.get("w").unwrap();

    // Same seed → same output
    for (a, b) in data1.iter().zip(data2.iter()) {
        assert!((a - b).abs() < 1e-6, "Deterministic DARE should produce identical results");
    }
}

#[test]
fn test_dare_rescale_factor() {
    // With drop_rate = 0.5, kept elements should be scaled by 1/(1-0.5) = 2.0
    // This is tested via the internal dare_merge function
    let mut base = BTreeMap::new();
    base.insert("w".to_string(), (vec![0.0, 0.0, 0.0, 0.0], vec![4]));

    let mut task = BTreeMap::new();
    task.insert("w".to_string(), (vec![1.0, 1.0, 1.0, 1.0], vec![4]));

    let task_models = vec![task];
    let result = dare_merge(&base, &task_models, 0.0001, 42, None);

    // With drop_rate ~0 almost all elements are kept and scaled by ~1.0
    let (data, _) = result.get("w").unwrap();
    // All deltas should be close to 1.0 (nearly no dropping)
    let sum: f32 = data.iter().sum();
    // With 4 elements and almost no dropping, sum should be ~4.0
    assert!(sum > 3.5, "Very low drop_rate should keep most elements");
}

#[test]
fn test_dare_validate_requires_base() {
    let options = MergeOptions {
        strategy: MergeStrategy::Dare,
        ..Default::default()
    };
    let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("base-model"));
}

#[test]
fn test_dare_validate_drop_rate_range() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let mut base = BTreeMap::new();
    base.insert("w".to_string(), (vec![1.0], vec![1]));
    create_test_model(&base_path, &base).unwrap();

    // drop_rate = 0.0 is out of (0, 1) range
    let options = MergeOptions {
        strategy: MergeStrategy::Dare,
        base_model: Some(base_path.clone()),
        drop_rate: 0.0,
        ..Default::default()
    };
    let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("drop_rate"));

    // drop_rate = 1.0 is out of (0, 1) range
    let options = MergeOptions {
        strategy: MergeStrategy::Dare,
        base_model: Some(base_path),
        drop_rate: 1.0,
        ..Default::default()
    };
    let result = validate_merge_options(&["a.safetensors", "b.safetensors"], &options);
    assert!(result.is_err());
}

#[test]
fn test_dare_full_merge() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("base.safetensors");
    let task1_path = dir.path().join("task1.safetensors");
    let task2_path = dir.path().join("task2.safetensors");
    let output_path = dir.path().join("dare_merged.safetensors");

    let mut base = BTreeMap::new();
    base.insert("w".to_string(), (vec![0.0, 0.0], vec![2]));

    let mut task1 = BTreeMap::new();
    task1.insert("w".to_string(), (vec![10.0, 5.0], vec![2]));

    let mut task2 = BTreeMap::new();
    task2.insert("w".to_string(), (vec![8.0, 3.0], vec![2]));

    create_test_model(&base_path, &base).unwrap();
    create_test_model(&task1_path, &task1).unwrap();
    create_test_model(&task2_path, &task2).unwrap();

    let options = MergeOptions {
        strategy: MergeStrategy::Dare,
        base_model: Some(base_path),
        drop_rate: 0.5,
        seed: 42,
        ..Default::default()
    };
    let report =
        apr_merge(&[&task1_path, &task2_path], &output_path, options).unwrap();

    assert_eq!(report.strategy, MergeStrategy::Dare);
    assert_eq!(report.model_count, 2);
    assert!(output_path.exists());
}

// ========================================================================
// Lerp helper test
// ========================================================================

#[test]
fn test_lerp_vectors() {
    let a = vec![0.0, 10.0];
    let b = vec![10.0, 0.0];
    let result = lerp_vectors(&a, &b, 0.3);
    assert!((result[0] - 3.0).abs() < 1e-5); // 0*(1-0.3) + 10*0.3
    assert!((result[1] - 7.0).abs() < 1e-5); // 10*(1-0.3) + 0*0.3
}

#[test]
fn test_vector_norm() {
    let v = vec![3.0, 4.0];
    let norm = vector_norm(&v);
    assert!((norm - 5.0).abs() < 1e-10);
}
