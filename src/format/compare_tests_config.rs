use super::*;

#[test]
fn test_compare_config_with_prefixes() {
    let config = CompareConfig {
        source_prefix: Some("model.".to_string()),
        target_prefix: Some("encoder.".to_string()),
        ..CompareConfig::default()
    };
    assert_eq!(config.source_prefix.as_deref(), Some("model."));
    assert_eq!(config.target_prefix.as_deref(), Some("encoder."));
}

// --- TensorComparison::is_match_with_tolerance edge cases ---

#[test]
fn test_is_match_with_tolerance_l2_exceeds_max_within() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: Some(1.0),   // exceeds tolerance
        max_diff: Some(1e-8), // within tolerance
        mean_diff: Some(0.0),
    };
    // l2 exceeds -> not a match
    assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
}

#[test]
fn test_is_match_with_tolerance_max_exceeds_l2_within() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: Some(1e-8), // within tolerance
        max_diff: Some(1.0), // exceeds tolerance
        mean_diff: Some(0.0),
    };
    // max exceeds -> not a match
    assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
}

#[test]
fn test_is_match_with_tolerance_both_exceed() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: Some(1.0),
        max_diff: Some(1.0),
        mean_diff: Some(0.5),
    };
    assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
}

#[test]
fn test_is_match_with_tolerance_exact_boundary() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: Some(1e-5),  // exactly at tolerance
        max_diff: Some(1e-5), // exactly at tolerance
        mean_diff: Some(0.0),
    };
    // <= so exact boundary should match
    assert!(comparison.is_match_with_tolerance(1e-5, 1e-5));
}

#[test]
fn test_is_match_with_tolerance_none_l2() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: None,
        max_diff: Some(0.0),
        mean_diff: None,
    };
    // l2_diff is None -> is_some_and returns false -> not a match
    assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
}

#[test]
fn test_is_match_with_tolerance_none_max() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: Some(0.0),
        max_diff: None,
        mean_diff: None,
    };
    // max_diff is None -> not a match
    assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
}

#[test]
fn test_is_match_with_tolerance_shape_mismatch_but_diffs_ok() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: false,
        source_shape: vec![4],
        target_shape: vec![2, 2],
        l2_diff: Some(0.0),
        max_diff: Some(0.0),
        mean_diff: Some(0.0),
    };
    // shape_match false -> not a match regardless of diffs
    assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
}

#[test]
fn test_is_match_with_zero_tolerance() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: Some(0.0),
        max_diff: Some(0.0),
        mean_diff: Some(0.0),
    };
    assert!(comparison.is_match_with_tolerance(0.0, 0.0));
}

// --- TensorComparison::element_count edge cases ---

#[test]
fn test_element_count_empty_shape() {
    let comparison = TensorComparison {
        name: "scalar".to_string(),
        shape_match: true,
        source_shape: vec![],
        target_shape: vec![],
        l2_diff: Some(0.0),
        max_diff: Some(0.0),
        mean_diff: Some(0.0),
    };
    // Product of empty iterator is 1 (identity element)
    assert_eq!(comparison.element_count(), 1);
}

#[test]
fn test_element_count_single_dim() {
    let comparison = TensorComparison {
        name: "vector".to_string(),
        shape_match: true,
        source_shape: vec![128],
        target_shape: vec![128],
        l2_diff: None,
        max_diff: None,
        mean_diff: None,
    };
    assert_eq!(comparison.element_count(), 128);
}

#[test]
fn test_element_count_high_rank() {
    let comparison = TensorComparison {
        name: "4d".to_string(),
        shape_match: true,
        source_shape: vec![2, 3, 4, 5],
        target_shape: vec![2, 3, 4, 5],
        l2_diff: None,
        max_diff: None,
        mean_diff: None,
    };
    assert_eq!(comparison.element_count(), 120);
}

// --- compute_diff_stats edge cases ---

#[test]
fn test_compare_tensors_empty_data() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let data: Vec<f32> = vec![];
    let shape: Vec<usize> = vec![0];

    let comparison = comparer.compare_tensors("empty", &data, &shape, &data, &shape);
    assert!(comparison.shape_match);
    // compute_diff_stats returns (0.0, 0.0, 0.0) for empty
    assert_eq!(comparison.l2_diff, Some(0.0));
    assert_eq!(comparison.max_diff, Some(0.0));
    assert_eq!(comparison.mean_diff, Some(0.0));
}

#[test]
fn test_compare_tensors_with_nan_in_data() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![1.0_f32, f32::NAN, 3.0];
    let target = vec![1.0_f32, 2.0, 3.0];
    let shape = vec![3];

    let comparison = comparer.compare_tensors("nan_test", &source, &shape, &target, &shape);
    // NaN pairs are skipped; only finite pairs contribute
    assert!(comparison.shape_match);
    assert!(comparison.l2_diff.is_some());
    // The NaN element is skipped, so only elements 0 and 2 are compared
    // Both are identical, so diff should be 0
    assert_eq!(comparison.l2_diff, Some(0.0));
    assert_eq!(comparison.max_diff, Some(0.0));
    assert_eq!(comparison.mean_diff, Some(0.0));
}

#[test]
fn test_compare_tensors_with_inf_in_data() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![1.0_f32, f32::INFINITY, 3.0];
    let target = vec![1.0_f32, 2.0, 3.0];
    let shape = vec![3];

    let comparison = comparer.compare_tensors("inf_test", &source, &shape, &target, &shape);
    assert!(comparison.shape_match);
    // Inf is not finite, so that pair is skipped
    assert_eq!(comparison.l2_diff, Some(0.0));
}

#[test]
fn test_compare_tensors_all_non_finite() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![f32::NAN, f32::INFINITY];
    let target = vec![f32::NAN, f32::NEG_INFINITY];
    let shape = vec![2];

    let comparison = comparer.compare_tensors("all_nan", &source, &shape, &target, &shape);
    assert!(comparison.shape_match);
    // All pairs skipped => count=0 => mean=0.0
    assert_eq!(comparison.l2_diff, Some(0.0));
    assert_eq!(comparison.max_diff, Some(0.0));
    assert_eq!(comparison.mean_diff, Some(0.0));
}

#[test]
fn test_compare_tensors_same_shape_different_data_length() {
    // Shapes match but data lengths differ (unusual/corrupt state)
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![1.0_f32, 2.0, 3.0, 4.0];
    let target = vec![1.0_f32, 2.0];
    let source_shape = vec![2, 2];
    let target_shape = vec![2, 2]; // same shape

    let comparison = comparer.compare_tensors(
        "mismatch_len",
        &source,
        &source_shape,
        &target,
        &target_shape,
    );
    // shape_match is true, but data.len() != data.len() => None diffs
    assert!(comparison.shape_match);
    assert!(comparison.l2_diff.is_none());
    assert!(comparison.max_diff.is_none());
    assert!(comparison.mean_diff.is_none());
}

#[test]
fn test_compare_tensors_single_element() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![3.14_f32];
    let target = vec![3.14_f32];
    let shape = vec![1];

    let comparison = comparer.compare_tensors("scalar", &source, &shape, &target, &shape);
    assert!(comparison.shape_match);
    assert!(comparison.is_match());
    assert_eq!(comparison.l2_diff, Some(0.0));
}

#[test]
fn test_compare_tensors_large_diff() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![0.0_f32; 4];
    let target = vec![100.0_f32; 4];
    let shape = vec![4];

    let comparison = comparer.compare_tensors("big_diff", &source, &shape, &target, &shape);
    assert!(comparison.shape_match);
    assert!(!comparison.is_match());
    // L2 = sqrt(4 * 100^2) = sqrt(40000) = 200
    let l2 = comparison.l2_diff.expect("should have l2");
    assert!((l2 - 200.0).abs() < 1e-6);
    let max = comparison.max_diff.expect("should have max");
    assert!((max - 100.0).abs() < 1e-6);
    let mean = comparison.mean_diff.expect("should have mean");
    assert!((mean - 100.0).abs() < 1e-6);
}

// --- compare_models with prefix stripping ---

#[test]
fn test_compare_models_with_source_prefix() {
    let config = CompareConfig {
        source_prefix: Some("model.".to_string()),
        ..CompareConfig::default()
    };
    let comparer = WeightComparer::new(config);

    let mut source = HashMap::new();
    source.insert(
        "model.layer1.weight".to_string(),
        (vec![1.0_f32, 2.0], vec![2]),
    );

    let mut target = HashMap::new();
    target.insert(
        "model.layer1.weight".to_string(),
        (vec![1.0_f32, 2.0], vec![2]),
    );

    let report = comparer.compare_models(&source, &target);
    assert_eq!(report.tensors.len(), 1);
    // The normalized name should strip the prefix
    assert_eq!(report.tensors[0].name, "layer1.weight");
    assert!(report.all_match());
}

#[test]
fn test_compare_models_prefix_no_match() {
    // Source prefix doesn't match tensor name => name unchanged
    let config = CompareConfig {
        source_prefix: Some("encoder.".to_string()),
        ..CompareConfig::default()
    };
    let comparer = WeightComparer::new(config);

    let mut source = HashMap::new();
    source.insert("decoder.weight".to_string(), (vec![1.0_f32], vec![1]));

    let mut target = HashMap::new();
    target.insert("decoder.weight".to_string(), (vec![1.0_f32], vec![1]));

    let report = comparer.compare_models(&source, &target);
    assert_eq!(report.tensors.len(), 1);
    // Prefix "encoder." doesn't match "decoder.weight" so name is unchanged
    assert_eq!(report.tensors[0].name, "decoder.weight");
}

#[test]
fn test_compare_models_all_source_only() {
    let comparer = WeightComparer::new(CompareConfig::default());

    let mut source = HashMap::new();
    source.insert("a.weight".to_string(), (vec![1.0_f32], vec![1]));
    source.insert("b.weight".to_string(), (vec![2.0_f32], vec![1]));

    let target: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    let report = comparer.compare_models(&source, &target);
    assert!(report.tensors.is_empty());
    assert_eq!(report.source_only.len(), 2);
    assert!(report.target_only.is_empty());
    assert!(!report.all_match());
}

#[test]
fn test_compare_models_all_target_only() {
    let comparer = WeightComparer::new(CompareConfig::default());

    let source: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    let mut target = HashMap::new();
    target.insert("x.weight".to_string(), (vec![1.0_f32], vec![1]));

    let report = comparer.compare_models(&source, &target);
    assert!(report.tensors.is_empty());
    assert!(report.source_only.is_empty());
    assert_eq!(report.target_only.len(), 1);
    assert!(!report.all_match());
}

#[test]
fn test_compare_models_mixed_match_mismatch() {
    let comparer = WeightComparer::new(CompareConfig::default());

    let mut source = HashMap::new();
    source.insert("good.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
    source.insert("bad.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

    let mut target = HashMap::new();
    target.insert("good.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
    target.insert("bad.weight".to_string(), (vec![100.0_f32, 200.0], vec![2]));

    let report = comparer.compare_models(&source, &target);
    assert_eq!(report.tensors.len(), 2);
    assert_eq!(report.match_count(), 1);
    assert_eq!(report.mismatch_count(), 1);
    assert!(!report.all_match());
}

#[test]
fn test_compare_models_shape_mismatch_in_common() {
    let comparer = WeightComparer::new(CompareConfig::default());

    let mut source = HashMap::new();
    source.insert("layer.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

    let mut target = HashMap::new();
    target.insert(
        "layer.weight".to_string(),
        (vec![1.0_f32, 2.0, 3.0], vec![3]),
    );

    let report = comparer.compare_models(&source, &target);
    assert_eq!(report.tensors.len(), 1);
    assert!(!report.tensors[0].shape_match);
    assert!(report.tensors[0].l2_diff.is_none());
    assert_eq!(report.mismatch_count(), 1);
    assert!(!report.all_match());
}

#[test]
fn test_compare_models_accumulates_l2_and_max() {
    let comparer = WeightComparer::new(CompareConfig::default());

    let mut source = HashMap::new();
    source.insert("a".to_string(), (vec![0.0_f32], vec![1]));
    source.insert("b".to_string(), (vec![0.0_f32], vec![1]));

    let mut target = HashMap::new();
    target.insert("a".to_string(), (vec![3.0_f32], vec![1]));
    target.insert("b".to_string(), (vec![4.0_f32], vec![1]));

    let report = comparer.compare_models(&source, &target);
    // total_l2_diff = sqrt(3^2 + 4^2) = sqrt(9+16) = 5
    assert!((report.total_l2_diff - 5.0).abs() < 1e-6);
    // global_max_diff = max(3, 4) = 4
    assert!((report.global_max_diff - 4.0).abs() < 1e-6);
}
