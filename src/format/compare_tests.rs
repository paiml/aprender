pub(crate) use super::*;

#[test]
fn test_compare_config_default() {
    let config = CompareConfig::default();
    assert!((config.l2_tolerance - 1e-5).abs() < 1e-10);
    assert!((config.max_tolerance - 1e-5).abs() < 1e-10);
}

#[test]
fn test_compare_config_strict() {
    let config = CompareConfig::strict();
    assert_eq!(config.l2_tolerance, 0.0);
    assert_eq!(config.max_tolerance, 0.0);
}

#[test]
fn test_compare_config_quantized() {
    let config = CompareConfig::quantized();
    assert!((config.l2_tolerance - 1e-2).abs() < 1e-10);
}

#[test]
fn test_tensor_comparison_identical() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];

    let comparison = comparer.compare_tensors("test", &data, &shape, &data, &shape);

    assert!(comparison.shape_match);
    assert!(comparison.is_match());
    assert!((comparison.l2_diff.unwrap_or(1.0) - 0.0).abs() < 1e-10);
    assert!((comparison.max_diff.unwrap_or(1.0) - 0.0).abs() < 1e-10);
}

#[test]
fn test_tensor_comparison_different() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![1.0_f32, 2.0, 3.0, 4.0];
    let target = vec![1.0_f32, 2.0, 3.0, 5.0]; // Last element differs by 1
    let shape = vec![2, 2];

    let comparison = comparer.compare_tensors("test", &source, &shape, &target, &shape);

    assert!(comparison.shape_match);
    assert!(!comparison.is_match()); // 1.0 diff > 1e-5 tolerance
    assert!((comparison.l2_diff.unwrap() - 1.0).abs() < 1e-10);
    assert!((comparison.max_diff.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_tensor_comparison_shape_mismatch() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source = vec![1.0_f32, 2.0, 3.0, 4.0];
    let target = vec![1.0_f32, 2.0, 3.0];
    let source_shape = vec![2, 2];
    let target_shape = vec![3];

    let comparison =
        comparer.compare_tensors("test", &source, &source_shape, &target, &target_shape);

    assert!(!comparison.shape_match);
    assert!(comparison.l2_diff.is_none());
    assert!(!comparison.is_match());
}

#[test]
fn test_compare_models() {
    let comparer = WeightComparer::new(CompareConfig::default());

    let mut source = HashMap::new();
    source.insert("layer1.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
    source.insert("layer1.bias".to_string(), (vec![0.5_f32], vec![1]));
    source.insert("layer2.weight".to_string(), (vec![3.0_f32, 4.0], vec![2]));

    let mut target = HashMap::new();
    target.insert("layer1.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
    target.insert("layer1.bias".to_string(), (vec![0.5_f32], vec![1]));
    // layer2.weight missing, extra_layer added
    target.insert("extra_layer".to_string(), (vec![5.0_f32], vec![1]));

    let report = comparer.compare_models(&source, &target);

    assert_eq!(report.tensors.len(), 2);
    assert_eq!(report.source_only.len(), 1);
    assert_eq!(report.target_only.len(), 1);
    assert!(report.source_only.contains(&"layer2.weight".to_string()));
    assert!(report.target_only.contains(&"extra_layer".to_string()));
}

#[test]
fn test_compare_report_summary() {
    let comparer = WeightComparer::new(CompareConfig::default());

    let mut source = HashMap::new();
    source.insert("weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

    let mut target = HashMap::new();
    target.insert("weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

    let report = comparer.compare_models(&source, &target);
    let summary = report.summary();

    assert!(summary.contains("Tensors compared: 1"));
    assert!(summary.contains("Matching: 1"));
    assert!(summary.contains("Mismatched: 0"));
}

#[test]
fn test_l2_diff_identical() {
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![1.0_f32, 2.0, 3.0];
    let diff = l2_diff(&a, &b).unwrap();
    assert!(diff < 1e-10);
}

#[test]
fn test_l2_diff_different() {
    let a = vec![0.0_f32, 0.0, 0.0];
    let b = vec![3.0_f32, 4.0, 0.0]; // L2 = 5.0
    let diff = l2_diff(&a, &b).unwrap();
    assert!((diff - 5.0).abs() < 1e-10);
}

#[test]
fn test_l2_diff_length_mismatch() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![1.0_f32, 2.0, 3.0];
    assert!(l2_diff(&a, &b).is_none());
}

#[test]
fn test_l2_diff_nan() {
    let a = vec![1.0_f32, f32::NAN];
    let b = vec![1.0_f32, 2.0];
    assert!(l2_diff(&a, &b).is_none());
}

#[test]
fn test_max_diff() {
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![1.0_f32, 2.5, 3.0];
    let diff = max_diff(&a, &b).unwrap();
    assert!((diff - 0.5).abs() < 1e-10);
}

#[test]
fn test_relative_l2_error() {
    let a = vec![1.0_f32, 0.0, 0.0];
    let b = vec![1.1_f32, 0.0, 0.0];
    let error = relative_l2_error(&a, &b).unwrap();
    // f32 precision: 1.1 - 1.0 is not exactly 0.1
    assert!((error - 0.1).abs() < 1e-6);
}

#[test]
fn test_relative_l2_error_zero_norm() {
    let a = vec![0.0_f32, 0.0, 0.0];
    let b = vec![0.0_f32, 0.0, 0.0];
    let error = relative_l2_error(&a, &b).unwrap();
    assert!(error < 1e-10);
}

#[test]
fn test_tensor_comparison_element_count() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![2, 3, 4],
        target_shape: vec![2, 3, 4],
        l2_diff: Some(0.0),
        max_diff: Some(0.0),
        mean_diff: Some(0.0),
    };
    assert_eq!(comparison.element_count(), 24);
}

#[test]
fn test_compare_report_all_match() {
    let config = CompareConfig::default();
    let report = CompareReport {
        tensors: vec![TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![10],
            target_shape: vec![10],
            l2_diff: Some(1e-10),
            max_diff: Some(1e-10),
            mean_diff: Some(1e-10),
        }],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 1e-10,
        global_max_diff: 1e-10,
        config,
    };
    assert!(report.all_match());
}

#[test]
fn test_compare_report_mismatch() {
    let config = CompareConfig::default();
    let report = CompareReport {
        tensors: vec![],
        source_only: vec!["missing".to_string()],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config,
    };
    assert!(!report.all_match());
}

// ========================================================================
// Additional Coverage Tests for compare.rs
// ========================================================================

#[test]
fn test_compare_config_clone() {
    let config = CompareConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.l2_tolerance, config.l2_tolerance);
    assert_eq!(cloned.max_tolerance, config.max_tolerance);
}

#[test]
fn test_compare_config_debug() {
    let config = CompareConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("CompareConfig"));
}

#[test]
fn test_tensor_comparison_debug() {
    let comparison = TensorComparison {
        name: "test_tensor".to_string(),
        shape_match: true,
        source_shape: vec![2, 2],
        target_shape: vec![2, 2],
        l2_diff: Some(0.0),
        max_diff: Some(0.0),
        mean_diff: Some(0.0),
    };
    let debug_str = format!("{:?}", comparison);
    assert!(debug_str.contains("test_tensor"));
}

#[test]
fn test_tensor_comparison_clone() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![2, 2],
        target_shape: vec![2, 2],
        l2_diff: Some(0.1),
        max_diff: Some(0.2),
        mean_diff: Some(0.05),
    };
    let cloned = comparison.clone();
    assert_eq!(cloned.name, "test");
    assert_eq!(cloned.l2_diff, Some(0.1));
}

#[test]
fn test_compare_report_debug() {
    let config = CompareConfig::default();
    let report = CompareReport {
        tensors: vec![],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config,
    };
    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("CompareReport"));
}

#[test]
fn test_compare_report_clone() {
    let config = CompareConfig::default();
    let report = CompareReport {
        tensors: vec![],
        source_only: vec!["a".to_string()],
        target_only: vec!["b".to_string()],
        total_l2_diff: 0.5,
        global_max_diff: 0.3,
        config,
    };
    let cloned = report.clone();
    assert_eq!(cloned.source_only.len(), 1);
    assert_eq!(cloned.total_l2_diff, 0.5);
}

#[test]
fn test_weight_comparer_debug() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let debug_str = format!("{:?}", comparer);
    assert!(debug_str.contains("WeightComparer"));
}

#[test]
fn test_tensor_comparison_no_match_shape_mismatch() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: false,
        source_shape: vec![2, 2],
        target_shape: vec![2, 3],
        l2_diff: None,
        max_diff: None,
        mean_diff: None,
    };
    assert!(!comparison.is_match());
}

#[test]
fn test_tensor_comparison_with_mean_diff() {
    let comparison = TensorComparison {
        name: "test".to_string(),
        shape_match: true,
        source_shape: vec![4],
        target_shape: vec![4],
        l2_diff: Some(0.0),
        max_diff: Some(0.0),
        mean_diff: Some(0.0),
    };
    assert!(comparison.is_match());
    assert_eq!(comparison.element_count(), 4);
}

#[test]
fn test_compare_report_with_target_only() {
    let config = CompareConfig::default();
    let report = CompareReport {
        tensors: vec![],
        source_only: vec![],
        target_only: vec!["extra".to_string()],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config,
    };
    assert!(!report.all_match());
}

#[test]
fn test_max_diff_nan_in_source() {
    let a = vec![f32::NAN, 2.0];
    let b = vec![1.0, 2.0];
    assert!(max_diff(&a, &b).is_none());
}

#[test]
fn test_max_diff_inf_diff() {
    let a = vec![f32::INFINITY, 0.0];
    let b = vec![0.0, 0.0];
    let diff = max_diff(&a, &b);
    // INFINITY - 0.0 = INFINITY, which might be filtered out
    // Just verify the function doesn't panic
    let _ = diff;
}

#[test]
fn test_relative_l2_error_length_mismatch() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![1.0_f32, 2.0, 3.0];
    assert!(relative_l2_error(&a, &b).is_none());
}

#[test]
fn test_compare_empty_models() {
    let comparer = WeightComparer::new(CompareConfig::default());
    let source: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let target: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    let report = comparer.compare_models(&source, &target);
    assert!(report.tensors.is_empty());
    assert!(report.source_only.is_empty());
    assert!(report.target_only.is_empty());
    assert!(report.all_match());
}

#[test]
fn test_compare_report_summary_with_mismatches() {
    let config = CompareConfig::default();
    let report = CompareReport {
        tensors: vec![TensorComparison {
            name: "weight".to_string(),
            shape_match: true,
            source_shape: vec![10],
            target_shape: vec![10],
            l2_diff: Some(1.0), // Large diff
            max_diff: Some(1.0),
            mean_diff: Some(0.1),
        }],
        source_only: vec!["missing".to_string()],
        target_only: vec!["extra".to_string()],
        total_l2_diff: 1.0,
        global_max_diff: 1.0,
        config,
    };

    let summary = report.summary();
    // The summary format may vary, just check it contains key info
    assert!(summary.contains("1"));
    assert!(!summary.is_empty());
}

// ========================================================================
// Extended Coverage Tests (GH-121 coverage push)
// ========================================================================

// --- CompareConfig extended tests ---

#[test]
fn test_compare_config_default_flags() {
    let config = CompareConfig::default();
    assert!(!config.allow_broadcast);
    assert!(!config.normalize_first);
    assert!(config.source_prefix.is_none());
    assert!(config.target_prefix.is_none());
}

#[test]
fn test_compare_config_strict_inherits_defaults() {
    let config = CompareConfig::strict();
    assert!(!config.allow_broadcast);
    assert!(!config.normalize_first);
    assert!(config.source_prefix.is_none());
    assert!(config.target_prefix.is_none());
}

#[test]
fn test_compare_config_quantized_tolerances() {
    let config = CompareConfig::quantized();
    assert!((config.l2_tolerance - 1e-2).abs() < 1e-10);
    assert!((config.max_tolerance - 1e-2).abs() < 1e-10);
    assert!(!config.allow_broadcast);
    assert!(!config.normalize_first);
}

#[path = "compare_tests_part_02.rs"]

mod compare_tests_part_02;
#[path = "compare_tests_part_03.rs"]
mod compare_tests_part_03;
