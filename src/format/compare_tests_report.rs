use super::*;

// --- CompareReport::summary() branch coverage ---

#[test]
fn test_summary_no_source_only_no_target_only() {
    let report = CompareReport {
        tensors: vec![TensorComparison {
            name: "w".to_string(),
            shape_match: true,
            source_shape: vec![2],
            target_shape: vec![2],
            l2_diff: Some(0.0),
            max_diff: Some(0.0),
            mean_diff: Some(0.0),
        }],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config: CompareConfig::default(),
    };
    let summary = report.summary();
    assert!(summary.contains("Weight Comparison Report"));
    assert!(summary.contains("Tensors compared: 1"));
    assert!(summary.contains("Matching: 1"));
    assert!(summary.contains("Mismatched: 0"));
    assert!(summary.contains("Source only: 0"));
    assert!(summary.contains("Target only: 0"));
    // Should NOT contain the "Tensors only in source/target" sections
    assert!(!summary.contains("Tensors only in source:"));
    assert!(!summary.contains("Tensors only in target:"));
    assert!(!summary.contains("Mismatched tensors:"));
}

#[test]
fn test_summary_source_only_section() {
    let report = CompareReport {
        tensors: vec![],
        source_only: vec!["layer1.weight".to_string(), "layer2.bias".to_string()],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config: CompareConfig::default(),
    };
    let summary = report.summary();
    assert!(summary.contains("Tensors only in source:"));
    assert!(summary.contains("  - layer1.weight"));
    assert!(summary.contains("  - layer2.bias"));
    assert!(!summary.contains("Tensors only in target:"));
}

#[test]
fn test_summary_target_only_section() {
    let report = CompareReport {
        tensors: vec![],
        source_only: vec![],
        target_only: vec!["extra.weight".to_string()],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config: CompareConfig::default(),
    };
    let summary = report.summary();
    assert!(!summary.contains("Tensors only in source:"));
    assert!(summary.contains("Tensors only in target:"));
    assert!(summary.contains("  - extra.weight"));
}

#[test]
fn test_summary_mismatched_tensor_with_shape_match() {
    // Mismatched tensor where shapes match but l2 exceeds tolerance
    let report = CompareReport {
        tensors: vec![TensorComparison {
            name: "attn.weight".to_string(),
            shape_match: true,
            source_shape: vec![4, 4],
            target_shape: vec![4, 4],
            l2_diff: Some(0.5),
            max_diff: Some(0.1),
            mean_diff: Some(0.01),
        }],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.5,
        global_max_diff: 0.1,
        config: CompareConfig::default(),
    };
    let summary = report.summary();
    assert!(summary.contains("Mismatched tensors:"));
    // shape_match=true -> format shows single shape
    assert!(summary.contains("[4, 4]"));
    assert!(summary.contains("L2="));
    assert!(summary.contains("attn.weight"));
}

#[test]
fn test_summary_mismatched_tensor_with_shape_mismatch() {
    // Mismatched tensor where shapes differ -> "shape mismatch" string
    let report = CompareReport {
        tensors: vec![TensorComparison {
            name: "ffn.weight".to_string(),
            shape_match: false,
            source_shape: vec![4, 4],
            target_shape: vec![4, 8],
            l2_diff: None,
            max_diff: None,
            mean_diff: None,
        }],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config: CompareConfig::default(),
    };
    let summary = report.summary();
    assert!(summary.contains("Mismatched tensors:"));
    // shape_match=false -> format shows "vs"
    assert!(summary.contains("[4, 4] vs [4, 8]"));
    // l2_diff is None -> "shape mismatch" text
    assert!(summary.contains("shape mismatch"));
    assert!(summary.contains("ffn.weight"));
}

#[test]
fn test_summary_tolerances_displayed() {
    let config = CompareConfig {
        l2_tolerance: 1e-3,
        max_tolerance: 2e-4,
        ..CompareConfig::default()
    };
    let report = CompareReport {
        tensors: vec![],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config,
    };
    let summary = report.summary();
    assert!(summary.contains("L2 tolerance:"));
    assert!(summary.contains("Max tolerance:"));
}

#[test]
fn test_summary_total_l2_and_global_max_displayed() {
    let report = CompareReport {
        tensors: vec![],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 1.234_567,
        global_max_diff: 0.987_654,
        config: CompareConfig::default(),
    };
    let summary = report.summary();
    assert!(summary.contains("Total L2 diff:"));
    assert!(summary.contains("Global max diff:"));
}

// --- CompareReport::match_count / mismatch_count ---

#[test]
fn test_match_count_all_matching() {
    let report = CompareReport {
        tensors: vec![
            TensorComparison {
                name: "a".to_string(),
                shape_match: true,
                source_shape: vec![1],
                target_shape: vec![1],
                l2_diff: Some(0.0),
                max_diff: Some(0.0),
                mean_diff: Some(0.0),
            },
            TensorComparison {
                name: "b".to_string(),
                shape_match: true,
                source_shape: vec![1],
                target_shape: vec![1],
                l2_diff: Some(1e-8),
                max_diff: Some(1e-8),
                mean_diff: Some(1e-8),
            },
        ],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config: CompareConfig::default(),
    };
    assert_eq!(report.match_count(), 2);
    assert_eq!(report.mismatch_count(), 0);
}

#[test]
fn test_match_count_none_matching() {
    let report = CompareReport {
        tensors: vec![TensorComparison {
            name: "x".to_string(),
            shape_match: false,
            source_shape: vec![1],
            target_shape: vec![2],
            l2_diff: None,
            max_diff: None,
            mean_diff: None,
        }],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config: CompareConfig::default(),
    };
    assert_eq!(report.match_count(), 0);
    assert_eq!(report.mismatch_count(), 1);
}

#[test]
fn test_match_count_empty_tensors() {
    let report = CompareReport {
        tensors: vec![],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 0.0,
        global_max_diff: 0.0,
        config: CompareConfig::default(),
    };
    assert_eq!(report.match_count(), 0);
    assert_eq!(report.mismatch_count(), 0);
}

#[test]
fn test_match_count_with_quantized_tolerance() {
    let report = CompareReport {
        tensors: vec![TensorComparison {
            name: "q".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(5e-3), // exceeds default but within quantized
            max_diff: Some(5e-3),
            mean_diff: Some(1e-3),
        }],
        source_only: vec![],
        target_only: vec![],
        total_l2_diff: 5e-3,
        global_max_diff: 5e-3,
        config: CompareConfig::quantized(),
    };
    assert_eq!(report.match_count(), 1);
    assert_eq!(report.mismatch_count(), 0);
    assert!(report.all_match());
}

// --- Utility function edge cases ---

#[test]
fn test_l2_diff_empty_slices() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let diff = l2_diff(&a, &b).expect("should succeed for empty");
    assert_eq!(diff, 0.0);
}

#[test]
fn test_l2_diff_nan_in_target() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![1.0_f32, f32::NAN];
    assert!(l2_diff(&a, &b).is_none());
}

#[test]
fn test_l2_diff_inf_in_source() {
    let a = vec![f32::INFINITY, 2.0];
    let b = vec![1.0_f32, 2.0];
    assert!(l2_diff(&a, &b).is_none());
}

#[test]
fn test_l2_diff_neg_inf() {
    let a = vec![f32::NEG_INFINITY];
    let b = vec![0.0_f32];
    assert!(l2_diff(&a, &b).is_none());
}

#[test]
fn test_l2_diff_single_element() {
    let a = vec![5.0_f32];
    let b = vec![2.0_f32];
    let diff = l2_diff(&a, &b).expect("should succeed");
    assert!((diff - 3.0).abs() < 1e-6);
}

#[test]
fn test_max_diff_empty_slices() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let diff = max_diff(&a, &b).expect("should succeed for empty");
    assert_eq!(diff, 0.0);
}

#[test]
fn test_max_diff_length_mismatch() {
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![1.0_f32];
    assert!(max_diff(&a, &b).is_none());
}

#[test]
fn test_max_diff_nan_in_target() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![1.0_f32, f32::NAN];
    assert!(max_diff(&a, &b).is_none());
}

#[test]
fn test_max_diff_identical() {
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![1.0_f32, 2.0, 3.0];
    let diff = max_diff(&a, &b).expect("should succeed");
    assert_eq!(diff, 0.0);
}

#[test]
fn test_max_diff_negative_values() {
    let a = vec![-5.0_f32, -3.0];
    let b = vec![-2.0_f32, -1.0];
    let diff = max_diff(&a, &b).expect("should succeed");
    // max(|-5 - (-2)|, |-3 - (-1)|) = max(3, 2) = 3
    assert!((diff - 3.0).abs() < 1e-6);
}

#[test]
fn test_relative_l2_error_nan_input() {
    let a = vec![f32::NAN];
    let b = vec![1.0_f32];
    // l2_diff returns None for NaN -> relative_l2_error returns None
    assert!(relative_l2_error(&a, &b).is_none());
}

#[test]
fn test_relative_l2_error_zero_source_nonzero_diff() {
    let a = vec![0.0_f32, 0.0];
    let b = vec![1.0_f32, 0.0];
    let error = relative_l2_error(&a, &b).expect("should succeed");
    // a_norm ~ 0, diff_norm = 1.0 > EPSILON -> INFINITY
    assert!(error.is_infinite());
}

#[test]
fn test_relative_l2_error_identical_nonzero() {
    let a = vec![3.0_f32, 4.0]; // norm = 5.0
    let b = vec![3.0_f32, 4.0];
    let error = relative_l2_error(&a, &b).expect("should succeed");
    assert!(error < 1e-10);
}

#[test]
fn test_relative_l2_error_proportional() {
    // If b = 2*a, diff = a, relative error = ||a|| / ||a|| = 1.0
    let a = vec![3.0_f32, 4.0]; // norm = 5.0
    let b = vec![6.0_f32, 8.0]; // diff norm = 5.0
    let error = relative_l2_error(&a, &b).expect("should succeed");
    assert!((error - 1.0).abs() < 1e-6);
}

// --- WeightComparer Clone/Debug ---

#[test]
fn test_weight_comparer_clone() {
    let comparer = WeightComparer::new(CompareConfig::quantized());
    let cloned = comparer.clone();
    let debug_orig = format!("{:?}", comparer);
    let debug_clone = format!("{:?}", cloned);
    assert_eq!(debug_orig, debug_clone);
}

// --- Integration-level: full round-trip ---

#[test]
fn test_full_round_trip_matching_model() {
    let config = CompareConfig::default();
    let comparer = WeightComparer::new(config);

    let mut model = HashMap::new();
    model.insert(
        "embed.weight".to_string(),
        (vec![0.1_f32, 0.2, 0.3, 0.4], vec![2, 2]),
    );
    model.insert("head.bias".to_string(), (vec![0.01_f32, 0.02], vec![2]));

    let report = comparer.compare_models(&model, &model);
    assert!(report.all_match());
    assert_eq!(report.match_count(), 2);
    assert_eq!(report.mismatch_count(), 0);
    assert!(report.source_only.is_empty());
    assert!(report.target_only.is_empty());
    assert!(report.total_l2_diff < 1e-10);
    assert!(report.global_max_diff < 1e-10);

    let summary = report.summary();
    assert!(summary.contains("Matching: 2"));
    assert!(summary.contains("Mismatched: 0"));
}

#[test]
fn test_full_round_trip_mismatching_model() {
    let config = CompareConfig::strict();
    let comparer = WeightComparer::new(config);

    let mut source = HashMap::new();
    source.insert("w1".to_string(), (vec![1.0_f32], vec![1]));
    source.insert("w2".to_string(), (vec![2.0_f32], vec![1]));
    source.insert("w3".to_string(), (vec![3.0_f32], vec![1]));

    let mut target = HashMap::new();
    target.insert("w1".to_string(), (vec![1.0_f32], vec![1])); // match
    target.insert("w2".to_string(), (vec![2.5_f32], vec![1])); // mismatch
                                                               // w3 missing from target, w4 extra in target
    target.insert("w4".to_string(), (vec![4.0_f32], vec![1]));

    let report = comparer.compare_models(&source, &target);
    assert!(!report.all_match());

    let summary = report.summary();
    assert!(summary.contains("Source only: 1"));
    assert!(summary.contains("Target only: 1"));
}
