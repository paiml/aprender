use super::*;

#[test]
fn test_compare_tensors_different_shape() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        size_bytes: 800,
        stats: None,
    }];
    let t2 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![20, 20],
        size_bytes: 1600,
        stats: None,
    }];

    let mut diffs = Vec::new();
    let options = DiffOptions::default();
    compare_tensors(&t1, &t2, &options, &mut diffs);

    assert!(diffs.iter().any(|d| d.field.contains("shape")));
}

#[test]
fn test_compare_tensors_different_dtype() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        size_bytes: 800,
        stats: None,
    }];
    let t2 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "Q8_0".to_string(),
        shape: vec![10, 20],
        size_bytes: 400,
        stats: None,
    }];

    let mut diffs = Vec::new();
    let options = DiffOptions::default();
    compare_tensors(&t1, &t2, &options, &mut diffs);

    assert!(diffs.iter().any(|d| d.field.contains("dtype")));
}

#[test]
fn test_compare_tensors_with_filter() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![
        TensorInfo {
            name: "embed.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            size_bytes: 400,
            stats: None,
        },
        TensorInfo {
            name: "lm_head.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            size_bytes: 400,
            stats: None,
        },
    ];
    let t2 = vec![
        TensorInfo {
            name: "embed.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![200], // Different
            size_bytes: 800,
            stats: None,
        },
        TensorInfo {
            name: "lm_head.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![200], // Different
            size_bytes: 800,
            stats: None,
        },
    ];

    let mut diffs = Vec::new();
    let options = DiffOptions::new().with_filter("embed");
    compare_tensors(&t1, &t2, &options, &mut diffs);

    // Should only report embed differences due to filter
    assert!(diffs.iter().all(|d| d.field.contains("embed")));
}

#[test]
fn test_compare_tensors_with_stats() {
    use crate::format::rosetta::{TensorInfo, TensorStats};

    let t1 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10],
        size_bytes: 40,
        stats: Some(TensorStats {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.1,
        }),
    }];
    let t2 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10],
        size_bytes: 40,
        stats: Some(TensorStats {
            min: 0.0,
            max: 2.0,  // Different
            mean: 0.6, // Different
            std: 0.1,
        }),
    }];

    let mut diffs = Vec::new();
    let options = DiffOptions::new().with_stats();
    compare_tensors(&t1, &t2, &options, &mut diffs);

    assert!(diffs.iter().any(|d| d.field.contains("max")));
    assert!(diffs.iter().any(|d| d.field.contains("mean")));
}

#[test]
fn test_compare_tensor_stats_one_missing() {
    use crate::format::rosetta::{TensorInfo, TensorStats};

    let t1 = TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10],
        size_bytes: 40,
        stats: Some(TensorStats {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.1,
        }),
    };
    let t2 = TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10],
        size_bytes: 40,
        stats: None,
    };

    let mut diffs = Vec::new();
    compare_tensor_stats(&t1, &t2, &mut diffs);

    assert_eq!(diffs.len(), 1);
    assert!(diffs[0].field.contains("stats"));
}

// ========================================================================
// Integration Tests (File-based, minimal)
// ========================================================================

#[test]
fn test_diff_models_file_not_found() {
    let result = diff_models(
        Path::new("/nonexistent/a.apr"),
        Path::new("/nonexistent/b.apr"),
        DiffOptions::default(),
    );
    assert!(result.is_err());
}

// ====================================================================
// Coverage: compute_differences all branches
// ====================================================================

pub(crate) fn make_report(
    format: FormatType,
    size: usize,
    params: usize,
    arch: Option<&str>,
    quant: Option<&str>,
) -> InspectionReport {
    InspectionReport {
        format,
        file_size: size,
        metadata: std::collections::BTreeMap::new(),
        tensors: Vec::new(),
        total_params: params,
        quantization: quant.map(String::from),
        architecture: arch.map(String::from),
    }
}

#[test]
fn test_compute_differences_identical() {
    let r = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
    let diffs = compute_differences(&r, &r, &DiffOptions::default());
    assert!(diffs.is_empty());
}

#[test]
fn test_compute_differences_format_differs() {
    let r1 = make_report(FormatType::Apr, 1000, 100, None, None);
    let r2 = make_report(FormatType::Gguf, 1000, 100, None, None);
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    assert!(diffs.iter().any(|d| d.field == "format"));
}

#[test]
fn test_compute_differences_size_differs() {
    let r1 = make_report(FormatType::Apr, 1000, 100, None, None);
    let r2 = make_report(FormatType::Apr, 2000, 100, None, None);
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    assert!(diffs.iter().any(|d| d.field == "file_size"));
}

#[test]
fn test_compute_differences_params_differs() {
    let r1 = make_report(FormatType::Apr, 1000, 100, None, None);
    let r2 = make_report(FormatType::Apr, 1000, 200, None, None);
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    assert!(diffs.iter().any(|d| d.field == "total_params"));
}

#[test]
fn test_compute_differences_architecture_differs() {
    let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
    let r2 = make_report(FormatType::Apr, 1000, 100, Some("qwen2"), None);
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    let arch_diff = diffs.iter().find(|d| d.field == "architecture").unwrap();
    assert!(arch_diff.value1.contains("llama"));
    assert!(arch_diff.value2.contains("qwen2"));
}

#[test]
fn test_compute_differences_architecture_one_none() {
    let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
    let r2 = make_report(FormatType::Apr, 1000, 100, None, None);
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    let arch_diff = diffs.iter().find(|d| d.field == "architecture").unwrap();
    assert!(arch_diff.value2.contains("(none)"));
}

#[test]
fn test_compute_differences_quantization_differs() {
    let r1 = make_report(FormatType::Apr, 1000, 100, None, Some("Q4_K"));
    let r2 = make_report(FormatType::Apr, 1000, 100, None, Some("Q8_0"));
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    assert!(diffs.iter().any(|d| d.field == "quantization"));
}

#[test]
fn test_compute_differences_quantization_one_none() {
    let r1 = make_report(FormatType::Apr, 1000, 100, None, Some("Q4_K"));
    let r2 = make_report(FormatType::Apr, 1000, 100, None, None);
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    let q_diff = diffs.iter().find(|d| d.field == "quantization").unwrap();
    assert!(q_diff.value2.contains("(none)"));
}

#[test]
fn test_compute_differences_multiple() {
    let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), Some("F32"));
    let r2 = make_report(FormatType::Gguf, 2000, 200, Some("qwen2"), Some("Q4_K"));
    let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
    assert!(diffs.len() >= 4);
}

#[test]
fn test_compute_differences_no_tensors() {
    let r = make_report(FormatType::Apr, 1000, 100, None, None);
    let opts = DiffOptions::new().without_tensors();
    let diffs = compute_differences(&r, &r, &opts);
    assert!(diffs.is_empty());
}

// ====================================================================
// Coverage: compare_tensor_stats all stat branches
// ====================================================================

#[test]
fn test_compare_tensor_stats_min_differs() {
    use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
    let mut diffs = Vec::new();
    let t1 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: Some(RTS {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.1,
        }),
    };
    let t2 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: Some(RTS {
            min: 0.5,
            max: 1.0,
            mean: 0.5,
            std: 0.1,
        }),
    };
    compare_tensor_stats(&t1, &t2, &mut diffs);
    assert!(diffs.iter().any(|d| d.field.contains("min")));
    assert!(!diffs.iter().any(|d| d.field.contains("max")));
}

#[test]
fn test_compare_tensor_stats_all_differ() {
    use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
    let mut diffs = Vec::new();
    let t1 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: Some(RTS {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.1,
        }),
    };
    let t2 = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: Some(RTS {
            min: 1.0,
            max: 2.0,
            mean: 1.5,
            std: 0.5,
        }),
    };
    compare_tensor_stats(&t1, &t2, &mut diffs);
    assert_eq!(diffs.len(), 4); // min, max, mean, std
}

#[test]
fn test_compare_tensor_stats_none_none() {
    use crate::format::rosetta::TensorInfo as RTI;
    let mut diffs = Vec::new();
    let t = RTI {
        name: "w".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        stats: None,
    };
    compare_tensor_stats(&t, &t, &mut diffs);
    assert!(diffs.is_empty());
}

// ====================================================================
// Coverage: DiffReport additional method tests
// ====================================================================

#[test]
fn test_diff_report_by_category_filtering() {
    let report = DiffReport {
        path1: "a".to_string(),
        path2: "b".to_string(),
        format1: "APR".to_string(),
        format2: "APR".to_string(),
        differences: vec![
            DiffEntry {
                field: "file_size".to_string(),
                value1: "100".to_string(),
                value2: "200".to_string(),
                category: DiffCategory::Size,
            },
            DiffEntry {
                field: "architecture".to_string(),
                value1: "llama".to_string(),
                value2: "qwen2".to_string(),
                category: DiffCategory::Metadata,
            },
        ],
        inspection1: None,
        inspection2: None,
    };
    assert_eq!(report.differences_by_category(DiffCategory::Size).len(), 1);
    assert_eq!(
        report.differences_by_category(DiffCategory::Metadata).len(),
        1
    );
    assert_eq!(
        report.differences_by_category(DiffCategory::Format).len(),
        0
    );
}

// ====================================================================
// Coverage: normalize_dtype exhaustive branch tests
// ====================================================================

#[test]
fn test_normalize_dtype_numeric_codes() {
    // All GGUF numeric codes
    assert_eq!(normalize_dtype("0"), "F32");
    assert_eq!(normalize_dtype("1"), "F16");
    assert_eq!(normalize_dtype("2"), "Q4_0");
    assert_eq!(normalize_dtype("3"), "Q4_1");
    assert_eq!(normalize_dtype("6"), "Q5_0");
    assert_eq!(normalize_dtype("7"), "Q5_1");
    assert_eq!(normalize_dtype("8"), "Q8_0");
    assert_eq!(normalize_dtype("9"), "Q8_1");
    assert_eq!(normalize_dtype("10"), "Q2_K");
    assert_eq!(normalize_dtype("11"), "Q3_K");
    assert_eq!(normalize_dtype("12"), "Q4_K");
    assert_eq!(normalize_dtype("13"), "Q5_K");
    assert_eq!(normalize_dtype("14"), "Q6_K");
    assert_eq!(normalize_dtype("15"), "Q8_K");
    assert_eq!(normalize_dtype("16"), "IQ2_XXS");
    assert_eq!(normalize_dtype("17"), "IQ2_XS");
    assert_eq!(normalize_dtype("18"), "IQ3_XXS");
    assert_eq!(normalize_dtype("19"), "IQ1_S");
}
