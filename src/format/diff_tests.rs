pub(crate) use super::*;
pub(crate) use diff_tests_tensor_cmp::make_report;

// ========================================================================
// DiffOptions Tests
// ========================================================================

#[test]
fn test_diff_options_default() {
    let opts = DiffOptions::default();
    assert!(opts.compare_tensors);
    assert!(opts.compare_metadata);
    assert!(!opts.compare_stats);
    assert!(opts.tensor_filter.is_none());
}

#[test]
fn test_diff_options_new() {
    let opts = DiffOptions::new();
    assert!(opts.compare_tensors);
    assert!(opts.compare_metadata);
}

#[test]
fn test_diff_options_with_tensors() {
    let opts = DiffOptions::new().without_tensors().with_tensors();
    assert!(opts.compare_tensors);
}

#[test]
fn test_diff_options_without_tensors() {
    let opts = DiffOptions::new().without_tensors();
    assert!(!opts.compare_tensors);
}

#[test]
fn test_diff_options_with_metadata() {
    let opts = DiffOptions::new().without_metadata().with_metadata();
    assert!(opts.compare_metadata);
}

#[test]
fn test_diff_options_without_metadata() {
    let opts = DiffOptions::new().without_metadata();
    assert!(!opts.compare_metadata);
}

#[test]
fn test_diff_options_with_stats() {
    let opts = DiffOptions::new().with_stats();
    assert!(opts.compare_stats);
}

#[test]
fn test_diff_options_with_filter() {
    let opts = DiffOptions::new().with_filter("embed");
    assert_eq!(opts.tensor_filter, Some("embed".to_string()));
}

// ========================================================================
// DiffCategory Tests
// ========================================================================

#[test]
fn test_diff_category_names() {
    assert_eq!(DiffCategory::Format.name(), "format");
    assert_eq!(DiffCategory::Metadata.name(), "metadata");
    assert_eq!(DiffCategory::Tensor.name(), "tensor");
    assert_eq!(DiffCategory::Quantization.name(), "quantization");
    assert_eq!(DiffCategory::Size.name(), "size");
}

// ========================================================================
// DiffEntry Tests
// ========================================================================

#[test]
fn test_diff_entry_serialization() {
    let entry = DiffEntry {
        field: "version".to_string(),
        value1: "1.0".to_string(),
        value2: "2.0".to_string(),
        category: DiffCategory::Format,
    };
    let json = serde_json::to_string(&entry).expect("serialize");
    assert!(json.contains("version"));
    assert!(json.contains("1.0"));
    assert!(json.contains("2.0"));
    assert!(json.contains("Format"));
}

#[test]
fn test_diff_entry_equality() {
    let entry1 = DiffEntry {
        field: "test".to_string(),
        value1: "a".to_string(),
        value2: "b".to_string(),
        category: DiffCategory::Metadata,
    };
    let entry2 = entry1.clone();
    assert_eq!(entry1, entry2);
}

// ========================================================================
// DiffReport Tests
// ========================================================================

#[test]
fn test_diff_report_identical() {
    let report = DiffReport {
        path1: "a.apr".to_string(),
        path2: "b.apr".to_string(),
        format1: "APR".to_string(),
        format2: "APR".to_string(),
        differences: vec![],
        inspection1: None,
        inspection2: None,
    };
    assert!(report.is_identical());
    assert_eq!(report.diff_count(), 0);
    assert!(report.same_format());
}

#[test]
fn test_diff_report_with_differences() {
    let report = DiffReport {
        path1: "a.apr".to_string(),
        path2: "b.gguf".to_string(),
        format1: "APR".to_string(),
        format2: "GGUF".to_string(),
        differences: vec![
            DiffEntry {
                field: "format".to_string(),
                value1: "APR".to_string(),
                value2: "GGUF".to_string(),
                category: DiffCategory::Format,
            },
            DiffEntry {
                field: "tensor_count".to_string(),
                value1: "10".to_string(),
                value2: "12".to_string(),
                category: DiffCategory::Tensor,
            },
        ],
        inspection1: None,
        inspection2: None,
    };
    assert!(!report.is_identical());
    assert_eq!(report.diff_count(), 2);
    assert!(!report.same_format());
}

#[test]
fn test_diff_report_by_category() {
    let report = DiffReport {
        path1: "a.apr".to_string(),
        path2: "b.apr".to_string(),
        format1: "APR".to_string(),
        format2: "APR".to_string(),
        differences: vec![
            DiffEntry {
                field: "tensor_count".to_string(),
                value1: "10".to_string(),
                value2: "12".to_string(),
                category: DiffCategory::Tensor,
            },
            DiffEntry {
                field: "metadata.name".to_string(),
                value1: "model_a".to_string(),
                value2: "model_b".to_string(),
                category: DiffCategory::Metadata,
            },
            DiffEntry {
                field: "tensor.embed.shape".to_string(),
                value1: "[100]".to_string(),
                value2: "[200]".to_string(),
                category: DiffCategory::Tensor,
            },
        ],
        inspection1: None,
        inspection2: None,
    };

    let tensor_diffs = report.differences_by_category(DiffCategory::Tensor);
    assert_eq!(tensor_diffs.len(), 2);

    let metadata_diffs = report.differences_by_category(DiffCategory::Metadata);
    assert_eq!(metadata_diffs.len(), 1);

    let format_diffs = report.differences_by_category(DiffCategory::Format);
    assert_eq!(format_diffs.len(), 0);
}

#[test]
fn test_diff_report_summary_identical() {
    let report = DiffReport {
        path1: "a.apr".to_string(),
        path2: "b.apr".to_string(),
        format1: "APR".to_string(),
        format2: "APR".to_string(),
        differences: vec![],
        inspection1: None,
        inspection2: None,
    };
    assert!(report.summary().contains("IDENTICAL"));
}

#[test]
fn test_diff_report_summary_different() {
    let report = DiffReport {
        path1: "a.apr".to_string(),
        path2: "b.apr".to_string(),
        format1: "APR".to_string(),
        format2: "APR".to_string(),
        differences: vec![DiffEntry {
            field: "test".to_string(),
            value1: "a".to_string(),
            value2: "b".to_string(),
            category: DiffCategory::Metadata,
        }],
        inspection1: None,
        inspection2: None,
    };
    assert!(report.summary().contains("differ"));
    assert!(report.summary().contains("1"));
}

#[test]
fn test_diff_report_serialization() {
    let report = DiffReport {
        path1: "a.apr".to_string(),
        path2: "b.apr".to_string(),
        format1: "APR".to_string(),
        format2: "APR".to_string(),
        differences: vec![],
        inspection1: None,
        inspection2: None,
    };
    let json = serde_json::to_string(&report).expect("serialize");
    assert!(json.contains("a.apr"));
    assert!(json.contains("b.apr"));
}

// ========================================================================
// Helper Function Tests
// ========================================================================

#[test]
fn test_format_size_bytes() {
    assert_eq!(format_size(100), "100 B");
    assert_eq!(format_size(0), "0 B");
}

#[test]
fn test_format_size_kb() {
    assert_eq!(format_size(1024), "1.0 KB");
    assert_eq!(format_size(2048), "2.0 KB");
    assert_eq!(format_size(1536), "1.5 KB");
}

#[test]
fn test_format_size_mb() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
    assert_eq!(format_size(10 * 1024 * 1024), "10.0 MB");
}

#[test]
fn test_format_size_gb() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.0 GB");
}

#[test]
fn test_format_params() {
    assert_eq!(format_params(100), "100");
    assert_eq!(format_params(1_000), "1.00K");
    assert_eq!(format_params(1_500), "1.50K");
    assert_eq!(format_params(1_000_000), "1.00M");
    assert_eq!(format_params(7_000_000_000), "7.00B");
}

#[test]
fn test_truncate_value() {
    assert_eq!(truncate_value("short", 10), "short");
    assert_eq!(
        truncate_value("this is a very long string", 10),
        "this is a ..."
    );
}

// ========================================================================
// Validate Path Tests
// ========================================================================

#[test]
fn test_validate_path_not_found() {
    let result = validate_path(Path::new("/nonexistent/model.apr"));
    assert!(result.is_err());
}

#[test]
fn test_validate_path_is_directory() {
    use tempfile::tempdir;
    let dir = tempdir().expect("create dir");
    let result = validate_path(dir.path());
    assert!(result.is_err());
}

#[test]
fn test_validate_path_valid() {
    use tempfile::NamedTempFile;
    let file = NamedTempFile::new().expect("create file");
    let result = validate_path(file.path());
    assert!(result.is_ok());
}

// ========================================================================
// Metadata Comparison Tests
// ========================================================================

#[test]
fn test_compare_metadata_identical() {
    use std::collections::BTreeMap;
    let mut m1 = BTreeMap::new();
    m1.insert("key1".to_string(), "value1".to_string());
    m1.insert("key2".to_string(), "value2".to_string());

    let m2 = m1.clone();
    let mut diffs = Vec::new();
    compare_metadata(&m1, &m2, &mut diffs);
    assert!(diffs.is_empty());
}

#[test]
fn test_compare_metadata_different_value() {
    use std::collections::BTreeMap;
    let mut m1 = BTreeMap::new();
    m1.insert("key1".to_string(), "value1".to_string());

    let mut m2 = BTreeMap::new();
    m2.insert("key1".to_string(), "value2".to_string());

    let mut diffs = Vec::new();
    compare_metadata(&m1, &m2, &mut diffs);
    assert_eq!(diffs.len(), 1);
    assert!(diffs[0].field.contains("key1"));
}

#[test]
fn test_compare_metadata_missing_key() {
    use std::collections::BTreeMap;
    let mut m1 = BTreeMap::new();
    m1.insert("key1".to_string(), "value1".to_string());
    m1.insert("key2".to_string(), "value2".to_string());

    let mut m2 = BTreeMap::new();
    m2.insert("key1".to_string(), "value1".to_string());

    let mut diffs = Vec::new();
    compare_metadata(&m1, &m2, &mut diffs);
    assert_eq!(diffs.len(), 1);
    assert!(diffs[0].field.contains("key2"));
    assert!(diffs[0].value2.contains("missing"));
}

#[test]
fn test_compare_metadata_extra_key() {
    use std::collections::BTreeMap;
    let mut m1 = BTreeMap::new();
    m1.insert("key1".to_string(), "value1".to_string());

    let mut m2 = BTreeMap::new();
    m2.insert("key1".to_string(), "value1".to_string());
    m2.insert("key2".to_string(), "value2".to_string());

    let mut diffs = Vec::new();
    compare_metadata(&m1, &m2, &mut diffs);
    assert_eq!(diffs.len(), 1);
    assert!(diffs[0].field.contains("key2"));
    assert!(diffs[0].value1.contains("missing"));
}

// ========================================================================
// Tensor Comparison Tests
// ========================================================================

#[test]
fn test_compare_tensors_identical() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        size_bytes: 800,
        stats: None,
    }];
    let t2 = t1.clone();

    let mut diffs = Vec::new();
    let options = DiffOptions::default();
    compare_tensors(&t1, &t2, &options, &mut diffs);
    assert!(diffs.is_empty());
}

#[test]
fn test_compare_tensors_different_count() {
    use crate::format::rosetta::TensorInfo;

    let t1 = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        size_bytes: 800,
        stats: None,
    }];
    let t2 = vec![
        TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        },
        TensorInfo {
            name: "bias".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: None,
        },
    ];

    let mut diffs = Vec::new();
    let options = DiffOptions::default();
    compare_tensors(&t1, &t2, &options, &mut diffs);

    // Should have count diff and missing tensor diff
    assert!(diffs.iter().any(|d| d.field == "tensor_count"));
    assert!(diffs.iter().any(|d| d.field.contains("bias")));
}

#[path = "diff_tests_tensor_cmp.rs"]
mod diff_tests_tensor_cmp;
#[path = "diff_tests_part_03.rs"]
mod diff_tests_part_03;
#[path = "diff_tests_part_04.rs"]
mod diff_tests_part_04;
