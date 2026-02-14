    use super::*;

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
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(2048), "2.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.00 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
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

    fn make_report(
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

    #[test]
    fn test_normalize_dtype_string_lowercase() {
        assert_eq!(normalize_dtype("f32"), "F32");
        assert_eq!(normalize_dtype("f16"), "F16");
        assert_eq!(normalize_dtype("q4_0"), "Q4_0");
        assert_eq!(normalize_dtype("q4_1"), "Q4_1");
        assert_eq!(normalize_dtype("q5_0"), "Q5_0");
        assert_eq!(normalize_dtype("q5_1"), "Q5_1");
        assert_eq!(normalize_dtype("q8_0"), "Q8_0");
        assert_eq!(normalize_dtype("q8_1"), "Q8_1");
        assert_eq!(normalize_dtype("q2_k"), "Q2_K");
        assert_eq!(normalize_dtype("q3_k"), "Q3_K");
        assert_eq!(normalize_dtype("q4_k"), "Q4_K");
        assert_eq!(normalize_dtype("q5_k"), "Q5_K");
        assert_eq!(normalize_dtype("q6_k"), "Q6_K");
        assert_eq!(normalize_dtype("q8_k"), "Q8_K");
        assert_eq!(normalize_dtype("iq2_xxs"), "IQ2_XXS");
        assert_eq!(normalize_dtype("iq2_xs"), "IQ2_XS");
        assert_eq!(normalize_dtype("iq3_xxs"), "IQ3_XXS");
        assert_eq!(normalize_dtype("iq1_s"), "IQ1_S");
    }

    #[test]
    fn test_normalize_dtype_string_uppercase() {
        assert_eq!(normalize_dtype("F32"), "F32");
        assert_eq!(normalize_dtype("F16"), "F16");
        assert_eq!(normalize_dtype("Q4_0"), "Q4_0");
        assert_eq!(normalize_dtype("Q4_1"), "Q4_1");
        assert_eq!(normalize_dtype("Q5_0"), "Q5_0");
        assert_eq!(normalize_dtype("Q5_1"), "Q5_1");
        assert_eq!(normalize_dtype("Q8_0"), "Q8_0");
        assert_eq!(normalize_dtype("Q8_1"), "Q8_1");
        assert_eq!(normalize_dtype("Q2_K"), "Q2_K");
        assert_eq!(normalize_dtype("Q3_K"), "Q3_K");
        assert_eq!(normalize_dtype("Q4_K"), "Q4_K");
        assert_eq!(normalize_dtype("Q5_K"), "Q5_K");
        assert_eq!(normalize_dtype("Q6_K"), "Q6_K");
        assert_eq!(normalize_dtype("Q8_K"), "Q8_K");
        assert_eq!(normalize_dtype("IQ2_XXS"), "IQ2_XXS");
        assert_eq!(normalize_dtype("IQ2_XS"), "IQ2_XS");
        assert_eq!(normalize_dtype("IQ3_XXS"), "IQ3_XXS");
        assert_eq!(normalize_dtype("IQ1_S"), "IQ1_S");
    }

    #[test]
    fn test_normalize_dtype_short_aliases() {
        // Short aliases without underscore (q2k, Q2K, etc.)
        assert_eq!(normalize_dtype("q2k"), "Q2_K");
        assert_eq!(normalize_dtype("Q2K"), "Q2_K");
        assert_eq!(normalize_dtype("q3k"), "Q3_K");
        assert_eq!(normalize_dtype("Q3K"), "Q3_K");
        assert_eq!(normalize_dtype("q4k"), "Q4_K");
        assert_eq!(normalize_dtype("Q4K"), "Q4_K");
        assert_eq!(normalize_dtype("q5k"), "Q5_K");
        assert_eq!(normalize_dtype("Q5K"), "Q5_K");
        assert_eq!(normalize_dtype("q6k"), "Q6_K");
        assert_eq!(normalize_dtype("Q6K"), "Q6_K");
        assert_eq!(normalize_dtype("q8k"), "Q8_K");
        assert_eq!(normalize_dtype("Q8K"), "Q8_K");
    }

    #[test]
    fn test_normalize_dtype_bf16() {
        assert_eq!(normalize_dtype("bf16"), "BF16");
        assert_eq!(normalize_dtype("BF16"), "BF16");
    }

    #[test]
    fn test_normalize_dtype_unknown_fallback() {
        // Catch-all: unknown types get uppercased
        assert_eq!(normalize_dtype("custom_type"), "CUSTOM_TYPE");
        assert_eq!(normalize_dtype("fp8"), "FP8");
        assert_eq!(normalize_dtype("int4"), "INT4");
        assert_eq!(normalize_dtype("26"), "26"); // Numeric code not in map
    }

    // ====================================================================
    // Coverage: is_compatible_quant exhaustive branch tests
    // ====================================================================

    #[test]
    fn test_is_compatible_quant_same_after_normalization() {
        // Same type after normalization returns true
        assert!(is_compatible_quant("f32", "F32"));
        assert!(is_compatible_quant("0", "F32"));
        assert!(is_compatible_quant("q4_k", "Q4_K"));
        assert!(is_compatible_quant("12", "Q4_K"));
        assert!(is_compatible_quant("q8_0", "Q8_0"));
        assert!(is_compatible_quant("8", "Q8_0"));
    }

    #[test]
    fn test_is_compatible_quant_q5_q6_compatible() {
        // Q5 <-> Q6 compatibility (common import path)
        assert!(is_compatible_quant("Q5_0", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q5_0"));
        assert!(is_compatible_quant("Q5_K", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q5_K"));
        assert!(is_compatible_quant("Q5_1", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q5_1"));
    }

    #[test]
    fn test_is_compatible_quant_q4_variants() {
        // Q4 <-> Q4 variants compatible
        assert!(is_compatible_quant("Q4_0", "Q4_K"));
        assert!(is_compatible_quant("Q4_K", "Q4_0"));
        assert!(is_compatible_quant("Q4_1", "Q4_K"));
        assert!(is_compatible_quant("Q4_K", "Q4_1"));
        assert!(is_compatible_quant("Q4_0", "Q4_1"));
    }

    #[test]
    fn test_is_compatible_quant_q8_q6_compatible() {
        // Q8 <-> Q6 compatibility (downgrade)
        assert!(is_compatible_quant("Q8_0", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q8_0"));
        assert!(is_compatible_quant("Q8_K", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q8_K"));
    }

    #[test]
    fn test_is_compatible_quant_incompatible_pairs() {
        // Truly incompatible pairs
        assert!(!is_compatible_quant("F32", "Q4_K"));
        assert!(!is_compatible_quant("Q4_K", "F32"));
        assert!(!is_compatible_quant("Q2_K", "Q8_0"));
        assert!(!is_compatible_quant("Q8_0", "Q2_K"));
        assert!(!is_compatible_quant("F16", "Q4_0"));
        assert!(!is_compatible_quant("BF16", "Q6_K"));
        assert!(!is_compatible_quant("Q3_K", "Q8_0"));
        assert!(!is_compatible_quant("F32", "F16"));
        assert!(!is_compatible_quant("IQ2_XXS", "Q4_K"));
    }

    #[test]
    fn test_is_compatible_quant_with_numeric_codes() {
        // Using numeric GGUF codes should also work through normalization
        assert!(is_compatible_quant("12", "Q4_0")); // 12 = Q4_K, compatible with Q4_0
        assert!(is_compatible_quant("6", "14")); // Q5_0 <-> Q6_K compatible
        assert!(!is_compatible_quant("0", "12")); // F32 <-> Q4_K incompatible
    }

    // ====================================================================
    // Coverage: map_gguf_to_apr_name all branches
    // ====================================================================

    #[test]
    fn test_map_gguf_to_apr_name_attn_q_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.attn_q.weight");
        assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_q_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.3.attn_q.bias");
        assert_eq!(name, "model.layers.3.self_attn.q_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_k_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.1.attn_k.weight");
        assert_eq!(name, "model.layers.1.self_attn.k_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_k_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.7.attn_k.bias");
        assert_eq!(name, "model.layers.7.self_attn.k_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_v_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.2.attn_v.weight");
        assert_eq!(name, "model.layers.2.self_attn.v_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_v_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.5.attn_v.bias");
        assert_eq!(name, "model.layers.5.self_attn.v_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_output_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.4.attn_output.weight");
        assert_eq!(name, "model.layers.4.self_attn.o_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_output_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.attn_output.bias");
        assert_eq!(name, "model.layers.0.self_attn.o_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_norm() {
        let (name, mapped) = map_gguf_to_apr_name("blk.6.attn_norm.weight");
        assert_eq!(name, "model.layers.6.input_layernorm.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_gate() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.ffn_gate.weight");
        assert_eq!(name, "model.layers.0.mlp.gate_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_up() {
        let (name, mapped) = map_gguf_to_apr_name("blk.10.ffn_up.weight");
        assert_eq!(name, "model.layers.10.mlp.up_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_down() {
        let (name, mapped) = map_gguf_to_apr_name("blk.31.ffn_down.weight");
        assert_eq!(name, "model.layers.31.mlp.down_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_norm() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.ffn_norm.weight");
        assert_eq!(name, "model.layers.0.post_attention_layernorm.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_unknown_layer_suffix() {
        // Unknown suffix within blk.N.* returns unchanged
        let (name, mapped) = map_gguf_to_apr_name("blk.0.unknown_suffix.weight");
        assert_eq!(name, "blk.0.unknown_suffix.weight");
        assert!(!mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_token_embd() {
        let (name, mapped) = map_gguf_to_apr_name("token_embd.weight");
        assert_eq!(name, "model.embed_tokens.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_output_weight() {
        let (name, mapped) = map_gguf_to_apr_name("output.weight");
        assert_eq!(name, "lm_head.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_output_norm() {
        let (name, mapped) = map_gguf_to_apr_name("output_norm.weight");
        assert_eq!(name, "model.norm.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_unknown_non_layer() {
        // Unknown non-layer tensor returns unchanged
        let (name, mapped) = map_gguf_to_apr_name("some_other_tensor");
        assert_eq!(name, "some_other_tensor");
        assert!(!mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_already_apr_style() {
        // APR-style names pass through unchanged
        let (name, mapped) = map_gguf_to_apr_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
        assert!(!mapped);
    }

    // ====================================================================
    // Coverage: build_cross_format_map tests
    // ====================================================================

    #[test]
    fn test_build_cross_format_map_gguf_names() {
        use crate::format::rosetta::TensorInfo;

        let tensors = vec![
            TensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dtype: "Q4_K".to_string(),
                shape: vec![4096, 4096],
                size_bytes: 1000,
                stats: None,
            },
            TensorInfo {
                name: "token_embd.weight".to_string(),
                dtype: "F16".to_string(),
                shape: vec![32000, 4096],
                size_bytes: 2000,
                stats: None,
            },
        ];

        let map = build_cross_format_map(&tensors);

        // Original GGUF names should be present
        assert!(map.contains_key("blk.0.attn_q.weight"));
        assert!(map.contains_key("token_embd.weight"));

        // Mapped APR names should also be present
        assert!(map.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(map.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    fn test_build_cross_format_map_hf_names() {
        use crate::format::rosetta::TensorInfo;

        // HF/APR names that won't be mapped (no blk. prefix, not in non-layer map)
        let tensors = vec![TensorInfo {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![4096, 4096],
            size_bytes: 1000,
            stats: None,
        }];

        let map = build_cross_format_map(&tensors);

        // Original name present
        assert!(map.contains_key("model.layers.0.self_attn.q_proj.weight"));
        // Only one entry (no mapping was applied)
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_build_cross_format_map_empty() {
        let tensors: Vec<crate::format::rosetta::TensorInfo> = vec![];
        let map = build_cross_format_map(&tensors);
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_cross_format_map_unknown_suffix() {
        use crate::format::rosetta::TensorInfo;

        // Unknown suffix within blk.N.* -- no mapping added
        let tensors = vec![TensorInfo {
            name: "blk.0.custom_thing.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            size_bytes: 400,
            stats: None,
        }];

        let map = build_cross_format_map(&tensors);

        // Only the original name is present (no mapping)
        assert!(map.contains_key("blk.0.custom_thing.weight"));
        assert_eq!(map.len(), 1);
    }

    // ====================================================================
    // Coverage: diff_inspections public API tests
    // ====================================================================

    #[test]
    fn test_diff_inspections_identical() {
        let r = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
        let report = diff_inspections(&r, &r, "model_a.apr", "model_b.apr", DiffOptions::default());
        assert!(report.is_identical());
        assert_eq!(report.path1, "model_a.apr");
        assert_eq!(report.path2, "model_b.apr");
        assert_eq!(report.format1, "APR");
        assert_eq!(report.format2, "APR");
        assert!(report.inspection1.is_none());
        assert!(report.inspection2.is_none());
    }

    #[test]
    fn test_diff_inspections_different_formats() {
        let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), Some("Q4_K"));
        let r2 = make_report(FormatType::Gguf, 2000, 200, Some("qwen2"), Some("Q8_0"));
        let report = diff_inspections(&r1, &r2, "a.apr", "b.gguf", DiffOptions::default());
        assert!(!report.is_identical());
        assert!(!report.same_format());
        assert!(report.differences.iter().any(|d| d.field == "format"));
        assert!(report.differences.iter().any(|d| d.field == "file_size"));
        assert!(report.differences.iter().any(|d| d.field == "total_params"));
        assert!(report.differences.iter().any(|d| d.field == "architecture"));
        assert!(report.differences.iter().any(|d| d.field == "quantization"));
    }

    #[test]
    fn test_diff_inspections_with_tensors() {
        use crate::format::rosetta::TensorInfo;

        let mut r1 = make_report(FormatType::Apr, 1000, 100, None, None);
        r1.tensors = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];

        let mut r2 = make_report(FormatType::Apr, 1000, 100, None, None);
        r2.tensors = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![20, 10], // Transposed -- compatible
            size_bytes: 800,
            stats: None,
        }];

        let report = diff_inspections(&r1, &r2, "a.apr", "b.apr", DiffOptions::default());
        // Transposed shapes are compatible, so no shape diff
        assert!(report.is_identical());
    }

    #[test]
    fn test_diff_inspections_no_metadata() {
        use std::collections::BTreeMap;

        let mut r1 = make_report(FormatType::Apr, 1000, 100, None, None);
        r1.metadata = {
            let mut m = BTreeMap::new();
            m.insert("key".to_string(), "val1".to_string());
            m
        };
        let mut r2 = make_report(FormatType::Apr, 1000, 100, None, None);
        r2.metadata = {
            let mut m = BTreeMap::new();
            m.insert("key".to_string(), "val2".to_string());
            m
        };

        // With metadata comparison disabled, no diff
        let report = diff_inspections(
            &r1,
            &r2,
            "a.apr",
            "b.apr",
            DiffOptions::new().without_metadata(),
        );
        assert!(report.is_identical());

        // With metadata comparison enabled, diff present
        let report2 = diff_inspections(
            &r1,
            &r2,
            "a.apr",
            "b.apr",
            DiffOptions::new().with_metadata(),
        );
        assert!(!report2.is_identical());
    }

    // ====================================================================
    // Coverage: compare_tensor_stats None->Some branch
    // ====================================================================

    #[test]
    fn test_compare_tensor_stats_none_some() {
        use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
        let mut diffs = Vec::new();
        let t1 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: None,
        };
        let t2 = RTI {
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
        compare_tensor_stats(&t1, &t2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("stats"));
        assert_eq!(diffs[0].value1, "(none)");
        assert_eq!(diffs[0].value2, "present");
    }

    // ====================================================================
    // Coverage: compare_tensor_stats std differs alone
    // ====================================================================

    #[test]
    fn test_compare_tensor_stats_std_differs_only() {
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
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.9,
            }),
        };
        compare_tensor_stats(&t1, &t2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("std"));
    }

    // ====================================================================
    // Coverage: cross-format tensor comparison with GGUF name mapping
    // ====================================================================

    #[test]
    fn test_compare_tensors_cross_format_gguf_to_apr() {
        use crate::format::rosetta::TensorInfo;

        // Model 1 uses GGUF naming
        let t1 = vec![
            TensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dtype: "Q4_K".to_string(),
                shape: vec![4096, 4096],
                size_bytes: 1000,
                stats: None,
            },
            TensorInfo {
                name: "token_embd.weight".to_string(),
                dtype: "F16".to_string(),
                shape: vec![32000, 4096],
                size_bytes: 2000,
                stats: None,
            },
        ];

        // Model 2 uses APR/HF naming with same shapes
        let t2 = vec![
            TensorInfo {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                dtype: "Q4_K".to_string(),
                shape: vec![4096, 4096],
                size_bytes: 1000,
                stats: None,
            },
            TensorInfo {
                name: "model.embed_tokens.weight".to_string(),
                dtype: "F16".to_string(),
                shape: vec![32000, 4096],
                size_bytes: 2000,
                stats: None,
            },
        ];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        // Cross-format mapping should make these match
        assert!(
            diffs.is_empty(),
            "Expected no diffs for cross-format name mapping, got: {diffs:?}"
        );
    }

    #[test]
    fn test_compare_tensors_transposed_shapes_compatible() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100, 200],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![200, 100], // Transposed
            size_bytes: 800,
            stats: None,
        }];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        // Transposed 2D shapes are considered compatible
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compare_tensors_compatible_quant_no_diff() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "Q5_0".to_string(),
            shape: vec![10, 20],
            size_bytes: 400,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "Q6_K".to_string(),
            shape: vec![10, 20],
            size_bytes: 400,
            stats: None,
        }];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        // Q5_0 and Q6_K are considered compatible
        assert!(
            diffs.is_empty(),
            "Expected no dtype diff for compatible quants Q5_0 and Q6_K, got: {diffs:?}"
        );
    }

    #[test]
    fn test_compare_tensors_only_in_model1() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "unique_tensor".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: None,
        }];
        let t2: Vec<TensorInfo> = vec![];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        assert!(diffs
            .iter()
            .any(|d| d.field.contains("unique_tensor") && d.value2 == "(missing)"));
    }

    #[test]
    fn test_compare_tensors_only_in_model2() {
        use crate::format::rosetta::TensorInfo;

        let t1: Vec<TensorInfo> = vec![];
        let t2 = vec![TensorInfo {
            name: "extra_tensor".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: None,
        }];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        assert!(diffs
            .iter()
            .any(|d| d.field.contains("extra_tensor") && d.value1 == "(missing)"));
    }
