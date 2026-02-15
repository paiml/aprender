
    #[test]
    fn test_validate_paths_first_not_found() {
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(Path::new("/nonexistent/model1.apr"), file2.path());
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_second_not_found() {
        let file1 = NamedTempFile::new().expect("create file");
        let result = validate_paths(file1.path(), Path::new("/nonexistent/model2.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_first_is_directory() {
        let dir = tempdir().expect("create dir");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(dir.path(), file2.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_paths_valid() {
        let file1 = NamedTempFile::new().expect("create file");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(file1.path(), file2.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_file_not_found() {
        let file = NamedTempFile::new().expect("create file");
        let result = run(
            Path::new("/nonexistent/model.apr"),
            file.path(),
            false,
            false,
            None,
            10,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_files() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create file");

        // Write minimal data (less than header size)
        file1.write_all(b"short").expect("write");
        file2.write_all(b"short").expect("write");

        let result = run(
            file1.path(),
            file2.path(),
            false,
            false,
            None,
            10,
            false,
            false,
        );
        // Should fail because files are too small/invalid
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_diff_status_thresholds() {
        // Test with matching shapes
        let shape = &[10, 10];
        let elem_count = 100;

        // Identical: max_diff = 0.0
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0, shape, shape, 100, elem_count),
            TensorDiffStatus::Identical
        );
        // Nearly identical: max_diff < 0.001
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0005, shape, shape, 50, elem_count),
            TensorDiffStatus::NearlyIdentical
        );
        // Small diff: max_diff < 0.01
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.005, shape, shape, 10, elem_count),
            TensorDiffStatus::SmallDiff
        );
        // Medium diff: max_diff < 0.1
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.05, shape, shape, 5, elem_count),
            TensorDiffStatus::MediumDiff
        );
        // Large diff: max_diff < 1.0
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.5, shape, shape, 1, elem_count),
            TensorDiffStatus::LargeDiff
        );
        // Critical: max_diff >= 1.0
        assert_eq!(
            TensorDiffStatus::from_diff_info(1.5, shape, shape, 0, elem_count),
            TensorDiffStatus::Critical
        );
        // Incompatible shape mismatch (different element counts) is critical
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0, &[10, 10], &[5, 5], 25, 25),
            TensorDiffStatus::Critical
        );
        // Transposed shapes with identical values
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0, &[10, 20], &[20, 10], 200, 200),
            TensorDiffStatus::Transposed
        );
    }

    #[test]
    fn test_compute_tensor_diff_stats_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_tensor_diff_stats("test", &[4], &[4], &data, &data, false);
        assert_eq!(stats.status, TensorDiffStatus::Identical);
        assert_eq!(stats.max_diff, 0.0);
        assert_eq!(stats.identical_count, 4);
    }

    #[test]
    fn test_compute_tensor_diff_stats_small_diff() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![1.0001, 2.0001, 3.0001, 4.0001];
        let stats = compute_tensor_diff_stats("test", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.status, TensorDiffStatus::NearlyIdentical);
        assert!(stats.max_diff < 0.001);
    }

    #[test]
    fn test_normalize_tensor_name() {
        assert!(normalize_tensor_name("blk.0.attn_q.weight").contains("model.layers.0"));
        assert!(normalize_tensor_name("blk.0.attn_q.weight").contains("self_attn.q_proj"));
    }

    // ==================== TensorDiffStatus::from_diff_info - exhaustive branch coverage ====================

    #[test]
    fn test_from_diff_info_transposed_high_ident_ratio() {
        // Transposed shapes with >99% identical values => Transposed
        let status = TensorDiffStatus::from_diff_info(0.0, &[10, 20], &[20, 10], 199, 200);
        assert_eq!(status, TensorDiffStatus::Transposed);
    }

    #[test]
    fn test_from_diff_info_transposed_low_ident_ratio_small_max_diff() {
        // Transposed shapes, low ident ratio, max_diff < 0.1 => MediumDiff
        let status = TensorDiffStatus::from_diff_info(0.05, &[10, 20], &[20, 10], 10, 200);
        assert_eq!(status, TensorDiffStatus::MediumDiff);
    }

    #[test]
    fn test_from_diff_info_transposed_low_ident_ratio_large_max_diff() {
        // Transposed shapes, low ident ratio, max_diff >= 0.1 => falls through to value classification
        // max_diff=0.5 => LargeDiff (0.1 <= 0.5 < 1.0)
        let status = TensorDiffStatus::from_diff_info(0.5, &[10, 20], &[20, 10], 10, 200);
        assert_eq!(status, TensorDiffStatus::LargeDiff);
    }

    #[test]
    fn test_from_diff_info_transposed_critical_max_diff() {
        // Transposed shapes, low ident ratio, max_diff >= 1.0 => Critical
        let status = TensorDiffStatus::from_diff_info(2.0, &[10, 20], &[20, 10], 5, 200);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_incompatible_1d_vs_2d() {
        // Different dimensionality shapes => Critical
        let status = TensorDiffStatus::from_diff_info(0.0, &[100], &[10, 10], 100, 100);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_incompatible_3d_shapes() {
        // 3D shapes that don't match => Critical (is_transpose only for 2D)
        let status = TensorDiffStatus::from_diff_info(0.0, &[2, 3, 4], &[4, 3, 2], 24, 24);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_boundary_nearly_identical() {
        // max_diff just below 0.001 => NearlyIdentical
        let status = TensorDiffStatus::from_diff_info(0.000_999, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::NearlyIdentical);
    }

    #[test]
    fn test_from_diff_info_boundary_small_diff() {
        // max_diff exactly at 0.001 => SmallDiff (0.001 is NOT < 0.001)
        let status = TensorDiffStatus::from_diff_info(0.001, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::SmallDiff);
    }

    #[test]
    fn test_from_diff_info_boundary_medium_diff() {
        // max_diff exactly at 0.01 => MediumDiff (0.01 is NOT < 0.01)
        let status = TensorDiffStatus::from_diff_info(0.01, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::MediumDiff);
    }

    #[test]
    fn test_from_diff_info_boundary_large_diff() {
        // max_diff exactly at 0.1 => LargeDiff (0.1 is NOT < 0.1)
        let status = TensorDiffStatus::from_diff_info(0.1, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::LargeDiff);
    }

    #[test]
    fn test_from_diff_info_boundary_critical() {
        // max_diff exactly at 1.0 => Critical (1.0 is NOT < 1.0)
        let status = TensorDiffStatus::from_diff_info(1.0, &[10], &[10], 0, 10);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_same_1d_shape_identical() {
        // 1D shapes that match, max_diff = 0.0
        let status = TensorDiffStatus::from_diff_info(0.0, &[256], &[256], 256, 256);
        assert_eq!(status, TensorDiffStatus::Identical);
    }

    #[test]
    fn test_from_diff_info_transposed_exact_boundary_ident_ratio() {
        // Transposed shapes with exactly 99% identical (should NOT be Transposed - need >0.99)
        // 99/100 = 0.99, which is NOT > 0.99
        let status = TensorDiffStatus::from_diff_info(0.0, &[10, 10], &[10, 10], 99, 100);
        // Same shapes => not transposed, max_diff=0.0 => Identical
        assert_eq!(status, TensorDiffStatus::Identical);
    }

    #[test]
    fn test_from_diff_info_transposed_boundary_ident_ratio_below_threshold() {
        // Transposed shapes, ident ratio exactly 0.99 (not > 0.99), max_diff = 0.0
        // 99/100 = 0.99 which is NOT > 0.99 => goes to max_diff check
        // max_diff=0.0 < 0.1 => MediumDiff
        let status = TensorDiffStatus::from_diff_info(0.0, &[5, 20], &[20, 5], 99, 100);
        assert_eq!(status, TensorDiffStatus::MediumDiff);
    }

    // ==================== TensorDiffStatus::colored_string - all variants ====================

    #[test]
    fn test_colored_string_identical() {
        let s = TensorDiffStatus::Identical.colored_string();
        // colored_string returns a ColoredString; check the underlying text
        assert_eq!(s.to_string().contains("IDENTICAL"), true);
    }

    #[test]
    fn test_colored_string_nearly_identical() {
        let s = TensorDiffStatus::NearlyIdentical.colored_string();
        assert!(s.to_string().contains("IDENT"));
    }

    #[test]
    fn test_colored_string_small_diff() {
        let s = TensorDiffStatus::SmallDiff.colored_string();
        assert!(s.to_string().contains("SMALL"));
    }

    #[test]
    fn test_colored_string_medium_diff() {
        let s = TensorDiffStatus::MediumDiff.colored_string();
        assert!(s.to_string().contains("MEDIUM"));
    }

    #[test]
    fn test_colored_string_large_diff() {
        let s = TensorDiffStatus::LargeDiff.colored_string();
        assert!(s.to_string().contains("LARGE"));
    }

    #[test]
    fn test_colored_string_transposed() {
        let s = TensorDiffStatus::Transposed.colored_string();
        assert!(s.to_string().contains("TRANSPOSED"));
    }

    #[test]
    fn test_colored_string_critical() {
        let s = TensorDiffStatus::Critical.colored_string();
        assert!(s.to_string().contains("CRITICAL"));
    }

    // ==================== DiffResultJson::from(&DiffReport) ====================

    #[test]
    fn test_diff_result_json_from_empty_report() {
        let report = DiffReport {
            path1: "model_a.apr".to_string(),
            path2: "model_b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        let json = DiffResultJson::from(&report);
        assert_eq!(json.file1, "model_a.apr");
        assert_eq!(json.file2, "model_b.apr");
        assert_eq!(json.format1, "APR");
        assert_eq!(json.format2, "APR");
        assert!(json.identical);
        assert_eq!(json.difference_count, 0);
        assert!(json.differences.is_empty());
    }

    #[test]
    fn test_diff_result_json_from_report_with_diffs() {
        let report = DiffReport {
            path1: "a.gguf".to_string(),
            path2: "b.safetensors".to_string(),
            format1: "GGUF".to_string(),
            format2: "SafeTensors".to_string(),
            differences: vec![
                DiffEntry {
                    field: "tensor_count".to_string(),
                    value1: "100".to_string(),
                    value2: "200".to_string(),
                    category: DiffCategory::Tensor,
                },
                DiffEntry {
                    field: "format_version".to_string(),
                    value1: "v2".to_string(),
                    value2: "v3".to_string(),
                    category: DiffCategory::Format,
                },
            ],
            inspection1: None,
            inspection2: None,
        };
        let json = DiffResultJson::from(&report);
        assert_eq!(json.file1, "a.gguf");
        assert_eq!(json.file2, "b.safetensors");
        assert_eq!(json.format1, "GGUF");
        assert_eq!(json.format2, "SafeTensors");
        assert!(!json.identical);
        assert_eq!(json.difference_count, 2);
        assert_eq!(json.differences.len(), 2);
        assert_eq!(json.differences[0].field, "tensor_count");
        assert_eq!(json.differences[0].file1_value, "100");
        assert_eq!(json.differences[0].file2_value, "200");
        assert_eq!(json.differences[0].category, "tensor");
        assert_eq!(json.differences[1].field, "format_version");
        assert_eq!(json.differences[1].category, "format");
    }

    #[test]
    fn test_diff_result_json_serialization() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![DiffEntry {
                field: "file_size".to_string(),
                value1: "1024".to_string(),
                value2: "2048".to_string(),
                category: DiffCategory::Size,
            }],
            inspection1: None,
            inspection2: None,
        };
        let json_result = DiffResultJson::from(&report);
        let serialized = serde_json::to_string(&json_result).expect("serialize");
        assert!(serialized.contains("\"file1\":\"a.apr\""));
        assert!(serialized.contains("\"identical\":false"));
        assert!(serialized.contains("\"difference_count\":1"));
        assert!(serialized.contains("\"file_size\""));
    }

    // ==================== normalize_tensor_name - all replacement patterns ====================

    #[test]
    fn test_normalize_tensor_name_attn_k() {
        let result = normalize_tensor_name("blk.5.attn_k.weight");
        assert_eq!(result, "model.layers.5.self_attn.k_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_attn_v() {
        let result = normalize_tensor_name("blk.3.attn_v.weight");
        assert_eq!(result, "model.layers.3.self_attn.v_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_attn_output() {
        let result = normalize_tensor_name("blk.1.attn_output.weight");
        assert_eq!(result, "model.layers.1.self_attn.o_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_gate() {
        let result = normalize_tensor_name("blk.0.ffn_gate.weight");
        assert_eq!(result, "model.layers.0.mlp.gate_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_up() {
        let result = normalize_tensor_name("blk.2.ffn_up.weight");
        assert_eq!(result, "model.layers.2.mlp.up_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_down() {
        let result = normalize_tensor_name("blk.4.ffn_down.weight");
        assert_eq!(result, "model.layers.4.mlp.down_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_attn_norm() {
        let result = normalize_tensor_name("blk.0.attn_norm.weight");
        assert_eq!(result, "model.layers.0.input_layernorm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_norm() {
        let result = normalize_tensor_name("blk.0.ffn_norm.weight");
        assert_eq!(result, "model.layers.0.post_attention_layernorm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_no_changes() {
        // Already in HF naming convention - should be unchanged
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(normalize_tensor_name(name), name);
    }
