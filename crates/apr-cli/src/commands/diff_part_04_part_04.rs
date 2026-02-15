
    #[test]
    fn test_print_tensor_diff_row_medium_diff() {
        let stats = TensorValueStats {
            name: "ffn.weight".to_string(),
            shape_a: vec![32, 32],
            shape_b: vec![32, 32],
            element_count: 1024,
            mean_diff: 0.03,
            max_diff: 0.08,
            rmse: 0.04,
            cosine_similarity: 0.998,
            identical_count: 100,
            small_diff_count: 200,
            medium_diff_count: 500,
            large_diff_count: 224,
            status: TensorDiffStatus::MediumDiff,
        };
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_large_diff() {
        let stats = TensorValueStats {
            name: "lm_head.weight".to_string(),
            shape_a: vec![50, 50],
            shape_b: vec![50, 50],
            element_count: 2500,
            mean_diff: 0.3,
            max_diff: 0.9,
            rmse: 0.4,
            cosine_similarity: 0.95,
            identical_count: 0,
            small_diff_count: 0,
            medium_diff_count: 500,
            large_diff_count: 2000,
            status: TensorDiffStatus::LargeDiff,
        };
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_small_diff() {
        let stats = TensorValueStats {
            name: "embed.weight".to_string(),
            shape_a: vec![16, 16],
            shape_b: vec![16, 16],
            element_count: 256,
            mean_diff: 0.003,
            max_diff: 0.008,
            rmse: 0.004,
            cosine_similarity: 0.9999,
            identical_count: 50,
            small_diff_count: 150,
            medium_diff_count: 56,
            large_diff_count: 0,
            status: TensorDiffStatus::SmallDiff,
        };
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_long_name_truncation() {
        let long_name = "model.layers.99.self_attn.q_proj.weight.extra.suffix.that.is.very.long";
        let stats = TensorValueStats {
            name: long_name.to_string(),
            shape_a: vec![4],
            shape_b: vec![4],
            element_count: 4,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 4,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Identical,
        };
        // Exercises truncate_str for name > 40 chars
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_cosine_similarity_ranges() {
        // Test cosine similarity coloring: > 0.9999
        let make_stats = |cos: f32| TensorValueStats {
            name: "t".to_string(),
            shape_a: vec![4],
            shape_b: vec![4],
            element_count: 4,
            mean_diff: 0.01,
            max_diff: 0.02,
            rmse: 0.01,
            cosine_similarity: cos,
            identical_count: 0,
            small_diff_count: 4,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::SmallDiff,
        };
        // > 0.9999 (green)
        print_tensor_diff_row(&make_stats(0.99999));
        // > 0.999 (blue)
        print_tensor_diff_row(&make_stats(0.9995));
        // > 0.99 (yellow)
        print_tensor_diff_row(&make_stats(0.995));
        // <= 0.99 (red)
        print_tensor_diff_row(&make_stats(0.5));
    }

    // ==================== DiffEntryJson struct ====================

    #[test]
    fn test_diff_entry_json_serialization() {
        let entry = DiffEntryJson {
            field: "tensor_count".to_string(),
            file1_value: "100".to_string(),
            file2_value: "200".to_string(),
            category: "tensor".to_string(),
            diff_type: "data".to_string(),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("\"field\":\"tensor_count\""));
        assert!(json.contains("\"file1_value\":\"100\""));
        assert!(json.contains("\"file2_value\":\"200\""));
        assert!(json.contains("\"category\":\"tensor\""));
        assert!(json.contains("\"type\":\"data\""));
    }

    // ==================== DiffResultJson category mapping ====================

    #[test]
    fn test_diff_result_json_all_categories() {
        let report = DiffReport {
            path1: "a".to_string(),
            path2: "b".to_string(),
            format1: "APR".to_string(),
            format2: "GGUF".to_string(),
            differences: vec![
                DiffEntry {
                    field: "f1".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Format,
                },
                DiffEntry {
                    field: "f2".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Metadata,
                },
                DiffEntry {
                    field: "f3".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Tensor,
                },
                DiffEntry {
                    field: "f4".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Quantization,
                },
                DiffEntry {
                    field: "f5".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Size,
                },
            ],
            inspection1: None,
            inspection2: None,
        };
        let json = DiffResultJson::from(&report);
        assert_eq!(json.difference_count, 5);
        assert_eq!(json.differences[0].category, "format");
        assert_eq!(json.differences[1].category, "metadata");
        assert_eq!(json.differences[2].category, "tensor");
        assert_eq!(json.differences[3].category, "quantization");
        assert_eq!(json.differences[4].category, "size");
    }

    // ==================== Integration: compute_tensor_diff_stats + from_diff_info ====================

    #[test]
    fn test_compute_stats_transposed_identical_values() {
        // Simulate a tensor that's transposed but values happen to match linearly
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let stats = compute_tensor_diff_stats("t", &[2, 3], &[3, 2], &data, &data, false);
        // All 6 elements identical linearly => ident_ratio = 1.0 > 0.99
        assert_eq!(stats.status, TensorDiffStatus::Transposed);
        assert_eq!(stats.identical_count, 6);
    }

    #[test]
    fn test_compute_stats_small_diff_boundary() {
        // max_diff exactly at SmallDiff boundary
        let data_a = vec![1.0, 2.0];
        let data_b = vec![1.001, 2.0]; // max_diff = 0.001 exactly
        let stats = compute_tensor_diff_stats("boundary", &[2], &[2], &data_a, &data_b, false);
        // 0.001 is NOT < 0.001, so SmallDiff
        assert_eq!(stats.status, TensorDiffStatus::SmallDiff);
    }

    #[test]
    fn test_compute_stats_one_element() {
        let stats = compute_tensor_diff_stats("single", &[1], &[1], &[42.0], &[42.0], false);
        assert_eq!(stats.status, TensorDiffStatus::Identical);
        assert_eq!(stats.element_count, 1);
        assert_eq!(stats.identical_count, 1);
        assert!((stats.cosine_similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_one_element_a_only() {
        // data_b is empty => element_count = min(1, 0) = 0 => early return Critical
        let stats = compute_tensor_diff_stats("asym", &[1], &[0], &[1.0], &[], false);
        assert_eq!(stats.status, TensorDiffStatus::Critical);
        assert_eq!(stats.element_count, 0);
    }
