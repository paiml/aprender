
    #[test]
    fn test_normalize_tensor_name_empty() {
        assert_eq!(normalize_tensor_name(""), "");
    }

    #[test]
    fn test_normalize_tensor_name_no_prefix() {
        // No blk. prefix but has the dot-delimited pattern
        let result = normalize_tensor_name("layer.attn_q.bias");
        assert_eq!(result, "layer.self_attn.q_proj.bias");
    }

    // ==================== truncate_path ====================

    #[test]
    fn test_truncate_path_short() {
        assert_eq!(truncate_path("short.apr", 20), "short.apr");
    }

    #[test]
    fn test_truncate_path_exact_length() {
        let path = "abcdefghij"; // 10 chars
        assert_eq!(truncate_path(path, 10), "abcdefghij");
    }

    #[test]
    fn test_truncate_path_long() {
        let path = "/very/long/path/to/some/model/file.apr";
        let result = truncate_path(path, 20);
        assert!(result.starts_with("..."));
        assert_eq!(result.len(), 20);
        assert!(result.ends_with("file.apr"));
    }

    #[test]
    fn test_truncate_path_one_over() {
        let path = "abcdefghijk"; // 11 chars
        let result = truncate_path(path, 10);
        assert!(result.starts_with("..."));
        assert_eq!(result.len(), 10);
    }

    // ==================== truncate_str ====================

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact_length() {
        assert_eq!(truncate_str("1234567890", 10), "1234567890");
    }

    #[test]
    fn test_truncate_str_long() {
        let result = truncate_str("this is a very long string", 10);
        assert_eq!(result, "this is...");
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_truncate_str_one_over() {
        let result = truncate_str("12345678901", 10);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 10);
    }

    // ==================== compute_tensor_diff_stats ====================

    #[test]
    fn test_compute_tensor_diff_stats_empty_data() {
        let stats = compute_tensor_diff_stats("empty", &[0], &[0], &[], &[], false);
        assert_eq!(stats.status, TensorDiffStatus::Critical);
        assert_eq!(stats.element_count, 0);
        assert_eq!(stats.mean_diff, 0.0);
        assert_eq!(stats.max_diff, 0.0);
        assert_eq!(stats.rmse, 0.0);
        assert_eq!(stats.cosine_similarity, 0.0);
        assert_eq!(stats.name, "empty");
    }

    #[test]
    fn test_compute_tensor_diff_stats_large_diff() {
        let data_a = vec![0.0, 0.0, 0.0, 0.0];
        let data_b = vec![10.0, 10.0, 10.0, 10.0];
        let stats = compute_tensor_diff_stats("large", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.status, TensorDiffStatus::Critical);
        assert_eq!(stats.max_diff, 10.0);
        assert_eq!(stats.large_diff_count, 4);
    }

    #[test]
    fn test_compute_tensor_diff_stats_medium_diff() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![1.05, 2.05, 3.05, 4.05];
        let stats = compute_tensor_diff_stats("med", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.status, TensorDiffStatus::MediumDiff);
        assert!(stats.max_diff > 0.01);
        assert!(stats.max_diff < 0.1);
    }

    #[test]
    fn test_compute_tensor_diff_stats_with_nan() {
        let data_a = vec![1.0, f32::NAN, 3.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats = compute_tensor_diff_stats("nan_test", &[3], &[3], &data_a, &data_b, false);
        // NaN values should be counted as large_diff
        assert!(stats.large_diff_count >= 1);
        // Non-NaN elements should still be compared
        assert!(stats.identical_count >= 2);
    }

    #[test]
    fn test_compute_tensor_diff_stats_with_inf() {
        let data_a = vec![1.0, f32::INFINITY, 3.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats = compute_tensor_diff_stats("inf_test", &[3], &[3], &data_a, &data_b, false);
        // Inf values should be counted as large_diff
        assert!(stats.large_diff_count >= 1);
    }

    #[test]
    fn test_compute_tensor_diff_stats_with_neg_inf() {
        let data_a = vec![f32::NEG_INFINITY, 2.0];
        let data_b = vec![1.0, 2.0];
        let stats = compute_tensor_diff_stats("neg_inf", &[2], &[2], &data_a, &data_b, false);
        assert!(stats.large_diff_count >= 1);
        assert_eq!(stats.identical_count, 1);
    }

    #[test]
    fn test_compute_tensor_diff_stats_both_nan() {
        let data_a = vec![f32::NAN, f32::NAN];
        let data_b = vec![f32::NAN, f32::NAN];
        let stats = compute_tensor_diff_stats("both_nan", &[2], &[2], &data_a, &data_b, false);
        // Both NaN => large_diff (NaN skipped in stats)
        assert_eq!(stats.large_diff_count, 2);
        assert_eq!(stats.identical_count, 0);
    }

    #[test]
    fn test_compute_tensor_diff_stats_all_zeros() {
        let data = vec![0.0, 0.0, 0.0, 0.0];
        let stats = compute_tensor_diff_stats("zeros", &[4], &[4], &data, &data, false);
        assert_eq!(stats.status, TensorDiffStatus::Identical);
        assert_eq!(stats.identical_count, 4);
        // Cosine similarity of zero vectors is 0.0 (division by zero guard)
        assert_eq!(stats.cosine_similarity, 0.0);
    }

    #[test]
    fn test_compute_tensor_diff_stats_cosine_similarity_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_tensor_diff_stats("cos", &[4], &[4], &data, &data, false);
        // Identical vectors => cosine_similarity = 1.0
        assert!((stats.cosine_similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_tensor_diff_stats_cosine_similarity_opposite() {
        let data_a = vec![1.0, 2.0, 3.0];
        let data_b = vec![-1.0, -2.0, -3.0];
        let stats = compute_tensor_diff_stats("opposite", &[3], &[3], &data_a, &data_b, false);
        // Opposite vectors => cosine_similarity = -1.0
        assert!((stats.cosine_similarity + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_tensor_diff_stats_different_length_data() {
        // data_b is shorter than data_a; element_count = min(len_a, len_b)
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats = compute_tensor_diff_stats("diff_len", &[5], &[3], &data_a, &data_b, false);
        assert_eq!(stats.element_count, 3);
    }

    #[test]
    fn test_compute_tensor_diff_stats_transpose_aware() {
        // A is [2, 3] with data [1,2,3,4,5,6]
        // B is [3, 2] with data = transposed version
        // A[0,0]=1, A[0,1]=2, A[0,2]=3, A[1,0]=4, A[1,1]=5, A[1,2]=6
        // B transposed from A: B[0,0]=1, B[0,1]=4, B[1,0]=2, B[1,1]=5, B[2,0]=3, B[2,1]=6
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B in row-major for shape [3, 2]: [[1,4],[2,5],[3,6]]
        let data_b = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let stats =
            compute_tensor_diff_stats("transpose", &[2, 3], &[3, 2], &data_a, &data_b, true);
        // With transpose_aware=true, it remaps indices
        // The function compares A[i] with B[remapped(i)]
        // Let's just verify it runs and produces a result
        assert_eq!(stats.element_count, 6);
        assert_eq!(stats.name, "transpose");
    }

    #[test]
    fn test_compute_tensor_diff_stats_transpose_aware_false() {
        // Same data but transpose_aware=false: linear comparison, shapes differ
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_b = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let stats =
            compute_tensor_diff_stats("no_transpose", &[2, 3], &[3, 2], &data_a, &data_b, false);
        // Without transpose_aware, shapes are transposed so from_diff_info detects Transposed
        // if ident ratio > 0.99. Let's check: 2 of 6 identical (indices 0 and 4) => 33%
        // So it won't be Transposed status. max_diff = |4-2| = 2.0 => Critical path likely
        assert_eq!(stats.element_count, 6);
    }

    #[test]
    fn test_compute_tensor_diff_stats_diff_buckets() {
        // Craft data to hit all diff buckets: identical, small (<0.001), medium (<0.01), large (>=0.01)
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![1.0, 2.0005, 3.005, 4.5];
        let stats = compute_tensor_diff_stats("buckets", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.identical_count, 1); // 1.0 == 1.0
        assert_eq!(stats.small_diff_count, 1); // |2.0-2.0005| = 0.0005 < 0.001
        assert_eq!(stats.medium_diff_count, 1); // |3.0-3.005| = 0.005 in [0.001, 0.01)
        assert_eq!(stats.large_diff_count, 1); // |4.0-4.5| = 0.5 >= 0.01
    }

    #[test]
    fn test_compute_tensor_diff_stats_rmse_and_mean() {
        let data_a = vec![0.0, 0.0];
        let data_b = vec![1.0, 1.0];
        let stats = compute_tensor_diff_stats("rmse", &[2], &[2], &data_a, &data_b, false);
        assert!((stats.mean_diff - 1.0).abs() < 1e-5);
        assert!((stats.rmse - 1.0).abs() < 1e-5);
        assert!((stats.max_diff - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_tensor_diff_stats_shape_stored() {
        let stats = compute_tensor_diff_stats("shapes", &[2, 3, 4], &[5, 6], &[1.0], &[2.0], false);
        assert_eq!(stats.shape_a, vec![2, 3, 4]);
        assert_eq!(stats.shape_b, vec![5, 6]);
    }

    // ==================== validate_paths - additional coverage ====================

    #[test]
    fn test_validate_paths_second_is_directory() {
        let file1 = NamedTempFile::new().expect("create file");
        let dir = tempdir().expect("create dir");
        let result = validate_paths(file1.path(), dir.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_paths_both_nonexistent() {
        let result = validate_paths(
            Path::new("/nonexistent/a.apr"),
            Path::new("/nonexistent/b.apr"),
        );
        assert!(result.is_err());
        // First path checked first
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, Path::new("/nonexistent/a.apr"));
            }
            _ => panic!("Expected FileNotFound error for first path"),
        }
    }

    // ==================== TensorValueStats struct ====================

    #[test]
    fn test_tensor_value_stats_construction() {
        let stats = TensorValueStats {
            name: "test_tensor".to_string(),
            shape_a: vec![10, 20],
            shape_b: vec![10, 20],
            element_count: 200,
            mean_diff: 0.001,
            max_diff: 0.005,
            rmse: 0.002,
            cosine_similarity: 0.999,
            identical_count: 150,
            small_diff_count: 30,
            medium_diff_count: 15,
            large_diff_count: 5,
            status: TensorDiffStatus::SmallDiff,
        };
        assert_eq!(stats.name, "test_tensor");
        assert_eq!(stats.element_count, 200);
        assert_eq!(stats.status, TensorDiffStatus::SmallDiff);
    }

    #[test]
    fn test_tensor_value_stats_serialization() {
        let stats = TensorValueStats {
            name: "layer.0.weight".to_string(),
            shape_a: vec![4, 4],
            shape_b: vec![4, 4],
            element_count: 16,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 16,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Identical,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        assert!(json.contains("\"name\":\"layer.0.weight\""));
        assert!(json.contains("\"element_count\":16"));
        assert!(json.contains("\"Identical\""));
    }

    // ==================== TensorDiffStatus serialization ====================

    #[test]
    fn test_tensor_diff_status_serialize_all_variants() {
        let variants = vec![
            (TensorDiffStatus::Identical, "\"Identical\""),
            (TensorDiffStatus::NearlyIdentical, "\"NearlyIdentical\""),
            (TensorDiffStatus::SmallDiff, "\"SmallDiff\""),
            (TensorDiffStatus::MediumDiff, "\"MediumDiff\""),
            (TensorDiffStatus::LargeDiff, "\"LargeDiff\""),
            (TensorDiffStatus::Transposed, "\"Transposed\""),
            (TensorDiffStatus::Critical, "\"Critical\""),
        ];
        for (variant, expected) in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, expected);
        }
    }

    #[test]
    fn test_tensor_diff_status_clone_and_copy() {
        let status = TensorDiffStatus::LargeDiff;
        let cloned = status.clone();
        let copied = status;
        assert_eq!(status, cloned);
        assert_eq!(status, copied);
    }

    #[test]
    fn test_tensor_diff_status_debug() {
        let debug = format!("{:?}", TensorDiffStatus::Transposed);
        assert_eq!(debug, "Transposed");
    }

    // ==================== print_tensor_diff_row - coverage for formatting ====================

    #[test]
    fn test_print_tensor_diff_row_identical() {
        let stats = TensorValueStats {
            name: "token_embd.weight".to_string(),
            shape_a: vec![100, 64],
            shape_b: vec![100, 64],
            element_count: 6400,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 6400,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Identical,
        };
        // Just ensure it doesn't panic
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_critical_with_shape_mismatch() {
        let stats = TensorValueStats {
            name: "output.weight".to_string(),
            shape_a: vec![100, 64],
            shape_b: vec![200, 32],
            element_count: 6400,
            mean_diff: 5.0,
            max_diff: 10.0,
            rmse: 6.0,
            cosine_similarity: 0.5,
            identical_count: 0,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 6400,
            status: TensorDiffStatus::Critical,
        };
        // Exercises the SHAPE MISMATCH branch (non-transpose, non-match)
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_transposed_shapes() {
        let stats = TensorValueStats {
            name: "attn.weight".to_string(),
            shape_a: vec![64, 128],
            shape_b: vec![128, 64],
            element_count: 8192,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 8192,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Transposed,
        };
        // Exercises the TRANSPOSED branch in shape printing
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_nearly_identical() {
        let stats = TensorValueStats {
            name: "norm.weight".to_string(),
            shape_a: vec![64],
            shape_b: vec![64],
            element_count: 64,
            mean_diff: 0.000_05,
            max_diff: 0.000_1,
            rmse: 0.000_07,
            cosine_similarity: 0.999_999,
            identical_count: 32,
            small_diff_count: 32,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::NearlyIdentical,
        };
        // Exercises the distribution printing path (status != Identical)
        print_tensor_diff_row(&stats);
    }
