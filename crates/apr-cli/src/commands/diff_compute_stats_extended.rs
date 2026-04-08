
    // ═══════════════════════════════════════════════════════════════════════════
    // Extended tests for compute_tensor_diff_stats and related functions
    // ═══════════════════════════════════════════════════════════════════════════

    // ── TensorDiffStatus::from_diff_info boundary tests ────────────────────

    #[test]
    fn test_status_identical() {
        let s = TensorDiffStatus::from_diff_info(0.0, &[4, 4], &[4, 4], 16, 16);
        assert_eq!(s, TensorDiffStatus::Identical);
    }

    #[test]
    fn test_status_nearly_identical_boundary() {
        // max_diff = 0.0009 < 0.001
        let s = TensorDiffStatus::from_diff_info(0.0009, &[4], &[4], 0, 4);
        assert_eq!(s, TensorDiffStatus::NearlyIdentical);
    }

    #[test]
    fn test_status_small_diff_boundary() {
        // max_diff = 0.001 (not < 0.001, so SmallDiff)
        let s = TensorDiffStatus::from_diff_info(0.001, &[4], &[4], 0, 4);
        assert_eq!(s, TensorDiffStatus::SmallDiff);
    }

    #[test]
    fn test_status_medium_diff_boundary() {
        let s = TensorDiffStatus::from_diff_info(0.01, &[4], &[4], 0, 4);
        assert_eq!(s, TensorDiffStatus::MediumDiff);
    }

    #[test]
    fn test_status_large_diff_boundary() {
        let s = TensorDiffStatus::from_diff_info(0.1, &[4], &[4], 0, 4);
        assert_eq!(s, TensorDiffStatus::LargeDiff);
    }

    #[test]
    fn test_status_critical_boundary() {
        let s = TensorDiffStatus::from_diff_info(1.0, &[4], &[4], 0, 4);
        assert_eq!(s, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_status_transposed_mostly_identical() {
        // 2D shapes are transposed, ident_ratio > 0.99
        let s = TensorDiffStatus::from_diff_info(0.0, &[3, 4], &[4, 3], 12, 12);
        assert_eq!(s, TensorDiffStatus::Transposed);
    }

    #[test]
    fn test_status_transposed_with_small_diffs() {
        // Transposed but not mostly identical, max_diff < 0.1
        let s = TensorDiffStatus::from_diff_info(0.05, &[3, 4], &[4, 3], 0, 12);
        assert_eq!(s, TensorDiffStatus::MediumDiff);
    }

    #[test]
    fn test_status_incompatible_shapes() {
        let s = TensorDiffStatus::from_diff_info(0.0, &[4], &[8], 4, 4);
        assert_eq!(s, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_status_3d_shape_mismatch() {
        // 3D tensors can't be "transposed" in the 2D sense
        let s = TensorDiffStatus::from_diff_info(0.0, &[2, 3, 4], &[4, 3, 2], 24, 24);
        assert_eq!(s, TensorDiffStatus::Critical);
    }

    // ── compute_tensor_diff_stats ──────────────────────────────────────────

    #[test]
    fn test_compute_stats_identical_large() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let stats = compute_tensor_diff_stats("embed", &[1000], &[1000], &data, &data, false);
        assert_eq!(stats.status, TensorDiffStatus::Identical);
        assert_eq!(stats.identical_count, 1000);
        assert!((stats.cosine_similarity - 1.0).abs() < 1e-5);
        assert_eq!(stats.max_diff, 0.0);
    }

    #[test]
    fn test_compute_stats_critical_value_diff() {
        let data_a = vec![0.0, 0.0, 0.0, 0.0];
        let data_b = vec![10.0, 10.0, 10.0, 10.0];
        let stats =
            compute_tensor_diff_stats("broken", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.status, TensorDiffStatus::Critical);
        assert!((stats.max_diff - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_nan_handling() {
        let data_a = vec![1.0, f32::NAN, 3.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats =
            compute_tensor_diff_stats("nan_test", &[3], &[3], &data_a, &data_b, false);
        // NaN pairs are counted as large_diff
        assert!(stats.large_diff_count >= 1);
    }

    #[test]
    fn test_compute_stats_inf_handling() {
        let data_a = vec![1.0, f32::INFINITY];
        let data_b = vec![1.0, 2.0];
        let stats =
            compute_tensor_diff_stats("inf_test", &[2], &[2], &data_a, &data_b, false);
        assert!(stats.large_diff_count >= 1);
    }

    #[test]
    fn test_compute_stats_transpose_aware_uses_lookup() {
        // Verify transpose_aware mode activates the lookup path and
        // produces different results than linear comparison.
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // identical data
        let stats_no_t =
            compute_tensor_diff_stats("t", &[2, 3], &[3, 2], &data_a, &data_b, false);
        let stats_with_t =
            compute_tensor_diff_stats("t", &[2, 3], &[3, 2], &data_a, &data_b, true);
        // With identical data and transpose_aware, the lookup reorders elements,
        // so results differ from linear comparison.
        // Both should have 6 elements compared
        assert_eq!(stats_no_t.element_count, 6);
        assert_eq!(stats_with_t.element_count, 6);
        // Linear comparison of identical data is identical; transpose lookup re-indexes
        assert_eq!(stats_no_t.max_diff, 0.0);
        // Transpose lookup of identical data compares different positions
        // (e.g. data_a[3]=4 vs data_b[lookup(3)]=data_b[2]=3, diff=1)
        assert!(stats_with_t.max_diff > 0.0, "transpose lookup should reorder");
    }

    #[test]
    fn test_compute_stats_unequal_lengths() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats =
            compute_tensor_diff_stats("unequal", &[5], &[3], &data_a, &data_b, false);
        // min(5, 3) = 3 elements compared
        assert_eq!(stats.element_count, 3);
    }

    // ── DiffAccumulator tests ──────────────────────────────────────────────

    #[test]
    fn test_diff_accumulator_basic() {
        let mut acc = DiffAccumulator::new();
        acc.accumulate(1.0, 1.0);
        acc.accumulate(2.0, 2.0);
        assert_eq!(acc.max_diff, 0.0);
        assert_eq!(acc.identical_count, 2);
        assert!((acc.cosine_similarity() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_diff_accumulator_with_diffs() {
        let mut acc = DiffAccumulator::new();
        acc.accumulate(1.0, 1.5); // diff = 0.5 => large
        acc.accumulate(2.0, 2.001); // diff = 0.001 => small (boundary)
        acc.accumulate(3.0, 3.0); // identical
        assert!((acc.max_diff - 0.5).abs() < 1e-5);
        assert_eq!(acc.identical_count, 1);
        assert_eq!(acc.large_diff_count, 1);
        assert_eq!(acc.small_diff_count, 1);
    }

    #[test]
    fn test_diff_accumulator_nan_counted_as_large() {
        let mut acc = DiffAccumulator::new();
        acc.accumulate(f32::NAN, 1.0);
        assert_eq!(acc.large_diff_count, 1);
        assert_eq!(acc.identical_count, 0);
    }

    #[test]
    fn test_diff_accumulator_mean_and_rmse() {
        let mut acc = DiffAccumulator::new();
        acc.accumulate(0.0, 1.0);
        acc.accumulate(0.0, 2.0);
        // mean_diff = (1.0 + 2.0) / 2 = 1.5
        assert!((acc.mean_diff(2) - 1.5).abs() < 1e-5);
        // rmse = sqrt((1 + 4) / 2) = sqrt(2.5) ~ 1.581
        assert!((acc.rmse(2) - 1.581_138_8).abs() < 0.001);
    }

    #[test]
    fn test_diff_accumulator_zero_elements() {
        let acc = DiffAccumulator::new();
        assert_eq!(acc.mean_diff(0), 0.0);
        assert_eq!(acc.rmse(0), 0.0);
        assert_eq!(acc.cosine_similarity(), 0.0);
    }

    // ── classify_diff tests ────────────────────────────────────────────────

    #[test]
    fn test_classify_diff_identical() {
        let (mut i, mut s, mut m, mut l) = (0, 0, 0, 0);
        classify_diff(0.0, &mut i, &mut s, &mut m, &mut l);
        assert_eq!((i, s, m, l), (1, 0, 0, 0));
    }

    #[test]
    fn test_classify_diff_small() {
        let (mut i, mut s, mut m, mut l) = (0, 0, 0, 0);
        classify_diff(0.0005, &mut i, &mut s, &mut m, &mut l);
        assert_eq!((i, s, m, l), (0, 1, 0, 0));
    }

    #[test]
    fn test_classify_diff_medium() {
        let (mut i, mut s, mut m, mut l) = (0, 0, 0, 0);
        classify_diff(0.005, &mut i, &mut s, &mut m, &mut l);
        assert_eq!((i, s, m, l), (0, 0, 1, 0));
    }

    #[test]
    fn test_classify_diff_large() {
        let (mut i, mut s, mut m, mut l) = (0, 0, 0, 0);
        classify_diff(0.05, &mut i, &mut s, &mut m, &mut l);
        assert_eq!((i, s, m, l), (0, 0, 0, 1));
    }

    // ── lookup_transposed_element tests ────────────────────────────────────

    #[test]
    fn test_lookup_transposed_basic() {
        // A is [2, 3], B is [3, 2]
        // data_b = [[1, 4], [2, 5], [3, 6]] stored as [1, 4, 2, 5, 3, 6]
        let data_b = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        // Looking up element at position 0 in A (row=0, col=0)
        // In B that's row*cols_b + col = 0*2 + 0 = index 0
        let val = lookup_transposed_element(&data_b, &[2, 3], &[3, 2], 0);
        assert_eq!(val, Some(1.0));
    }

    #[test]
    fn test_lookup_transposed_out_of_bounds() {
        let data_b = vec![1.0, 2.0, 3.0, 4.0];
        // A is [2, 3], index=5 => row=1, col=2 => j = 1*2 + 2 = 4 which is out of bounds
        let val = lookup_transposed_element(&data_b, &[2, 3], &[2, 2], 5);
        assert!(val.is_none());
    }

    // ── empty_tensor_stats ─────────────────────────────────────────────────

    #[test]
    fn test_empty_tensor_stats() {
        let s = empty_tensor_stats("test", &[0], &[0]);
        assert_eq!(s.element_count, 0);
        assert_eq!(s.status, TensorDiffStatus::Critical);
        assert_eq!(s.name, "test");
    }

    // ── normalize_tensor_name ──────────────────────────────────────────────

    #[test]
    fn test_normalize_name_gguf_blk_to_hf_layers() {
        let name = normalize_tensor_name("blk.0.attn_q.weight");
        assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_normalize_name_ffn_gate_proj() {
        let name = normalize_tensor_name("blk.5.ffn_gate.weight");
        assert_eq!(name, "model.layers.5.mlp.gate_proj.weight");
    }

    #[test]
    fn test_normalize_name_already_canonical() {
        let name = normalize_tensor_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
    }

    // ── truncate_path (extended edge cases) ─────────────────────────────

    #[test]
    fn test_truncate_path_nested_deep_dir() {
        let path = "/a/b/c/d/e/f/g/h/i/j/k/model-file.gguf";
        let r = truncate_path(path, 25);
        assert!(r.starts_with("..."));
        assert_eq!(r.len(), 25);
    }

    // ── print_diff_summary ─────────────────────────────────────────────────

    #[test]
    fn test_print_diff_summary_all_identical() {
        let results = vec![
            TensorValueStats {
                name: "t1".to_string(),
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
            },
        ];
        print_diff_summary(&results, 1, 0, 0, 0, 0);
    }

    #[test]
    fn test_print_diff_summary_with_critical() {
        let results = vec![
            TensorValueStats {
                name: "broken".to_string(),
                shape_a: vec![4],
                shape_b: vec![8],
                element_count: 4,
                mean_diff: 5.0,
                max_diff: 10.0,
                rmse: 6.0,
                cosine_similarity: 0.5,
                identical_count: 0,
                small_diff_count: 0,
                medium_diff_count: 0,
                large_diff_count: 4,
                status: TensorDiffStatus::Critical,
            },
        ];
        print_diff_summary(&results, 0, 0, 1, 0, 0);
    }

    #[test]
    fn test_print_diff_diagnosis_large_with_transposed_diffs() {
        let results = vec![
            TensorValueStats {
                name: "t".to_string(),
                shape_a: vec![4, 8],
                shape_b: vec![8, 4],
                element_count: 32,
                mean_diff: 0.5,
                max_diff: 0.9,
                rmse: 0.6,
                cosine_similarity: 0.95,
                identical_count: 0,
                small_diff_count: 0,
                medium_diff_count: 10,
                large_diff_count: 22,
                status: TensorDiffStatus::LargeDiff,
            },
        ];
        // Exercises the transposed_with_diffs path
        print_diff_diagnosis(&results, 0, 0, 0, 1, 0);
    }

    #[test]
    fn test_print_diff_diagnosis_medium_only() {
        let results = vec![];
        print_diff_diagnosis(&results, 0, 0, 0, 0, 1);
    }

    #[test]
    fn test_print_diff_diagnosis_transposed_and_identical() {
        let results = vec![];
        print_diff_diagnosis(&results, 5, 2, 0, 0, 0);
    }

    #[test]
    fn test_print_diff_diagnosis_all_good() {
        let results = vec![];
        print_diff_diagnosis(&results, 10, 0, 0, 0, 0);
    }
