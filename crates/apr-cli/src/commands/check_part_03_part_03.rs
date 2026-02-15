
    #[test]
    fn test_result_aggregation_partial_pass() {
        let results = vec![
            StageResult {
                name: "S1",
                eli5: "t",
                passed: true,
                details: None,
            },
            StageResult {
                name: "S2",
                eli5: "t",
                passed: false,
                details: Some("Missing tensor".to_string()),
            },
            StageResult {
                name: "S3",
                eli5: "t",
                passed: true,
                details: None,
            },
        ];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 2);
        assert_eq!(total_count, 3);
        assert_ne!(passed_count, total_count);
    }

    #[test]
    fn test_result_aggregation_single_pass() {
        let results = vec![StageResult {
            name: "S1",
            eli5: "t",
            passed: true,
            details: None,
        }];
        let passed_count = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed_count, results.len());
    }

    #[test]
    fn test_result_aggregation_single_fail() {
        let results = vec![StageResult {
            name: "S1",
            eli5: "t",
            passed: false,
            details: None,
        }];
        let passed_count = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed_count, 0);
        assert_ne!(passed_count, results.len());
    }

    #[test]
    fn test_result_aggregation_empty() {
        let results: Vec<StageResult> = vec![];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 0);
        assert_eq!(total_count, 0);
        // Edge case: 0 == 0 means "all passed" which is vacuously true
        assert_eq!(passed_count, total_count);
    }

    // ========================================================================
    // print_results_table Edge Cases
    // ========================================================================

    #[test]
    fn test_print_results_single_element_no_separator_after() {
        // With a single result, there should be no separator between rows
        let results = vec![StageResult {
            name: "Only",
            eli5: "Single",
            passed: true,
            details: Some("OK".to_string()),
        }];
        // Should not panic and should skip the inter-row separator
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_three_elements() {
        let results = vec![
            StageResult {
                name: "A",
                eli5: "t",
                passed: true,
                details: Some("d1".to_string()),
            },
            StageResult {
                name: "B",
                eli5: "t",
                passed: false,
                details: None,
            },
            StageResult {
                name: "C",
                eli5: "t",
                passed: true,
                details: Some("d3".to_string()),
            },
        ];
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_ten_stages() {
        // Simulate a full 10-stage pipeline
        let results: Vec<StageResult> = (0..10)
            .map(|i| StageResult {
                name: "Stage",
                eli5: "test",
                passed: i % 2 == 0,
                details: Some(format!("stage {} details", i)),
            })
            .collect();
        assert_eq!(results.len(), 10);
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_details_none_shows_dash() {
        // When details is None, print_results_table should use "-"
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: None,
        };
        let details = result.details.as_deref().unwrap_or("-");
        assert_eq!(details, "-");
    }

    #[test]
    fn test_print_results_details_some_shows_value() {
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: Some("logits[32000]".to_string()),
        };
        let details = result.details.as_deref().unwrap_or("-");
        assert_eq!(details, "logits[32000]");
    }

    // ========================================================================
    // run() Function: Error Path Tests
    // ========================================================================

    #[test]
    fn test_run_empty_file_gguf() {
        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // Empty file should fail
        let result = run(file.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_empty_file_apr() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        // Empty file should fail
        let result = run(file.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_no_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("modelfile");
        std::fs::write(&path, b"some data").expect("write");
        let result = run(&path, false, false);
        // No extension -> unsupported format or feature disabled
        assert!(result.is_err());
    }

    #[test]
    fn test_run_uppercase_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.GGUF");
        std::fs::write(&path, b"not valid gguf").expect("write");
        let result = run(&path, false, false);
        // Should attempt GGUF parsing (lowercased) but fail due to invalid content
        assert!(result.is_err());
    }

    #[test]
    fn test_run_mixed_case_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.Apr");
        std::fs::write(&path, b"not valid apr").expect("write");
        let result = run(&path, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_txt_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.txt");
        std::fs::write(&path, b"text data").expect("write");
        let result = run(&path, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_json_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"{}").expect("write");
        let result = run(&path, false, false);
        assert!(result.is_err());
    }

    // ========================================================================
    // Softmax Validation Logic (inline tests)
    // ========================================================================

    #[test]
    fn test_softmax_sum_validation_exact() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        let valid = (prob_sum - 1.0).abs() < 0.001;
        assert!(valid, "softmax sum should be ~1.0, got {prob_sum}");
    }

    #[test]
    fn test_softmax_sum_single_element() {
        let logits = vec![5.0_f32];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        assert!((probs[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_softmax_sum_large_logits() {
        // Numerically stable softmax should handle large values
        let logits = vec![1000.0_f32, 1001.0, 999.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        assert!(!probs.iter().any(|x| x.is_nan()));
        assert!(!probs.iter().any(|x| x.is_infinite()));
    }

    #[test]
    fn test_softmax_sum_negative_logits() {
        let logits = vec![-10.0_f32, -20.0, -5.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax_all_zeros() {
        let logits = vec![0.0_f32, 0.0, 0.0, 0.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        // Uniform distribution
        for p in &probs {
            assert!((p - 0.25).abs() < 0.001);
        }
    }

    #[test]
    fn test_softmax_two_elements_dominant() {
        let logits = vec![100.0_f32, 0.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        // First element dominates
        assert!(probs[0] > 0.99);
        assert!(probs[1] < 0.01);
    }

    // ========================================================================
    // NaN/Inf Detection Logic (mirrors logits validation)
    // ========================================================================

    #[test]
    fn test_nan_detection_in_logits() {
        let logits = vec![1.0_f32, f32::NAN, 3.0];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(has_nan);
        assert!(!has_inf);
        assert!(!valid);
    }

    #[test]
    fn test_inf_detection_in_logits() {
        let logits = vec![1.0_f32, f32::INFINITY, 3.0];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(!has_nan);
        assert!(has_inf);
        assert!(!valid);
    }

    #[test]
    fn test_neg_inf_detection_in_logits() {
        let logits = vec![1.0_f32, f32::NEG_INFINITY, 3.0];
        let has_inf = logits.iter().any(|x| x.is_infinite());
        assert!(has_inf);
    }

    #[test]
    fn test_empty_logits_invalid() {
        let logits: Vec<f32> = vec![];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(!valid, "empty logits should be invalid");
    }

    #[test]
    fn test_valid_logits() {
        let logits = vec![0.1_f32, -0.5, 2.3, -1.0, 0.0];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        let valid = !has_nan && !has_inf && !logits.is_empty();
        assert!(valid);
    }

    #[test]
    fn test_logits_with_both_nan_and_inf() {
        let logits = vec![f32::NAN, f32::INFINITY];
        let has_nan = logits.iter().any(|x| x.is_nan());
        let has_inf = logits.iter().any(|x| x.is_infinite());
        assert!(has_nan);
        assert!(has_inf);
        assert!(!((!has_nan) && (!has_inf) && !logits.is_empty()));
    }

    // ========================================================================
    // Tensor Name Matching Logic (APR checks)
    // ========================================================================

    #[test]
    fn test_embedding_tensor_detection() {
        let names = vec!["token_embd.weight", "blk.0.attn_q.weight"];
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(has_embed);
    }

    #[test]
    fn test_embedding_tensor_detection_wte() {
        let names = vec!["wte.weight", "blk.0.ffn_gate.weight"];
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(has_embed);
    }

    #[test]
    fn test_embedding_tensor_detection_missing() {
        let names = vec!["blk.0.attn_q.weight", "blk.0.ffn_gate.weight"];
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(!has_embed);
    }

    #[test]
    fn test_qkv_projection_detection_all_present() {
        let names = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ];
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_qkv_projection_detection_missing_v() {
        let names = vec!["blk.0.attn_q.weight", "blk.0.attn_k.weight"];
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        assert!(has_q && has_k);
        assert!(!has_v);
        assert!(!(has_q && has_k && has_v));
    }
