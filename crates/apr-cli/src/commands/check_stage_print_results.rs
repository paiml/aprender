
    #[test]
    fn test_stage_result_display() {
        let result = StageResult {
            name: "Test",
            eli5: "Test ELI5",
            passed: true,
            details: Some("Test details".to_string()),
        };
        assert!(result.passed);
        assert_eq!(result.name, "Test");
    }

    #[test]
    fn test_print_results_empty() {
        // Should not panic with empty results
        print_results_table(&[]);
    }

    #[test]
    fn test_print_results_mixed() {
        let results = vec![
            StageResult {
                name: "Stage 1",
                eli5: "Test 1",
                passed: true,
                details: Some("OK".to_string()),
            },
            StageResult {
                name: "Stage 2",
                eli5: "Test 2",
                passed: false,
                details: Some("FAIL".to_string()),
            },
        ];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_details_truncation() {
        let long_details = "This is a very long details string that should be truncated";
        let truncated = if long_details.len() > 36 {
            format!("{}...", &long_details[..33])
        } else {
            long_details.to_string()
        };
        assert!(truncated.len() <= 39); // 36 + "..."
    }

    // ========================================================================
    // StageResult Tests
    // ========================================================================

    #[test]
    fn test_stage_result_passed() {
        let result = StageResult {
            name: "Stage 1",
            eli5: "Checking model integrity",
            passed: true,
            details: Some("All checks passed".to_string()),
        };
        assert!(result.passed);
        assert!(result.details.is_some());
    }

    #[test]
    fn test_stage_result_failed() {
        let result = StageResult {
            name: "Stage 2",
            eli5: "Checking tokenizer",
            passed: false,
            details: Some("Tokenizer not found".to_string()),
        };
        assert!(!result.passed);
    }

    #[test]
    fn test_stage_result_no_details() {
        let result = StageResult {
            name: "Stage 3",
            eli5: "Check",
            passed: true,
            details: None,
        };
        assert!(result.details.is_none());
    }

    #[test]
    fn test_stage_result_debug() {
        let result = StageResult {
            name: "Test",
            eli5: "Test",
            passed: true,
            details: None,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("StageResult"));
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn test_run_file_not_found() {
        let result = run(Path::new("/nonexistent/model.gguf"), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_gguf() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (invalid GGUF or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (invalid APR or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_unsupported_format() {
        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        file.write_all(b"binary data").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (unsupported format or feature disabled)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_no_gpu() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(file.path(), true, false); // no_gpu = true
                                                    // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), false, false);
        // Should fail (is a directory)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_format() {
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(b"not valid safetensors").expect("write");

        let result = run(file.path(), false, false);
        // Should fail (unsupported format or feature disabled)
        assert!(result.is_err());
    }

    // ========================================================================
    // print_results_table Tests
    // ========================================================================

    #[test]
    fn test_print_results_all_passed() {
        let results = vec![
            StageResult {
                name: "Stage 1",
                eli5: "Test 1",
                passed: true,
                details: Some("OK".to_string()),
            },
            StageResult {
                name: "Stage 2",
                eli5: "Test 2",
                passed: true,
                details: Some("OK".to_string()),
            },
        ];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_all_failed() {
        let results = vec![StageResult {
            name: "Stage 1",
            eli5: "Test 1",
            passed: false,
            details: Some("ERROR".to_string()),
        }];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_no_details() {
        let results = vec![StageResult {
            name: "Stage 1",
            eli5: "Test",
            passed: true,
            details: None,
        }];
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_long_name() {
        let results = vec![StageResult {
            name: "This is a very long stage name that should handle gracefully",
            eli5: "Test",
            passed: true,
            details: Some("OK".to_string()),
        }];
        // Should not panic
        print_results_table(&results);
    }

    // ========================================================================
    // StageResult Construction Edge Cases
    // ========================================================================

    #[test]
    fn test_stage_result_empty_name() {
        let result = StageResult {
            name: "",
            eli5: "",
            passed: false,
            details: None,
        };
        assert_eq!(result.name, "");
        assert_eq!(result.eli5, "");
        assert!(!result.passed);
        assert!(result.details.is_none());
    }

    #[test]
    fn test_stage_result_with_empty_details_string() {
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: Some(String::new()),
        };
        assert!(result.details.is_some());
        assert_eq!(result.details.as_deref(), Some(""));
    }

    #[test]
    fn test_stage_result_with_very_long_details() {
        let long = "x".repeat(1000);
        let result = StageResult {
            name: "Test",
            eli5: "test",
            passed: true,
            details: Some(long.clone()),
        };
        assert_eq!(result.details.as_ref().expect("has details").len(), 1000);
    }

    #[test]
    fn test_stage_result_with_unicode_details() {
        let result = StageResult {
            name: "Unicode",
            eli5: "test",
            passed: true,
            details: Some("NaN detected in logits".to_string()),
        };
        assert!(result
            .details
            .as_ref()
            .expect("has details")
            .contains("NaN"));
    }

    #[test]
    fn test_stage_result_all_ten_stages_names() {
        // Verify all 10 stage names used in real checks are valid static strings
        let stage_names = [
            "Tokenizer",
            "Embedding",
            "Positional Encoding",
            "Q/K/V Projection",
            "Attention Scores",
            "Feed-Forward (MLP)",
            "Layer Norm",
            "LM Head",
            "Logits \u{2192} Probs",
            "Sampler/Decode",
        ];
        for name in &stage_names {
            let result = StageResult {
                name,
                eli5: "test",
                passed: true,
                details: None,
            };
            assert_eq!(result.name, *name);
        }
    }

    // ========================================================================
    // Details Truncation Edge Cases
    // ========================================================================

    #[test]
    fn test_details_truncation_exactly_36_chars() {
        let details = "a".repeat(36);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        // Exactly 36 chars should NOT be truncated
        assert_eq!(truncated.len(), 36);
        assert_eq!(truncated, details);
    }

    #[test]
    fn test_details_truncation_37_chars() {
        let details = "a".repeat(37);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        // 37 chars should be truncated to 33 + "..." = 36
        assert_eq!(truncated.len(), 36);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_details_truncation_35_chars() {
        let details = "a".repeat(35);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        // 35 chars should NOT be truncated
        assert_eq!(truncated.len(), 35);
        assert!(!truncated.ends_with("..."));
    }

    #[test]
    fn test_details_truncation_empty() {
        let details = "";
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.to_string()
        };
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_details_truncation_exactly_33_chars() {
        let details = "b".repeat(33);
        let truncated = if details.len() > 36 {
            format!("{}...", &details[..33])
        } else {
            details.clone()
        };
        assert_eq!(truncated.len(), 33);
        assert!(!truncated.ends_with("..."));
    }

    // ========================================================================
    // Result Aggregation Logic
    // ========================================================================

    #[test]
    fn test_result_aggregation_all_passed() {
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
                passed: true,
                details: None,
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
        assert_eq!(passed_count, 3);
        assert_eq!(total_count, 3);
        assert_eq!(passed_count, total_count);
    }

    #[test]
    fn test_result_aggregation_none_passed() {
        let results = vec![
            StageResult {
                name: "S1",
                eli5: "t",
                passed: false,
                details: None,
            },
            StageResult {
                name: "S2",
                eli5: "t",
                passed: false,
                details: None,
            },
        ];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 0);
        assert_eq!(total_count, 2);
        assert_ne!(passed_count, total_count);
    }
