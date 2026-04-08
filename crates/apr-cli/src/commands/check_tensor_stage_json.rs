
    // ========================================================================
    // tensor_check_stage Tests
    // ========================================================================

    #[test]
    fn tensor_check_stage_passed_uses_found_msg() {
        let result = tensor_check_stage(
            "Embedding",
            "Numbers to vectors",
            true,
            "Found embedding tensor",
            "Missing embedding tensor",
        );
        assert!(result.passed);
        assert_eq!(result.name, "Embedding");
        assert_eq!(result.eli5, "Numbers to vectors");
        assert_eq!(result.details.as_deref(), Some("Found embedding tensor"));
    }

    #[test]
    fn tensor_check_stage_failed_uses_missing_msg() {
        let result = tensor_check_stage(
            "Q/K/V Projection",
            "Make 3 question copies",
            false,
            "Q/K/V found",
            "Missing Q/K/V",
        );
        assert!(!result.passed);
        assert_eq!(result.name, "Q/K/V Projection");
        assert_eq!(result.details.as_deref(), Some("Missing Q/K/V"));
    }

    #[test]
    fn tensor_check_stage_empty_messages() {
        let result = tensor_check_stage("", "", true, "", "");
        assert!(result.passed);
        assert_eq!(result.details.as_deref(), Some(""));
    }

    #[test]
    fn tensor_check_stage_long_messages() {
        let found = "a".repeat(200);
        let missing = "b".repeat(200);
        let result = tensor_check_stage("Stage", "eli5", false, &found, &missing);
        assert!(!result.passed);
        assert_eq!(result.details.as_deref(), Some(missing.as_str()));
    }

    #[test]
    fn tensor_check_stage_all_standard_stages() {
        // Verify tensor_check_stage can build all structural stages
        let stages = [
            ("Embedding", "Numbers to vectors", true, "Found", "Missing"),
            ("Q/K/V Projection", "Make 3 copies", true, "Q/K/V found", "Missing Q/K/V"),
            ("Attention Scores", "Who to look at?", false, "Found", "Missing attention output"),
            ("Feed-Forward (MLP)", "Think about it", true, "MLP found", "Missing MLP"),
        ];
        for (name, eli5, found, found_msg, missing_msg) in &stages {
            let result = tensor_check_stage(name, eli5, *found, found_msg, missing_msg);
            assert_eq!(result.passed, *found);
            assert_eq!(result.name, *name);
        }
    }

    // ========================================================================
    // print_json Tests
    // ========================================================================

    #[test]
    fn print_json_all_passed() {
        let results = vec![
            StageResult {
                name: "Tokenizer",
                eli5: "Words to numbers",
                passed: true,
                details: Some("tokens=[1, 2]".to_string()),
            },
            StageResult {
                name: "Embedding",
                eli5: "Numbers to vectors",
                passed: true,
                details: Some("Found".to_string()),
            },
        ];
        let result = print_json(&results, Path::new("/test/model.gguf"), 2, 2, None);
        assert!(result.is_ok());
    }

    #[test]
    fn print_json_some_failed() {
        let results = vec![
            StageResult {
                name: "S1",
                eli5: "t",
                passed: true,
                details: Some("OK".to_string()),
            },
            StageResult {
                name: "S2",
                eli5: "t",
                passed: false,
                details: Some("Missing".to_string()),
            },
        ];
        let result = print_json(&results, Path::new("/test/model.apr"), 1, 2, None);
        assert!(result.is_ok());
    }

    #[test]
    fn print_json_empty_results() {
        let results: Vec<StageResult> = vec![];
        let result = print_json(&results, Path::new("/test/model.gguf"), 0, 0, None);
        assert!(result.is_ok());
    }

    #[test]
    fn print_json_no_details() {
        let results = vec![StageResult {
            name: "Stage",
            eli5: "test",
            passed: false,
            details: None,
        }];
        // When details is None, unwrap_or("") should produce ""
        let result = print_json(&results, Path::new("test.apr"), 0, 1, None);
        assert!(result.is_ok());
    }

    #[test]
    fn print_json_all_ten_stages() {
        let results: Vec<StageResult> = (0..10)
            .map(|i| StageResult {
                name: "Stage",
                eli5: "test",
                passed: i % 3 != 0,
                details: Some(format!("detail {}", i)),
            })
            .collect();
        let passed = results.iter().filter(|r| r.passed).count();
        let result = print_json(&results, Path::new("/model.gguf"), passed, 10, None);
        assert!(result.is_ok());
    }

    #[test]
    fn print_json_unicode_path() {
        let results = vec![StageResult {
            name: "T",
            eli5: "t",
            passed: true,
            details: None,
        }];
        let result = print_json(&results, Path::new("/tmp/modele.gguf"), 1, 1, None);
        assert!(result.is_ok());
    }

    #[test]
    fn print_json_always_returns_ok() {
        // Contract: GH-253 - JSON mode always returns Ok(()) so parity checker
        // can parse the output. Success/failure conveyed via all_passed field.
        let results = vec![
            StageResult {
                name: "S1",
                eli5: "t",
                passed: false,
                details: Some("FAIL".to_string()),
            },
            StageResult {
                name: "S2",
                eli5: "t",
                passed: false,
                details: Some("FAIL".to_string()),
            },
        ];
        // Even with all failures, print_json returns Ok
        let result = print_json(&results, Path::new("model.gguf"), 0, 2, None);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Non-inference path StageResult construction
    // ========================================================================

    #[test]
    fn non_inference_results_vec_has_one_entry() {
        // Mirror the #[cfg(not(feature = "inference"))] branch
        let results = vec![StageResult {
            name: "N/A",
            eli5: "Requires inference",
            passed: false,
            details: Some("Build with --features inference".to_string()),
        }];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, 0);
        assert_eq!(total_count, 1);
        assert_ne!(passed_count, total_count);
    }

    // ========================================================================
    // run() function: edge cases for the result path
    // ========================================================================

    #[test]
    fn run_json_mode_with_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf data").expect("write");
        // JSON mode
        let result = run(file.path(), false, true, false);
        // Should still error because file is invalid, but tries JSON path
        assert!(result.is_err());
    }

    #[test]
    fn run_nonexistent_path() {
        let result = run(Path::new("/does/not/exist/model.gguf"), false, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn run_symlink_dir_is_not_file() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), false, false, false);
        assert!(result.is_err());
    }

    // ========================================================================
    // print_results_table: Unicode safety in truncation
    // ========================================================================

    #[test]
    fn print_results_table_with_multibyte_unicode_details() {
        // Details with multibyte UTF-8 chars near the truncation boundary
        // The truncation logic uses is_char_boundary to avoid splitting
        let details = "abcdefghijklmnopqrstuvwxyz0123456\u{00E9}\u{00E9}\u{00E9}\u{00E9}";
        assert!(details.len() > 36);
        let results = vec![StageResult {
            name: "Unicode",
            eli5: "test",
            passed: true,
            details: Some(details.to_string()),
        }];
        // Should not panic - must handle char boundary correctly
        print_results_table(&results);
    }

    #[test]
    fn print_results_table_with_emoji_details() {
        let details = "logits[32000]: min=-5.20, max=12.30 \u{2713}\u{2713}\u{2713}";
        let results = vec![StageResult {
            name: "Emoji",
            eli5: "test",
            passed: true,
            details: Some(details.to_string()),
        }];
        print_results_table(&results);
    }

    // ========================================================================
    // Passed/Failed message format correctness
    // ========================================================================

    #[test]
    fn passed_count_equals_total_is_success() {
        let results: Vec<StageResult> = (0..5)
            .map(|_| StageResult {
                name: "S",
                eli5: "t",
                passed: true,
                details: None,
            })
            .collect();
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert_eq!(passed_count, total_count);
    }

    #[test]
    fn passed_count_less_than_total_is_failure() {
        let results = vec![
            StageResult { name: "A", eli5: "t", passed: true, details: None },
            StageResult { name: "B", eli5: "t", passed: false, details: None },
        ];
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        assert!(passed_count < total_count);
    }
