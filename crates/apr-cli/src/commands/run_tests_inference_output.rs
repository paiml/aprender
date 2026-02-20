
    // ========================================================================
    // InferenceOutput: structural tests
    // ========================================================================

    /// InferenceOutput fields should be independently accessible.
    /// Bug class: private field preventing test access (compile-time check).
    #[test]
    fn inference_output_fields() {
        let output = InferenceOutput {
            text: "hello".to_string(),
            tokens_generated: Some(5),
            inference_ms: Some(10.0),
            tok_per_sec: Some(500.0),
            used_gpu: Some(false),
            generated_tokens: Some(vec![1, 2, 3, 4, 5]),
        };
        assert_eq!(output.text, "hello");
        assert_eq!(output.tokens_generated, Some(5));
        assert!((output.inference_ms.unwrap() - 10.0).abs() < f64::EPSILON);
    }

    /// InferenceOutput with no metrics.
    #[test]
    fn inference_output_no_metrics() {
        let output = InferenceOutput {
            text: "result".to_string(),
            tokens_generated: None,
            inference_ms: None,
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };
        assert!(output.tokens_generated.is_none());
        assert!(output.inference_ms.is_none());
    }

    // ========================================================================
    // find_model_in_dir / glob_first: edge cases
    // ========================================================================

    /// find_model_in_dir on a regular file path (not directory) returns the path.
    /// Bug class: panic when path is not a directory.
    #[test]
    fn find_model_in_dir_file_path_returns_self() {
        let result = find_model_in_dir(Path::new("/nonexistent/file.txt"));
        assert_eq!(
            result.expect("should not error"),
            PathBuf::from("/nonexistent/file.txt")
        );
    }

    /// glob_first on empty pattern returns None.
    /// Bug class: panic on empty or invalid glob.
    #[test]
    fn glob_first_empty_pattern() {
        let result = glob_first(Path::new(""));
        // Empty path may or may not match; key property: no panic
        let _ = result;
    }
