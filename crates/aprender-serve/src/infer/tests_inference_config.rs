
    // =========================================================================
    // InferenceConfig Builder Tests
    // =========================================================================

    #[test]
    fn test_inference_config_builder() {
        let config = InferenceConfig::new("/path/to/model.gguf")
            .with_prompt("Hello")
            .with_max_tokens(64)
            .with_temperature(0.7)
            .with_top_k(40)
            .without_gpu()
            .with_verbose(true);

        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
        assert_eq!(config.prompt, Some("Hello".to_string()));
        assert_eq!(config.max_tokens, 64);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 40);
        assert!(config.no_gpu);
        assert!(config.verbose);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::new("/model.gguf");
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_with_input_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![1, 2, 3, 4]);
        assert_eq!(config.input_tokens, Some(vec![1, 2, 3, 4]));
        assert!(config.prompt.is_none()); // prompt not set
    }

    #[test]
    fn test_inference_config_with_trace() {
        let config = InferenceConfig::new("/model.gguf").with_trace(true);
        assert!(config.trace);
        assert!(!config.trace_verbose);
    }

    // =========================================================================
    // Output Cleaning Tests
    // =========================================================================

    #[test]
    fn test_clean_model_output() {
        let raw = "<|im_start|>assistant\nHello world!<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_no_markers() {
        let raw = "Hello world!";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_multiple_markers() {
        let raw = "<|im_start|><|im_start|>assistant\nMultiple markers<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Multiple markers");
    }

    #[test]
    fn test_clean_model_output_whitespace_only() {
        let raw = "<|im_start|>assistant\n   <|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_with_newlines() {
        let raw = "<|im_start|>assistant\nLine 1\nLine 2<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Line 1\nLine 2");
    }

    // =========================================================================
    // Prefault Mmap Tests
    // =========================================================================

    #[test]
    fn test_prefault_mmap_empty() {
        let data: &[u8] = &[];
        prefault_mmap(data); // Should not panic
    }

    #[test]
    fn test_prefault_mmap_small() {
        let data = vec![0u8; 1000];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_aligned() {
        let data = vec![0u8; 4096 * 3]; // 3 pages
        prefault_mmap(&data);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_run_inference_file_not_found() {
        let config = InferenceConfig::new("/nonexistent/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        // F-SEC-222: validate_model_path now catches nonexistent files early
        assert!(
            err.to_string().contains("File not found")
                || err.to_string().contains("Failed to read model"),
            "Expected file not found error, got: {}",
            err
        );
    }

    #[test]
    fn test_run_inference_file_too_small() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("tiny_model.gguf");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[1, 2, 3]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_run_inference_invalid_format() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("invalid_format_model.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Format"));

        let _ = std::fs::remove_file(path);
    }

    // =========================================================================
    // InferenceResult Tests
    // =========================================================================

    #[test]
    fn test_inference_result_default() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 100.0,
            tok_per_sec: 20.0,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.text, "test");
        assert_eq!(result.tokens, vec![1, 2, 3]);
        assert_eq!(result.generated_token_count, 2);
    }

    #[test]
    fn test_inference_result_clone() {
        let result = InferenceResult {
            text: "hello".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 10.0,
            tok_per_sec: 0.0,
            load_ms: 5.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        let cloned = result.clone();
        assert_eq!(result.text, cloned.text);
        assert_eq!(result.used_gpu, cloned.used_gpu);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_inference_config_with_trace_enabled() {
        let config = InferenceConfig::new("/model.gguf").with_trace(true);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_with_trace_disabled() {
        let config = InferenceConfig::new("/model.gguf").with_trace(false);
        assert!(!config.trace);
    }

    #[test]
    fn test_inference_config_builder_all_options() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(128)
            .with_temperature(0.8)
            .with_top_k(50)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        assert_eq!(config.prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, 128);
        assert!((config.temperature - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    #[test]
    fn test_clean_model_output_chatml_markers() {
        let raw = "<|im_start|>assistant\nHello world!<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        // Should clean ChatML markers
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        assert!(!cleaned.contains("<|endoftext|>"));
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_model_output_empty_input() {
        let cleaned = clean_model_output("");
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_model_output_only_markers() {
        let raw = "<|im_start|><|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_prefault_mmap_large() {
        let data = vec![0u8; 4096 * 10]; // 10 pages
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_unaligned() {
        let data = vec![0u8; 4096 + 100]; // 1 page + extra
        prefault_mmap(&data);
    }

    #[test]
    fn test_inference_config_debug() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("test");
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("model_path"));
        assert!(debug_str.contains("prompt"));
    }

    #[test]
    fn test_inference_result_debug() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 1,
            inference_ms: 10.0,
            tok_per_sec: 100.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("text"));
        assert!(debug_str.contains("tokens"));
    }

    #[test]
    fn test_inference_config_with_zero_temperature() {
        let config = InferenceConfig::new("/model.gguf").with_temperature(0.0);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_with_high_temperature() {
        let config = InferenceConfig::new("/model.gguf").with_temperature(2.0);
        assert!((config.temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_with_zero_top_k() {
        let config = InferenceConfig::new("/model.gguf").with_top_k(0);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_inference_config_with_large_top_k() {
        let config = InferenceConfig::new("/model.gguf").with_top_k(1000);
        assert_eq!(config.top_k, 1000);
    }

    #[test]
    fn test_inference_config_with_zero_max_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(0);
        assert_eq!(config.max_tokens, 0);
    }

    #[test]
    fn test_inference_config_with_large_max_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(4096);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_inference_config_chaining() {
        // Test that builder methods can be chained in any order
        let config = InferenceConfig::new("/m.gguf")
            .with_verbose(true)
            .with_trace(true)
            .with_prompt("p")
            .without_gpu()
            .with_max_tokens(10);
        assert!(config.verbose);
        assert!(config.trace);
        assert_eq!(config.prompt, Some("p".to_string()));
        assert!(config.no_gpu);
        assert_eq!(config.max_tokens, 10);
    }

    #[test]
    fn test_clean_model_output_preserves_content() {
        let raw = "Hello, this is a test without any markers.";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, raw);
    }

    #[test]
    fn test_clean_model_output_partial_markers() {
        let raw = "Hello <|im_start|> world <|im_end|> test";
        let cleaned = clean_model_output(raw);
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
    }

    // =========================================================================
    // Extended Coverage Tests for InferenceConfig
    // =========================================================================

    #[test]
    fn test_inference_config_clone_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("test")
            .with_max_tokens(64)
            .with_temperature(0.5)
            .with_top_k(20)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);
        let cloned = config.clone();
        assert_eq!(config.model_path, cloned.model_path);
        assert_eq!(config.prompt, cloned.prompt);
        assert_eq!(config.max_tokens, cloned.max_tokens);
        assert!((config.temperature - cloned.temperature).abs() < f32::EPSILON);
        assert_eq!(config.top_k, cloned.top_k);
        assert_eq!(config.no_gpu, cloned.no_gpu);
        assert_eq!(config.verbose, cloned.verbose);
        assert_eq!(config.trace, cloned.trace);
    }

    #[test]
    fn test_inference_config_input_tokens_clone_cov() {
        let config = InferenceConfig::new("/m.gguf").with_input_tokens(vec![1, 2, 3, 4, 5]);
        let cloned = config.clone();
        assert_eq!(config.input_tokens, cloned.input_tokens);
    }

    #[test]
    fn test_inference_config_empty_prompt_cov() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("");
        assert_eq!(config.prompt, Some(String::new()));
    }

    #[test]
    fn test_inference_config_long_prompt_cov() {
        let long_prompt = "x".repeat(10000);
        let config = InferenceConfig::new("/model.gguf").with_prompt(&long_prompt);
        assert_eq!(config.prompt, Some(long_prompt));
    }

    #[test]
    fn test_inference_config_empty_input_tokens_cov() {
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![]);
        assert_eq!(config.input_tokens, Some(vec![]));
    }

    #[test]
    fn test_inference_config_large_input_tokens_cov() {
        let tokens: Vec<u32> = (0..1000).collect();
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(tokens.clone());
        assert_eq!(config.input_tokens, Some(tokens));
    }

    #[test]
    fn test_inference_config_negative_temperature_cov() {
        // Temperature can be set to any float, even negative
        let config = InferenceConfig::new("/model.gguf").with_temperature(-1.0);
        assert!((config.temperature - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_verbose_false_cov() {
        let config = InferenceConfig::new("/model.gguf").with_verbose(false);
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_path_with_spaces_cov() {
        let config = InferenceConfig::new("/path/with spaces/model.gguf");
        assert_eq!(
            config.model_path,
            PathBuf::from("/path/with spaces/model.gguf")
        );
    }
