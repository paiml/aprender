
    #[test]
    fn test_inference_config_path_from_string_cov() {
        let path_str = String::from("/model.gguf");
        let config = InferenceConfig::new(path_str);
        assert_eq!(config.model_path, PathBuf::from("/model.gguf"));
    }

    #[test]
    fn test_inference_config_path_from_pathbuf_cov() {
        let path = PathBuf::from("/model.gguf");
        let config = InferenceConfig::new(path.clone());
        assert_eq!(config.model_path, path);
    }

    #[test]
    fn test_inference_config_defaults_not_overwritten_cov() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
    }

    // =========================================================================
    // Extended Coverage Tests for InferenceResult
    // =========================================================================

    #[test]
    fn test_inference_result_with_zero_inference_time_cov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!((result.inference_ms - 0.0).abs() < f64::EPSILON);
        assert!((result.tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inference_result_with_high_tok_per_sec_cov() {
        let result = InferenceResult {
            text: "fast".to_string(),
            tokens: vec![1],
            input_token_count: 0,
            generated_token_count: 1000,
            inference_ms: 10.0,
            tok_per_sec: 100000.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        assert!(result.tok_per_sec > 10000.0);
    }

    #[test]
    fn test_inference_result_empty_text_cov() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(result.text.is_empty());
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_inference_result_empty_tokens_cov() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 5.0,
            tok_per_sec: 0.0,
            load_ms: 2.0,
            format: "SafeTensors".to_string(),
            used_gpu: false,
        };
        assert!(result.tokens.is_empty());
        assert_eq!(result.format, "SafeTensors");
    }

    #[test]
    fn test_inference_result_large_tokens_cov() {
        let tokens: Vec<u32> = (0..10000).collect();
        let result = InferenceResult {
            text: "large".to_string(),
            tokens: tokens.clone(),
            input_token_count: 100,
            generated_token_count: 9900,
            inference_ms: 1000.0,
            tok_per_sec: 9900.0,
            load_ms: 100.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.tokens.len(), 10000);
    }

    #[test]
    fn test_inference_result_format_variations_cov() {
        for format in ["GGUF", "APR", "SafeTensors", "custom"] {
            let result = InferenceResult {
                text: "t".to_string(),
                tokens: vec![1],
                input_token_count: 1,
                generated_token_count: 0,
                inference_ms: 1.0,
                tok_per_sec: 0.0,
                load_ms: 1.0,
                format: format.to_string(),
                used_gpu: false,
            };
            assert_eq!(result.format, format);
        }
    }

    // =========================================================================
    // Extended Coverage Tests for clean_model_output
    // =========================================================================

    #[test]
    fn test_clean_model_output_nested_markers_cov() {
        let raw = "<|im_start|>assistant\n<|im_start|>Hello<|im_end|><|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
    }

    #[test]
    fn test_clean_model_output_unicode_cov() {
        let raw = "<|im_start|>assistant\n你好世界<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "你好世界");
    }

    #[test]
    fn test_clean_model_output_emoji_cov() {
        let raw = "<|im_start|>assistant\n🎉 Hello! 🎊<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "🎉 Hello! 🎊");
    }

    #[test]
    fn test_clean_model_output_tabs_cov() {
        let raw = "<|im_start|>assistant\n\tTabbed content\t<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Tabbed content");
    }

    #[test]
    fn test_clean_model_output_carriage_return_cov() {
        let raw = "<|im_start|>assistant\r\nWindows line<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("Windows line"));
    }

    #[test]
    fn test_clean_model_output_special_chars_cov() {
        let raw = "Special: $@#%^&*()[]{}";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, raw);
    }

    #[test]
    fn test_clean_model_output_very_long_cov() {
        let content = "x".repeat(100000);
        let raw = format!("<|im_start|>assistant\n{}<|im_end|>", content);
        let cleaned = clean_model_output(&raw);
        assert_eq!(cleaned.len(), 100000);
    }

    #[test]
    fn test_clean_model_output_assistant_without_newline_cov() {
        // Tests the <|im_start|>assistant marker without trailing newline
        let raw = "<|im_start|>assistantHello<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello");
    }

    // =========================================================================
    // Extended Coverage Tests for prefault_mmap
    // =========================================================================

    #[test]
    fn test_prefault_mmap_single_byte_cov() {
        let data = vec![42u8; 1];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_exactly_one_page_cov() {
        let data = vec![0u8; 4096];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_minus_one_cov() {
        let data = vec![0u8; 4095];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_page_plus_one_cov() {
        let data = vec![0u8; 4097];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_with_nonzero_data_cov() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_all_255_cov() {
        let data = vec![255u8; 4096 * 2];
        prefault_mmap(&data);
    }

    // =========================================================================
    // Run Inference Error Path Tests
    // =========================================================================

    #[test]
    fn test_run_inference_empty_path_cov() {
        let config = InferenceConfig::new("");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_directory_path_cov() {
        let config = InferenceConfig::new("/tmp");
        let result = run_inference(&config);
        // Either fails to read as file or fails format detection
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_exactly_8_bytes_cov() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("exactly_8_bytes.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[0, 0, 0, 0, 0, 0, 0, 0]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        // Should fail format detection (not a valid magic)
        assert!(result.is_err());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_run_inference_7_bytes_cov() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("seven_bytes.bin");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&[1, 2, 3, 4, 5, 6, 7]).expect("write");

        let config = InferenceConfig::new(&path);
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));

        let _ = std::fs::remove_file(path);
    }

    // =========================================================================
    // InferenceConfig Trace Fields Tests
    // =========================================================================

    #[test]
    fn test_inference_config_trace_fields_debug_cov() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: Some("test".to_string()),
            input_tokens: None,
            max_tokens: 32,
            temperature: 0.0,
            top_k: 1,
            no_gpu: false,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/trace.json")),
            trace_steps: Some(vec!["embedding".to_string(), "attention".to_string()]),
            verbose: false,
            stop_tokens: Vec::new(),
            use_mock_backend: false,
        };
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("trace_verbose"));
        assert!(debug_str.contains("trace_output"));
        assert!(debug_str.contains("trace_steps"));
    }

    #[test]
    fn test_inference_config_trace_fields_clone_cov() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: None,
            input_tokens: Some(vec![1, 2, 3]),
            max_tokens: 64,
            temperature: 0.5,
            top_k: 10,
            no_gpu: true,
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: true,
            stop_tokens: Vec::new(),
            use_mock_backend: false,
        };
        let cloned = config.clone();
        assert_eq!(cloned.trace_verbose, config.trace_verbose);
        assert_eq!(cloned.trace_output, config.trace_output);
        assert_eq!(cloned.trace_steps, config.trace_steps);
    }

    #[test]
    fn test_inference_result_all_fields_cov() {
        let result = InferenceResult {
            text: "generated output".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
            generated_token_count: 3,
            inference_ms: 123.456,
            tok_per_sec: 24.32,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.text, "generated output");
        assert_eq!(result.tokens.len(), 5);
        assert_eq!(result.input_token_count, 2);
        assert_eq!(result.generated_token_count, 3);
        assert!((result.inference_ms - 123.456).abs() < 0.001);
        assert!((result.tok_per_sec - 24.32).abs() < 0.01);
        assert!((result.load_ms - 50.0).abs() < 0.01);
        assert_eq!(result.format, "GGUF");
        assert!(result.used_gpu);
    }

    // =========================================================================
    // Extended Coverage Tests: InferenceConfig builders
    // =========================================================================

    #[test]
    fn test_inference_config_all_methods_chain_ext_cov() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(100)
            .with_temperature(1.5)
            .with_top_k(50)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        assert_eq!(config.prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, 100);
        assert!((config.temperature - 1.5).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_defaults_ext_cov() {
        let config = InferenceConfig::new("/model.apr");

        // Check all default values
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_with_input_tokens_only_ext_cov() {
        let config =
            InferenceConfig::new("/model.safetensors").with_input_tokens(vec![100, 200, 300, 400]);

        assert_eq!(config.input_tokens, Some(vec![100, 200, 300, 400]));
        assert!(config.prompt.is_none());
    }

    #[test]
    fn test_inference_config_temperature_extremes_ext_cov() {
        let cold_config = InferenceConfig::new("/model.gguf").with_temperature(0.0);
        let hot_config = InferenceConfig::new("/model.gguf").with_temperature(2.0);

        assert!((cold_config.temperature - 0.0).abs() < f32::EPSILON);
        assert!((hot_config.temperature - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_config_top_k_values_ext_cov() {
        let greedy = InferenceConfig::new("/model.gguf").with_top_k(1);
        let wide = InferenceConfig::new("/model.gguf").with_top_k(100);
        let disabled = InferenceConfig::new("/model.gguf").with_top_k(0);

        assert_eq!(greedy.top_k, 1);
        assert_eq!(wide.top_k, 100);
        assert_eq!(disabled.top_k, 0);
    }

    // =========================================================================
    // Extended Coverage Tests: clean_model_output
    // =========================================================================

    #[test]
    fn test_clean_model_output_endoftext_ext_cov() {
        let raw = "Hello world<|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_model_output_preserves_other_text_ext_cov() {
        let raw = "[INST]user input[/INST]assistant response";
        let cleaned = clean_model_output(raw);
        // Should preserve text since [INST] tokens aren't in the markers list
        assert!(cleaned.contains("user input"));
        assert!(cleaned.contains("response"));
    }
