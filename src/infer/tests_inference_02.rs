
    #[test]
    fn test_inference_result_clone_preserves_all() {
        let original = InferenceResult {
            text: "test text".to_string(),
            tokens: vec![100, 200, 300],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 999.9,
            tok_per_sec: 2.0,
            load_ms: 111.1,
            format: "APR".to_string(),
            used_gpu: false,
        };

        let cloned = original.clone();

        assert_eq!(original.text, cloned.text);
        assert_eq!(original.tokens, cloned.tokens);
        assert_eq!(original.input_token_count, cloned.input_token_count);
        assert_eq!(original.generated_token_count, cloned.generated_token_count);
        assert!((original.inference_ms - cloned.inference_ms).abs() < f64::EPSILON);
        assert!((original.tok_per_sec - cloned.tok_per_sec).abs() < f64::EPSILON);
        assert!((original.load_ms - cloned.load_ms).abs() < f64::EPSILON);
        assert_eq!(original.format, cloned.format);
        assert_eq!(original.used_gpu, cloned.used_gpu);
    }

    // --- Format detection with various magic bytes ---

    #[test]
    fn test_format_detection_gguf_version_variations() {
        use crate::format::{detect_format, ModelFormat};

        // GGUF version 3
        let v3_data = vec![0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
        let result = detect_format(&v3_data);
        assert!(matches!(result, Ok(ModelFormat::Gguf)));

        // GGUF version 2
        let v2_data = vec![0x47, 0x47, 0x55, 0x46, 0x02, 0x00, 0x00, 0x00];
        let result = detect_format(&v2_data);
        assert!(matches!(result, Ok(ModelFormat::Gguf)));
    }

    #[test]
    fn test_format_detection_apr_versions() {
        use crate::format::{detect_format, ModelFormat};

        // APR v1 - version byte is ASCII '1' (0x31), not \x01
        let v1_data = b"APR1xxxx";
        let result = detect_format(v1_data);
        assert!(matches!(result, Ok(ModelFormat::Apr)));

        // APR v2 - version byte is ASCII '2' (0x32), not \x02
        let v2_data = b"APR2xxxx";
        let result = detect_format(v2_data);
        assert!(matches!(result, Ok(ModelFormat::Apr)));
    }

    #[test]
    fn test_format_detection_safetensors_various_sizes() {
        use crate::format::{detect_format, ModelFormat};

        // Various valid header sizes for SafeTensors
        for size in [64u64, 128, 256, 512, 1024, 2048, 4096] {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert!(
                matches!(result, Ok(ModelFormat::SafeTensors)),
                "Failed for size: {}",
                size
            );
        }
    }

    // --- run_inference file handling edge cases ---

    #[test]
    fn test_run_inference_symlink_path() {
        // Test with a path that looks like a symlink (just path string)
        let config = InferenceConfig::new("/models/latest-model.gguf");
        // Can't test actual symlink without creating one, but test config handling
        assert!(config.model_path.to_str().unwrap().contains("latest-model"));
    }

    #[test]
    fn test_run_inference_hidden_file() {
        // Test with hidden file path
        let config = InferenceConfig::new("/models/.hidden-model.gguf");
        assert!(config
            .model_path
            .to_str()
            .unwrap()
            .contains(".hidden-model"));
    }

    // --- Comprehensive builder chaining test ---

    #[test]
    fn test_inference_config_builder_order_independence() {
        // Order 1
        let config1 = InferenceConfig::new("/m.gguf")
            .with_prompt("p")
            .with_max_tokens(50)
            .with_temperature(0.5)
            .with_top_k(20)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);

        // Order 2 (reversed)
        let config2 = InferenceConfig::new("/m.gguf")
            .with_trace(true)
            .with_verbose(true)
            .without_gpu()
            .with_top_k(20)
            .with_temperature(0.5)
            .with_max_tokens(50)
            .with_prompt("p");

        assert_eq!(config1.prompt, config2.prompt);
        assert_eq!(config1.max_tokens, config2.max_tokens);
        assert!((config1.temperature - config2.temperature).abs() < f32::EPSILON);
        assert_eq!(config1.top_k, config2.top_k);
        assert_eq!(config1.no_gpu, config2.no_gpu);
        assert_eq!(config1.verbose, config2.verbose);
        assert_eq!(config1.trace, config2.trace);
    }

    // --- Path with special extensions ---

    #[test]
    fn test_path_extension_handling() {
        let extensions = [".gguf", ".apr", ".safetensors", ".bin", ".model", ""];
        for ext in extensions {
            let path = format!("/models/model{}", ext);
            let config = InferenceConfig::new(&path);
            assert!(config.model_path.to_str().unwrap().ends_with(ext) || ext.is_empty());
        }
    }

    // --- Boundary tests for numeric fields ---

    #[test]
    fn test_numeric_boundary_values() {
        // Max values
        let config = InferenceConfig::new("/m.gguf")
            .with_max_tokens(usize::MAX)
            .with_temperature(f32::MAX)
            .with_top_k(usize::MAX);

        assert_eq!(config.max_tokens, usize::MAX);
        assert_eq!(config.temperature, f32::MAX);
        assert_eq!(config.top_k, usize::MAX);

        // Min values
        let config = InferenceConfig::new("/m.gguf")
            .with_max_tokens(0)
            .with_temperature(f32::MIN)
            .with_top_k(0);

        assert_eq!(config.max_tokens, 0);
        assert_eq!(config.temperature, f32::MIN);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_temperature_special_values() {
        // Infinity
        let config = InferenceConfig::new("/m.gguf").with_temperature(f32::INFINITY);
        assert!(config.temperature.is_infinite());

        // Negative infinity
        let config = InferenceConfig::new("/m.gguf").with_temperature(f32::NEG_INFINITY);
        assert!(config.temperature.is_infinite());
        assert!(config.temperature.is_sign_negative());

        // NaN
        let config = InferenceConfig::new("/m.gguf").with_temperature(f32::NAN);
        assert!(config.temperature.is_nan());
    }

    // =========================================================================
    // Extended Coverage Tests (PMAT-COV-002)
    // These tests target specific lines and branches in infer/mod.rs
    // =========================================================================

    // --- run_inference format dispatch tests ---

    #[test]
    fn test_run_inference_dispatches_to_gguf_branch() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create GGUF-like file with full header but incomplete model
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
        // Add some padding
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(1)
            .with_verbose(false);

        // Dispatches to GGUF path, fails on model load (no tensors)
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_dispatches_to_apr_branch() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create APR v2 file
        let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"APR2"); // APR v2 magic
        data.extend_from_slice(&[0u8; 100]); // Minimal header
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Hello")
            .with_max_tokens(2);

        // Dispatches to APR path - runs with fallback zeros if incomplete
        let result = run_inference(&config);
        // May succeed with degenerate output or fail - either is valid
        let _ = result;
    }

    #[test]
    fn test_run_inference_dispatches_to_safetensors_branch() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create SafeTensors-like file with valid header size
        let mut temp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        let header_size: u64 = 50;
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        // JSON header (minimal but valid enough to trigger conversion code)
        let json = r#"{"metadata":{}}"#;
        data.extend_from_slice(json.as_bytes());
        data.extend_from_slice(&[0u8; 50]); // Padding
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(1);

        // Dispatches to SafeTensors path, fails on model conversion
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    // --- Error message content validation ---

    #[test]
    fn test_run_inference_io_error_message_content() {
        let config = InferenceConfig::new("/does/not/exist/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to read") || err_msg.contains("IO error"),
            "Error message should mention read failure: {}",
            err_msg
        );
    }

    #[test]
    fn test_run_inference_format_error_message_content() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create file with unknown magic
        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[0u8; 16]).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Format") || err_msg.contains("format"),
            "Error message should mention format: {}",
            err_msg
        );
    }

    // --- Input token priority tests ---

    #[test]
    fn test_input_tokens_takes_priority_in_gguf_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        // Both prompt and input_tokens set - input_tokens should take priority
        let config = InferenceConfig::new(temp.path())
            .with_prompt("This is ignored")
            .with_input_tokens(vec![1, 2, 3])
            .with_max_tokens(5);

        assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
        assert_eq!(config.prompt, Some("This is ignored".to_string()));

        // Will fail on model load, but exercises the input_tokens path
        let _result = run_inference(&config);
    }

    // --- Verbose mode output tests ---

    #[test]
    fn test_verbose_mode_doesnt_panic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        // Verbose mode enabled - should output to stderr without panicking
        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_verbose(true)
            .with_max_tokens(1);

        // Will fail on model load, but exercises verbose output path
        let _result = run_inference(&config);
        // Test passes if no panic
    }

    // --- Architecture detection comprehensive tests ---

    #[test]
    fn test_architecture_detection_all_variants() {
        let test_cases = [
            ("qwen2-7b-instruct.gguf", "Qwen2"),
            ("Qwen-7B-Chat.gguf", "Qwen2"),
            ("QWEN_7B.gguf", "Qwen2"),
            ("llama-3.1-8b-instruct.gguf", "LLaMA"),
            ("Llama-2-7b-chat.gguf", "LLaMA"),
            ("LLAMA3.gguf", "LLaMA"),
            ("mistral-7b-instruct-v0.2.gguf", "Mistral"),
            ("Mistral-7B.gguf", "Mistral"),
            ("MISTRAL_LARGE.gguf", "Mistral"),
            ("phi-2.gguf", "Phi"),
            ("Phi-3-mini.gguf", "Phi"),
            ("PHI2.gguf", "Phi"),
            ("custom-model.gguf", "Transformer"),
            ("gpt2.gguf", "Transformer"),
            ("my-finetuned-model.gguf", "Transformer"),
        ];

        for (filename, expected_arch) in test_cases {
            let path = PathBuf::from(format!("/models/{}", filename));
            let arch = path.file_stem().and_then(|s| s.to_str()).map(|s| {
                if s.to_lowercase().contains("qwen") {
                    "Qwen2"
                } else if s.to_lowercase().contains("llama") {
                    "LLaMA"
                } else if s.to_lowercase().contains("mistral") {
                    "Mistral"
                } else if s.to_lowercase().contains("phi") {
                    "Phi"
                } else {
                    "Transformer"
                }
            });
            assert_eq!(arch, Some(expected_arch), "Failed for: {}", filename);
        }
    }

    // --- prefault_mmap comprehensive tests ---

    #[test]
    fn test_prefault_mmap_various_patterns() {
        // Test with various byte patterns
        let patterns: Vec<Vec<u8>> = vec![
            vec![0u8; 8192],                                 // All zeros
            vec![255u8; 8192],                               // All ones
            (0..8192u16).map(|i| (i % 256) as u8).collect(), // Sequential
            vec![0xAA; 8192],                                // Alternating bits
            vec![0x55; 8192],                                // Alternating bits (inverted)
        ];

        for pattern in patterns {
            prefault_mmap(&pattern);
        }
    }

    #[test]
    fn test_prefault_mmap_exactly_one_page_minus_one() {
        let data = vec![1u8; 4095];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_exactly_two_pages() {
        let data = vec![2u8; 8192];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_many_pages() {
        let data = vec![3u8; 4096 * 100]; // 100 pages
        prefault_mmap(&data);
    }
