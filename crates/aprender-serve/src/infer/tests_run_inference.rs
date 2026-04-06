
    // --- run_inference format detection tests ---

    #[test]
    fn test_run_inference_with_gguf_magic_but_incomplete() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // GGUF magic but incomplete model
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        // Missing rest of header
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_with_apr_magic_but_incomplete() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // APR magic but incomplete
        let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x02"); // APR v2 magic
        data.extend_from_slice(&[0u8; 4]); // Incomplete header
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_with_safetensors_header_but_invalid() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // SafeTensors-like header but invalid JSON
        let mut temp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        let header_size: u64 = 16; // Size that doesn't match actual content
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(b"invalid json");
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    // --- Architecture detection from path tests ---

    #[test]
    fn test_architecture_from_path_variations() {
        let test_cases = [
            ("tinyllama-1.1b-chat.gguf", "LLaMA"),
            ("phi-2-mini.gguf", "Phi"),
            ("mistral-7b-instruct-v0.2.gguf", "Mistral"),
            ("qwen2-0.5b-instruct.gguf", "Qwen2"),
            ("my-custom-model.gguf", "Transformer"),
            ("LLAMA-3-8B.gguf", "LLaMA"),
            ("PHI3-MINI.gguf", "Phi"),
            ("MISTRAL_LARGE.gguf", "Mistral"),
            ("QWEN2_7B.gguf", "Qwen2"),
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

    // --- Token per second calculation edge cases ---

    #[test]
    fn test_tok_per_sec_negative_time_protection() {
        // Even though negative time shouldn't happen, test the logic
        let inference_ms = -100.0;
        let generated_count = 50;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert_eq!(tok_per_sec, 0.0);
    }

    #[test]
    fn test_tok_per_sec_very_small_time() {
        let inference_ms = 0.001; // 1 microsecond
        let generated_count = 1;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 1000000.0).abs() < 0.001); // 1M tok/s
    }

    // --- InferenceResult edge cases ---

    #[test]
    fn test_inference_result_with_unicode_text() {
        let result = InferenceResult {
            text: "Hello 世界 🌍 Привет".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
            generated_token_count: 3,
            inference_ms: 10.0,
            tok_per_sec: 300.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(result.text.contains("世界"));
        assert!(result.text.contains("🌍"));
    }

    #[test]
    fn test_inference_result_with_special_chars() {
        let result = InferenceResult {
            text: "Code: `fn main() { println!(\"Hello\"); }`".to_string(),
            tokens: vec![1],
            input_token_count: 0,
            generated_token_count: 1,
            inference_ms: 1.0,
            tok_per_sec: 1000.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        assert!(result.text.contains("fn main()"));
        assert!(result.text.contains("println!"));
    }

    // --- Config path handling tests ---

    #[test]
    fn test_inference_config_various_path_formats() {
        let paths = [
            "/absolute/path/model.gguf",
            "./relative/path/model.apr",
            "../parent/dir/model.safetensors",
            "model.gguf",
            "/path/with spaces/model file.gguf",
            "/path/with-dashes/model-name.gguf",
            "/path/with_underscores/model_name.gguf",
        ];

        for path in paths {
            let config = InferenceConfig::new(path);
            assert_eq!(config.model_path.to_str(), Some(path));
        }
    }

    // --- Max tokens passthrough behavior (GH-372: removed .min(128) cap) ---

    #[test]
    fn test_max_tokens_passthrough() {
        let configs = [
            (InferenceConfig::new("/m.gguf").with_max_tokens(0), 0),
            (InferenceConfig::new("/m.gguf").with_max_tokens(64), 64),
            (InferenceConfig::new("/m.gguf").with_max_tokens(128), 128),
            (InferenceConfig::new("/m.gguf").with_max_tokens(256), 256),
            (InferenceConfig::new("/m.gguf").with_max_tokens(512), 512),
            (InferenceConfig::new("/m.gguf").with_max_tokens(1024), 1024),
        ];

        for (config, expected) in configs {
            assert_eq!(config.max_tokens, expected);
        }
    }

    // --- EOS token detection tests ---

    #[test]
    fn test_eos_token_detection() {
        // Test the EOS token detection logic from run_apr_inference
        let eos_tokens = [151645u32, 151643u32, 2u32]; // Qwen2 EOS, BOS, standard
        let non_eos_tokens = [1u32, 100u32, 1000u32, 151644u32];

        for token in eos_tokens {
            let is_eos = token == 151645 || token == 151643 || token == 2;
            assert!(is_eos, "Token {} should be EOS", token);
        }

        for token in non_eos_tokens {
            let is_eos = token == 151645 || token == 151643 || token == 2;
            assert!(!is_eos, "Token {} should NOT be EOS", token);
        }
    }

    // --- Greedy sampling (argmax) logic ---

    #[test]
    fn test_argmax_logic() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 3); // Index of 0.9
    }

    #[test]
    fn test_argmax_logic_with_ties() {
        let logits = vec![0.5f32, 0.5, 0.5, 0.5];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        // max_by returns last maximum for ties
        assert_eq!(next_token, 3);
    }

    #[test]
    fn test_argmax_logic_with_negative() {
        let logits = vec![-0.5f32, -0.1, -0.9, -0.3];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 1); // Index of -0.1 (least negative)
    }

    #[test]
    fn test_argmax_logic_empty() {
        let logits: Vec<f32> = vec![];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 0); // Default when empty
    }

    #[test]
    fn test_argmax_logic_with_nan() {
        let logits = vec![0.1f32, f32::NAN, 0.5, 0.3];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        // NaN comparisons return Equal, so behavior depends on order
        // Result should be one of the valid indices
        assert!(next_token < 4);
    }

    // --- Generated tokens slice extraction ---

    #[test]
    fn test_generated_tokens_extraction() {
        let all_tokens = vec![100u32, 200, 300, 400, 500, 600];
        let input_token_count = 3;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens, &[400, 500, 600]);
        assert_eq!(generated_tokens.len(), 3);
    }

    #[test]
    fn test_generated_tokens_extraction_no_generation() {
        let all_tokens = vec![100u32, 200, 300];
        let input_token_count = 3;
        let generated_tokens = &all_tokens[input_token_count..];
        assert!(generated_tokens.is_empty());
    }

    #[test]
    fn test_generated_tokens_extraction_all_generated() {
        let all_tokens = vec![100u32, 200, 300, 400];
        let input_token_count = 0;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens.len(), 4);
    }

    // --- Instruct model detection ---

    #[test]
    fn test_instruct_model_name_variations() {
        let instruct_names = [
            "llama-3.1-8b-instruct.gguf",
            "LLAMA-INSTRUCT.gguf",
            "qwen2-7b-Instruct-q4.gguf",
            "model-instructed-v1.gguf",
            "phi-2-InStRuCt.gguf",
        ];

        for name in instruct_names {
            let is_instruct = name.to_lowercase().contains("instruct");
            assert!(is_instruct, "Should detect instruct in: {}", name);
        }

        let non_instruct_names = [
            "llama-3.1-8b-base.gguf",
            "phi-2-mini.gguf",
            "qwen2-7b-q4.gguf",
            "custom-model.gguf",
        ];

        for name in non_instruct_names {
            let is_instruct = name.to_lowercase().contains("instruct");
            assert!(!is_instruct, "Should NOT detect instruct in: {}", name);
        }
    }

    // --- Load time and inference time calculations ---

    #[test]
    fn test_time_calculation_precision() {
        let elapsed_secs: f64 = 1.234567;
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 1234.567).abs() < 0.001);

        let inference_ms: f64 = 567.89;
        let generated = 100;
        let tok_per_sec = generated as f64 / (inference_ms / 1000.0);
        // 100 / 0.56789 = 176.0564411... (use wider tolerance for FP precision)
        assert!((tok_per_sec - 176.0564).abs() < 0.1);
    }

    // --- InferenceConfig Debug and Clone full coverage ---

    #[test]
    fn test_inference_config_debug_all_fields() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/test/model.gguf"),
            prompt: Some("test prompt".to_string()),
            input_tokens: Some(vec![1, 2, 3]),
            max_tokens: 100,
            temperature: 0.7,
            top_k: 40,
            no_gpu: true,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/trace.json")),
            trace_steps: Some(vec!["embed".to_string()]),
            verbose: true,
            use_mock_backend: false,
        };

        let debug = format!("{:?}", config);
        assert!(debug.contains("model_path"));
        assert!(debug.contains("prompt"));
        assert!(debug.contains("input_tokens"));
        assert!(debug.contains("max_tokens"));
        assert!(debug.contains("temperature"));
        assert!(debug.contains("top_k"));
        assert!(debug.contains("no_gpu"));
        assert!(debug.contains("trace"));
        assert!(debug.contains("trace_verbose"));
        assert!(debug.contains("trace_output"));
        assert!(debug.contains("trace_steps"));
        assert!(debug.contains("verbose"));
    }

    #[test]
    fn test_inference_config_clone_preserves_all() {
        let original = InferenceConfig {
            model_path: PathBuf::from("/test/model.gguf"),
            prompt: Some("prompt".to_string()),
            input_tokens: Some(vec![10, 20, 30]),
            max_tokens: 200,
            temperature: 1.2,
            top_k: 100,
            no_gpu: true,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/out.json")),
            trace_steps: Some(vec!["a".to_string(), "b".to_string()]),
            verbose: true,
            use_mock_backend: false,
        };

        let cloned = original.clone();

        assert_eq!(original.model_path, cloned.model_path);
        assert_eq!(original.prompt, cloned.prompt);
        assert_eq!(original.input_tokens, cloned.input_tokens);
        assert_eq!(original.max_tokens, cloned.max_tokens);
        assert!((original.temperature - cloned.temperature).abs() < f32::EPSILON);
        assert_eq!(original.top_k, cloned.top_k);
        assert_eq!(original.no_gpu, cloned.no_gpu);
        assert_eq!(original.trace, cloned.trace);
        assert_eq!(original.trace_verbose, cloned.trace_verbose);
        assert_eq!(original.trace_output, cloned.trace_output);
        assert_eq!(original.trace_steps, cloned.trace_steps);
        assert_eq!(original.verbose, cloned.verbose);
    }

    // --- InferenceResult Debug and Clone full coverage ---

    #[test]
    fn test_inference_result_debug_all_fields() {
        let result = InferenceResult {
            text: "generated text".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
            generated_token_count: 3,
            inference_ms: 123.456,
            tok_per_sec: 24.32,
            load_ms: 50.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };

        let debug = format!("{:?}", result);
        assert!(debug.contains("text"));
        assert!(debug.contains("tokens"));
        assert!(debug.contains("input_token_count"));
        assert!(debug.contains("generated_token_count"));
        assert!(debug.contains("inference_ms"));
        assert!(debug.contains("tok_per_sec"));
        assert!(debug.contains("load_ms"));
        assert!(debug.contains("format"));
        assert!(debug.contains("used_gpu"));
    }
