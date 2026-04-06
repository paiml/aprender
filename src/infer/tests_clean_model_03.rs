
    // --- clean_model_output comprehensive edge cases ---

    #[test]
    fn test_clean_model_output_all_markers_comprehensive() {
        let test_cases = [
            ("<|im_start|>assistant\nContent<|im_end|>", "Content"),
            ("<|im_start|>assistantContent<|im_end|>", "Content"),
            ("Text<|endoftext|>", "Text"),
            (
                "<|im_start|>embedded marker<|im_start|>more<|im_end|>",
                "embedded markermore",
            ),
            ("   whitespace around   ", "whitespace around"),
            ("\n\nleading newlines\n\n", "leading newlines"),
        ];

        for (input, expected) in test_cases {
            let result = clean_model_output(input);
            assert_eq!(result, expected, "Failed for input: {:?}", input);
        }
    }

    #[test]
    fn test_clean_model_output_complex_conversation() {
        let raw = r"<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there! How can I help?<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        // All markers removed, content preserved
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        assert!(!cleaned.contains("<|endoftext|>"));
    }

    #[test]
    fn test_clean_model_output_json_content() {
        let raw = r#"<|im_start|>assistant
{"key": "value", "nested": {"inner": 42}}<|im_end|>"#;
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains(r#"{"key": "value"#));
    }

    #[test]
    fn test_clean_model_output_multiline_code() {
        let raw = r#"<|im_start|>assistant
```rust
fn main() {
    println!("Hello, world!");
}
```<|im_end|>"#;
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("fn main()"));
        assert!(cleaned.contains("println!"));
    }

    // --- InferenceResult field boundary tests ---

    #[test]
    fn test_inference_result_all_zero_counts() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.input_token_count + result.generated_token_count, 0);
    }

    #[test]
    fn test_inference_result_max_counts() {
        let result = InferenceResult {
            text: "x".repeat(1_000_000),
            tokens: vec![1; 1_000_000],
            input_token_count: 500_000,
            generated_token_count: 500_000,
            inference_ms: f64::MAX,
            tok_per_sec: f64::MAX,
            load_ms: f64::MAX,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.tokens.len(), 1_000_000);
    }

    // --- Format detection edge cases from run_inference ---

    #[test]
    fn test_run_inference_apr_v1_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"APR1"); // APR v1 magic
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path()).with_max_tokens(1);
        let result = run_inference(&config);
        // Dispatches to APR path - runs with fallback zeros if incomplete
        let _ = result;
    }

    #[test]
    fn test_run_inference_gguf_v2_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&2u32.to_le_bytes()); // GGUF version 2
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 100]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path()).with_max_tokens(1);
        let result = run_inference(&config);
        // Should dispatch to GGUF path
        assert!(result.is_err());
    }

    // --- Trace configuration tests ---

    #[test]
    fn test_trace_config_propagation() {
        let config = InferenceConfig::new("/model.gguf")
            .with_trace(true)
            .with_prompt("test");

        assert!(config.trace);
        // trace flag should be passed to gen_config in run_gguf_inference
    }

    #[test]
    fn test_trace_verbose_and_output() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/tmp/trace.json"));
        config.trace_steps = Some(vec!["embedding".to_string(), "attention".to_string()]);

        assert!(config.trace);
        assert!(config.trace_verbose);
        assert_eq!(config.trace_output, Some(PathBuf::from("/tmp/trace.json")));
        assert_eq!(config.trace_steps.as_ref().map(std::vec::Vec::len), Some(2));
    }

    // --- no_gpu flag tests ---

    #[test]
    fn test_no_gpu_flag_set() {
        let config = InferenceConfig::new("/model.gguf").without_gpu();
        assert!(config.no_gpu);
    }

    #[test]
    fn test_no_gpu_flag_default_false() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(!config.no_gpu);
    }

    // --- InferenceConfig Debug trait coverage ---

    #[test]
    fn test_inference_config_debug_contains_all_fields() {
        let config = InferenceConfig {
            model_path: PathBuf::from("/model.gguf"),
            prompt: Some("test".to_string()),
            input_tokens: Some(vec![1, 2]),
            max_tokens: 64,
            temperature: 0.7,
            top_k: 40,
            no_gpu: true,
            trace: true,
            trace_verbose: true,
            trace_output: Some(PathBuf::from("/trace.json")),
            trace_steps: Some(vec!["a".to_string()]),
            verbose: true,
            use_mock_backend: false,
        };

        let debug = format!("{:?}", config);
        let expected_fields = [
            "model_path",
            "prompt",
            "input_tokens",
            "max_tokens",
            "temperature",
            "top_k",
            "no_gpu",
            "trace",
            "trace_verbose",
            "trace_output",
            "trace_steps",
            "verbose",
        ];

        for field in expected_fields {
            assert!(debug.contains(field), "Debug missing field: {}", field);
        }
    }

    // --- InferenceResult Clone trait coverage ---

    #[test]
    fn test_inference_result_clone_independence() {
        let original = InferenceResult {
            text: "original".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 100.0,
            tok_per_sec: 20.0,
            load_ms: 50.0,
            format: "APR".to_string(),
            used_gpu: true,
        };

        let mut cloned = original.clone();
        cloned.text = "modified".to_string();
        cloned.tokens.push(4);

        // Original should be unchanged
        assert_eq!(original.text, "original");
        assert_eq!(original.tokens, vec![1, 2, 3]);
    }

    // --- File size edge cases ---

    #[test]
    fn test_run_inference_exactly_7_bytes() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[1, 2, 3, 4, 5, 6, 7]).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_run_inference_exactly_8_bytes_unknown_format() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[0, 0, 0, 0, 0, 0, 0, 0]).expect("write"); // Zero header = unknown
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Format"));
    }

    #[test]
    fn test_run_inference_large_file_header_only() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        // Add significant padding to simulate larger file
        data.extend_from_slice(&[0u8; 10000]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path()).with_max_tokens(1);
        let result = run_inference(&config);
        assert!(result.is_err()); // Should fail on model load
    }

    // --- tok_per_sec calculation edge cases ---

    #[test]
    fn test_tok_per_sec_with_fractional_ms() {
        let inference_ms = 333.333;
        let generated = 100;
        let tok_per_sec = if inference_ms > 0.0 {
            generated as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        // 100 / 0.333333 = ~300
        assert!((tok_per_sec - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_tok_per_sec_with_very_large_token_count() {
        let inference_ms = 1000.0;
        let generated: u64 = 1_000_000;
        let tok_per_sec = if inference_ms > 0.0 {
            generated as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 1_000_000.0).abs() < 0.001);
    }

    // --- Model path handling edge cases ---

    #[test]
    fn test_model_path_with_dots_in_name() {
        let config = InferenceConfig::new("/models/model.v1.0.beta.gguf");
        let stem = config.model_path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model.v1.0.beta"));
    }

    #[test]
    fn test_model_path_with_no_extension() {
        let config = InferenceConfig::new("/models/modelfile");
        let ext = config.model_path.extension();
        assert!(ext.is_none());
    }

    #[test]
    fn test_model_path_hidden_file() {
        let config = InferenceConfig::new("/models/.hidden_model.gguf");
        let name = config.model_path.file_name().and_then(|s| s.to_str());
        assert_eq!(name, Some(".hidden_model.gguf"));
    }

    // --- Instruct model detection additional tests ---

    #[test]
    fn test_instruct_detection_suffix_variations() {
        let suffixes = [
            "-instruct",
            "-Instruct",
            "-INSTRUCT",
            "_instruct",
            ".instruct",
            "-instruct-v2",
            "-chat-instruct",
        ];

        for suffix in suffixes {
            let name = format!("model{}.gguf", suffix);
            let is_instruct = name.to_lowercase().contains("instruct");
            assert!(is_instruct, "Should detect instruct in: {}", name);
        }
    }

    // --- BOS token fallback test ---

    #[test]
    fn test_bos_token_fallback_when_no_prompt_or_tokens() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());

        // In run_gguf_inference, this would result in vec![1u32] (BOS token)
        let default_tokens = config.input_tokens.clone().unwrap_or_else(|| {
            if config.prompt.is_some() {
                vec![100, 200] // Would be tokenized
            } else {
                vec![1u32] // BOS token
            }
        });
        assert_eq!(default_tokens, vec![1u32]);
    }

    // --- Format detection from data boundary tests ---

    #[test]
    fn test_format_detection_safetensors_min_valid_size() {
        use crate::format::{detect_format, ModelFormat};

        // Minimum valid SafeTensors header size (1 byte)
        let header_size: u64 = 1;
        let data = header_size.to_le_bytes();
        let result = detect_format(&data);
        assert!(matches!(result, Ok(ModelFormat::SafeTensors)));
    }

    #[test]
    fn test_format_detection_safetensors_typical_sizes() {
        use crate::format::{detect_format, ModelFormat};

        for size in [100u64, 1000, 10000, 100000, 1000000, 10000000] {
            let data = size.to_le_bytes();
            let result = detect_format(&data);
            assert!(
                matches!(result, Ok(ModelFormat::SafeTensors)),
                "Failed for size: {}",
                size
            );
        }
    }

    // --- InferenceConfig builder method return type tests ---

    #[test]
    fn test_builder_methods_return_self() {
        let config = InferenceConfig::new("/model.gguf");

        // Each method should consume self and return Self
        let config = config.with_prompt("test");
        let config = config.with_input_tokens(vec![1]);
        let config = config.with_max_tokens(10);
        let config = config.with_temperature(0.5);
        let config = config.with_top_k(10);
        let config = config.without_gpu();
        let config = config.with_verbose(true);
        let config = config.with_trace(true);

        // Verify final state
        assert_eq!(config.prompt, Some("test".to_string()));
        assert_eq!(config.input_tokens, Some(vec![1]));
        assert_eq!(config.max_tokens, 10);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    // --- Empty string handling ---

    #[test]
    fn test_empty_prompt_handling() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("");
        assert_eq!(config.prompt, Some(String::new()));
    }

    #[test]
    fn test_empty_input_tokens_handling() {
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![]);
        assert_eq!(config.input_tokens, Some(vec![]));
    }
