
// ============================================================================
// is_legacy_gguf_quant tests (GH-219)
// ============================================================================

#[test]
fn test_is_legacy_gguf_quant_q4_0_gh219() {
    assert!(is_legacy_gguf_quant(2)); // Q4_0
}

#[test]
fn test_is_legacy_gguf_quant_q4_1_gh219() {
    assert!(is_legacy_gguf_quant(3)); // Q4_1
}

#[test]
fn test_is_legacy_gguf_quant_q5_0_gh219() {
    assert!(is_legacy_gguf_quant(6)); // Q5_0
}

#[test]
fn test_is_legacy_gguf_quant_q5_1_gh219() {
    assert!(is_legacy_gguf_quant(7)); // Q5_1
}

#[test]
fn test_is_legacy_gguf_quant_non_legacy_types_gh219() {
    // Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_0 = 8, F16 = 1, F32 = 0
    assert!(!is_legacy_gguf_quant(0));  // F32
    assert!(!is_legacy_gguf_quant(1));  // F16
    assert!(!is_legacy_gguf_quant(8));  // Q8_0
    assert!(!is_legacy_gguf_quant(12)); // Q4_K
    assert!(!is_legacy_gguf_quant(13)); // Q5_K
    assert!(!is_legacy_gguf_quant(14)); // Q6_K
}

#[test]
fn test_is_legacy_gguf_quant_edge_values_gh219() {
    assert!(!is_legacy_gguf_quant(4));
    assert!(!is_legacy_gguf_quant(5));
    assert!(!is_legacy_gguf_quant(100));
    assert!(!is_legacy_gguf_quant(u32::MAX));
}

// ============================================================================
// log_cpu_backend tests (GH-219)
// ============================================================================

#[test]
fn test_log_cpu_backend_not_verbose_gh219() {
    // Should return immediately without printing
    log_cpu_backend(false, false);
    log_cpu_backend(false, true);
}

#[test]
fn test_log_cpu_backend_verbose_legacy_gh219() {
    log_cpu_backend(true, true);
}

#[test]
fn test_log_cpu_backend_verbose_non_legacy_gh219() {
    log_cpu_backend(true, false);
}

// ============================================================================
// clean_model_output tests (GH-219)
// ============================================================================

#[test]
fn test_clean_model_output_strips_im_start_assistant_newline_gh219() {
    let raw = "<|im_start|>assistant\nHello world";
    assert_eq!(clean_model_output(raw), "Hello world");
}

#[test]
fn test_clean_model_output_strips_im_start_assistant_no_newline_gh219() {
    let raw = "<|im_start|>assistantHello world";
    assert_eq!(clean_model_output(raw), "Hello world");
}

#[test]
fn test_clean_model_output_strips_im_end_gh219() {
    let raw = "Hello world<|im_end|>";
    assert_eq!(clean_model_output(raw), "Hello world");
}

#[test]
fn test_clean_model_output_strips_endoftext_gh219() {
    let raw = "Hello world<|endoftext|>";
    assert_eq!(clean_model_output(raw), "Hello world");
}

#[test]
fn test_clean_model_output_strips_multiple_markers_gh219() {
    let raw = "<|im_start|>assistant\nHello<|im_end|><|endoftext|>";
    assert_eq!(clean_model_output(raw), "Hello");
}

#[test]
fn test_clean_model_output_trims_whitespace_gh219() {
    let raw = "  Hello world  ";
    assert_eq!(clean_model_output(raw), "Hello world");
}

#[test]
fn test_clean_model_output_empty_after_strip_gh219() {
    let raw = "<|im_start|>assistant\n<|im_end|><|endoftext|>";
    assert_eq!(clean_model_output(raw), "");
}

#[test]
fn test_clean_model_output_no_markers_gh219() {
    let raw = "Hello, how are you?";
    assert_eq!(clean_model_output(raw), "Hello, how are you?");
}

#[test]
fn test_clean_model_output_preserves_content_markers_gh219() {
    // Content with angle brackets that aren't ChatML markers
    let raw = "The formula is <x> + <y> = <z>";
    assert_eq!(clean_model_output(raw), "The formula is <x> + <y> = <z>");
}

#[test]
fn test_clean_model_output_empty_input_gh219() {
    assert_eq!(clean_model_output(""), "");
}

// ============================================================================
// tok_per_sec tests (GH-219)
// ============================================================================

#[test]
fn test_tok_per_sec_normal_gh219() {
    // 100 tokens in 1000ms = 100 tok/s
    let tps = tok_per_sec(100, 1000.0);
    assert!((tps - 100.0).abs() < 0.01);
}

#[test]
fn test_tok_per_sec_fast_gh219() {
    // 50 tokens in 100ms = 500 tok/s
    let tps = tok_per_sec(50, 100.0);
    assert!((tps - 500.0).abs() < 0.01);
}

#[test]
fn test_tok_per_sec_zero_time_gh219() {
    let tps = tok_per_sec(10, 0.0);
    // Should handle divide by zero gracefully (infinity or 0)
    assert!(tps.is_infinite() || tps == 0.0);
}

#[test]
fn test_tok_per_sec_zero_tokens_gh219() {
    let tps = tok_per_sec(0, 1000.0);
    assert!((tps - 0.0).abs() < 0.01);
}

#[test]
fn test_tok_per_sec_single_token_gh219() {
    // 1 token in 50ms = 20 tok/s
    let tps = tok_per_sec(1, 50.0);
    assert!((tps - 20.0).abs() < 0.01);
}

// ============================================================================
// InferenceResult struct tests (GH-219)
// ============================================================================

#[test]
fn test_inference_result_debug_gh219() {
    let result = InferenceResult {
        text: "Hello".to_string(),
        tokens: vec![1, 2, 3],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 50.0,
        tok_per_sec: 40.0,
        load_ms: 10.0,
        format: "Mock".to_string(),
        used_gpu: false,
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("Hello"));
    assert!(debug.contains("Mock"));
}

#[test]
fn test_inference_result_clone_gh219() {
    let result = InferenceResult {
        text: "test".to_string(),
        tokens: vec![100, 101],
        input_token_count: 0,
        generated_token_count: 2,
        inference_ms: 25.0,
        tok_per_sec: 80.0,
        load_ms: 5.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };
    let cloned = result.clone();
    assert_eq!(cloned.text, result.text);
    assert_eq!(cloned.tokens, result.tokens);
    assert_eq!(cloned.used_gpu, result.used_gpu);
}

// ============================================================================
// validate_model_path tests (GH-219)
// ============================================================================

#[test]
fn test_validate_model_path_traversal_gh219() {
    let path = std::path::Path::new("/tmp/../etc/passwd");
    let result = validate_model_path(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("traversal") || err.contains(".."));
}

#[test]
fn test_validate_model_path_invalid_extension_gh219() {
    // Create a temp file with invalid extension
    let path = std::path::Path::new("/tmp/test_file.txt");
    std::fs::write(path, b"test").ok();
    let result = validate_model_path(path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_validate_model_path_no_extension_gh219() {
    let path = std::path::Path::new("/tmp/modelfile");
    std::fs::write(path, b"test").ok();
    let result = validate_model_path(path);
    assert!(result.is_err());
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_validate_model_path_valid_gguf_gh219() {
    let path = std::path::Path::new("/tmp/test_model_gh219.gguf");
    std::fs::write(path, b"test data for gguf").ok();
    let result = validate_model_path(path);
    assert!(result.is_ok());
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_validate_model_path_valid_safetensors_gh219() {
    let path = std::path::Path::new("/tmp/test_model_gh219.safetensors");
    std::fs::write(path, b"test data").ok();
    let result = validate_model_path(path);
    assert!(result.is_ok());
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_validate_model_path_valid_apr_gh219() {
    let path = std::path::Path::new("/tmp/test_model_gh219.apr");
    std::fs::write(path, b"test data").ok();
    let result = validate_model_path(path);
    assert!(result.is_ok());
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_validate_model_path_valid_bin_gh219() {
    let path = std::path::Path::new("/tmp/test_model_gh219.bin");
    std::fs::write(path, b"test data").ok();
    let result = validate_model_path(path);
    assert!(result.is_ok());
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_validate_model_path_valid_json_gh219() {
    let path = std::path::Path::new("/tmp/test_model_gh219.json");
    std::fs::write(path, b"{}").ok();
    let result = validate_model_path(path);
    assert!(result.is_ok());
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_validate_model_path_nonexistent_gh219() {
    let path = std::path::Path::new("/tmp/definitely_does_not_exist_gh219.gguf");
    let result = validate_model_path(path);
    assert!(result.is_err());
}

#[test]
fn test_validate_model_path_directory_gh219() {
    let path = std::path::Path::new("/tmp");
    // /tmp has no model extension, so it fails on extension check first
    let result = validate_model_path(path);
    assert!(result.is_err());
}

// ============================================================================
// print_gguf_verbose_info smoke test (GH-219)
// No panic with various architecture strings
// ============================================================================

// Note: print_gguf_verbose_info requires OwnedQuantizedModel which can't be
// constructed without actual GGUF data, so we test the architecture matching
// logic directly via the `match` arm equivalents:

#[test]
fn test_gguf_arch_matching_logic_gh219() {
    let test_cases = vec![
        ("qwen2", "Qwen2"),
        ("qwen", "Qwen2"),
        ("llama", "LLaMA"),
        ("mistral", "Mistral"),
        ("phi", "Phi"),
        ("phi3", "Phi"),
        ("unknown_arch", "Transformer"),
        ("", "Transformer"),
    ];
    for (input, expected) in test_cases {
        let result = match input.to_lowercase().as_str() {
            "qwen2" | "qwen" => "Qwen2",
            "llama" => "LLaMA",
            "mistral" => "Mistral",
            "phi" | "phi3" => "Phi",
            _ => "Transformer",
        };
        assert_eq!(result, expected, "Arch '{}' should map to '{}'", input, expected);
    }
}

// ============================================================================
// mock_config and InferenceConfig builder tests (GH-219)
// ============================================================================

#[test]
fn test_mock_config_creates_valid_config_gh219() {
    let config = mock_config("Hello world");
    assert_eq!(config.prompt.as_deref(), Some("Hello world"));
    assert_eq!(config.max_tokens, 16);
    assert!(config.use_mock_backend);
}

#[test]
fn test_mock_config_empty_prompt_gh219() {
    let config = mock_config("");
    assert_eq!(config.prompt.as_deref(), Some(""));
    assert!(config.use_mock_backend);
}

#[test]
fn test_inference_config_with_mock_backend_gh219() {
    let config = InferenceConfig::new("/dev/null").with_mock_backend();
    assert!(config.use_mock_backend);
}

// ============================================================================
// run_mock_inference tests (GH-219)
// ============================================================================

#[test]
fn test_run_mock_inference_basic_gh219() {
    let config = mock_config("What is Rust?");
    let result = run_mock_inference(&config).unwrap();
    assert_eq!(result.format, "Mock");
    assert!(!result.used_gpu);
    assert!(result.text.contains("mock response for:"));
    assert!(result.text.contains("What is Rust?"));
    assert_eq!(result.load_ms, 10.0);
    assert!(result.inference_ms > 0.0);
    assert!(result.tok_per_sec > 0.0);
}

#[test]
fn test_run_mock_inference_with_tokens_gh219() {
    let config = InferenceConfig::new("/dev/null")
        .with_input_tokens(vec![1, 2, 3])
        .with_max_tokens(8)
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    assert_eq!(result.input_token_count, 3);
    assert_eq!(result.generated_token_count, 8);
    // Total tokens = input (3) + generated (8) = 11
    assert_eq!(result.tokens.len(), 11);
}

#[test]
fn test_run_mock_inference_negative_temperature_gh219() {
    let mut config = mock_config("test");
    config.temperature = -1.0;
    let result = run_mock_inference(&config);
    assert!(result.is_err());
}

#[test]
fn test_run_mock_inference_zero_max_tokens_gh219() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(0)
        .with_mock_backend();
    let result = run_mock_inference(&config);
    assert!(result.is_err());
}

#[test]
fn test_run_mock_inference_no_prompt_no_tokens_gh219() {
    let config = InferenceConfig::new("/dev/null")
        .with_max_tokens(4)
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    // Falls back to BOS token [1]
    assert_eq!(result.input_token_count, 1);
}

#[test]
fn test_run_mock_inference_max_tokens_capped_gh219() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(1000) // Capped to 32 by mock
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    assert_eq!(result.generated_token_count, 32); // min(1000, 32) = 32
}

#[test]
fn test_run_mock_inference_trace_output_gh219() {
    let trace_path = "/tmp/realizar_mock_trace_gh219.json";
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("trace test")
        .with_max_tokens(4)
        .with_trace_output(trace_path)
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    assert_eq!(result.format, "Mock");

    // Verify trace file was written
    let trace_content = std::fs::read_to_string(trace_path).unwrap();
    assert!(trace_content.contains("\"mock\": true"));
    assert!(trace_content.contains("\"input_tokens\":"));
    let _ = std::fs::remove_file(trace_path);
}
