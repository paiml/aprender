//! T-COV-95 Phase 55: Extended coverage for infer/mod.rs
//!
//! Covers:
//! - run_mock_inference: all code paths (prompt, tokens, no prompt, trace, errors)
//! - mock_config: helper function
//! - InferenceConfig builder: with_input_tokens
//! - InferenceResult: Clone, Debug
//! - log_cpu_backend: verbose/non-verbose paths
//! - model_has_legacy_quant: edge cases (needs mock-able struct)
//! - clean_model_output: additional patterns
//! - prepare_tokens with raw token input
//! - prepare_tokens with no prompt

use super::*;

// ============================================================================
// run_mock_inference: full prompt path
// ============================================================================

#[test]
fn test_mock_inference_with_prompt() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("Hello world test")
        .with_max_tokens(8)
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("mock inference should succeed");
    assert_eq!(result.format, "Mock");
    assert!(!result.used_gpu);
    assert!(result.text.contains("mock response for: Hello world test"));
    assert_eq!(result.input_token_count, 3); // "Hello", "world", "test"
    assert_eq!(result.generated_token_count, 8);
    assert!(result.load_ms > 0.0);
    assert!(result.inference_ms > 0.0);
    assert!(result.tok_per_sec > 0.0);
}

#[test]
fn test_mock_inference_with_tokens() {
    let config = InferenceConfig::new("/dev/null")
        .with_input_tokens(vec![10, 20, 30, 40])
        .with_max_tokens(4)
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("mock inference should succeed");
    assert_eq!(result.input_token_count, 4);
    assert_eq!(result.generated_token_count, 4);
    // All tokens = input + generated
    assert_eq!(result.tokens.len(), 8);
    assert_eq!(result.tokens[0], 10);
    assert_eq!(result.tokens[4], 100); // first generated token
}

#[test]
fn test_mock_inference_no_prompt() {
    let config = InferenceConfig::new("/dev/null")
        .with_max_tokens(5)
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("mock inference should succeed");
    assert_eq!(result.input_token_count, 1); // BOS token
    assert_eq!(result.tokens[0], 1); // BOS
    assert!(result.text.contains("(no prompt)"));
}

#[test]
fn test_mock_inference_negative_temperature() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_temperature(-1.0)
        .with_max_tokens(4)
        .with_mock_backend();

    let result = run_mock_inference(&config);
    assert!(result.is_err(), "Negative temperature should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("temperature"),
        "Error should mention temperature: {}",
        err
    );
}

#[test]
fn test_mock_inference_zero_max_tokens() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(0)
        .with_mock_backend();

    let result = run_mock_inference(&config);
    assert!(result.is_err(), "Zero max_tokens should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("max_tokens"),
        "Error should mention max_tokens: {}",
        err
    );
}

#[test]
fn test_mock_inference_with_trace_output() {
    let trace_path = std::env::temp_dir().join("test_mock_trace_infer_10.json");
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("trace test")
        .with_max_tokens(4)
        .with_trace_output(trace_path.clone())
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("mock inference should succeed");
    assert!(result.text.contains("mock response"));

    // Verify trace file was written
    let trace_content = std::fs::read_to_string(&trace_path).expect("should read trace file");
    assert!(trace_content.contains("\"mock\": true"));
    assert!(trace_content.contains("\"input_tokens\""));

    std::fs::remove_file(&trace_path).ok();
}

#[test]
fn test_mock_inference_max_tokens_capped() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(1000) // over 32, should be capped
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("mock inference should succeed");
    assert_eq!(result.generated_token_count, 32); // capped at min(max_tokens, 32)
}

// ============================================================================
// mock_config helper
// ============================================================================

#[test]
fn test_mock_config_helper() {
    let config = mock_config("Hello");
    assert_eq!(config.prompt, Some("Hello".to_string()));
    assert_eq!(config.max_tokens, 16);
    assert!(config.use_mock_backend);
    assert_eq!(config.model_path, std::path::PathBuf::from("/dev/null"));
}

// ============================================================================
// run_inference with mock backend
// ============================================================================

#[test]
fn test_run_inference_mock_backend() {
    let config = mock_config("Testing run_inference mock path");
    let result = run_inference(&config).expect("should use mock backend");
    assert_eq!(result.format, "Mock");
    assert!(result.text.contains("mock response"));
}

// ============================================================================
// InferenceConfig builder: with_input_tokens
// ============================================================================

#[test]
fn test_config_with_input_tokens() {
    let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![1, 2, 3]);
    assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
    assert!(config.prompt.is_none());
}

#[test]
fn test_config_with_input_tokens_and_prompt() {
    // Both can be set; input_tokens takes precedence in prepare_tokens
    let config = InferenceConfig::new("/model.gguf")
        .with_prompt("Hello")
        .with_input_tokens(vec![1, 2, 3]);
    assert_eq!(config.prompt, Some("Hello".to_string()));
    assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
}

// ============================================================================
// InferenceResult Debug/Clone
// ============================================================================

#[test]
fn test_inference_result_debug() {
    let result = InferenceResult {
        text: "test".to_string(),
        tokens: vec![1],
        input_token_count: 1,
        generated_token_count: 0,
        inference_ms: 0.0,
        tok_per_sec: 0.0,
        load_ms: 0.0,
        format: "Mock".to_string(),
        used_gpu: false,
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("InferenceResult"));
    assert!(debug.contains("Mock"));
}

#[test]
fn test_inference_result_clone() {
    let result = InferenceResult {
        text: "hello".to_string(),
        tokens: vec![1, 2, 3],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 100.0,
        tok_per_sec: 20.0,
        load_ms: 50.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };
    let cloned = result.clone();
    assert_eq!(cloned.text, "hello");
    assert_eq!(cloned.tokens.len(), 3);
    assert!(cloned.used_gpu);
    assert_eq!(cloned.format, "GGUF");
}

// ============================================================================
// clean_model_output: more edge cases
// ============================================================================

#[test]
fn test_clean_model_output_whitespace_only_after_clean() {
    let cleaned = clean_model_output("  <|im_start|>assistant\n  <|im_end|>  ");
    assert_eq!(cleaned, "");
}

#[test]
fn test_clean_model_output_multiple_markers() {
    let cleaned = clean_model_output(
        "<|im_start|>assistant\nHello<|im_end|><|im_start|>assistant\nWorld<|im_end|>",
    );
    assert!(cleaned.contains("Hello"));
    assert!(cleaned.contains("World"));
}

#[test]
fn test_clean_model_output_im_start_alone() {
    let cleaned = clean_model_output("<|im_start|>leftover text");
    assert_eq!(cleaned, "leftover text");
}

#[test]
fn test_clean_model_output_no_markers() {
    let cleaned = clean_model_output("Just regular text with no markers");
    assert_eq!(cleaned, "Just regular text with no markers");
}

#[test]
fn test_clean_model_output_nested_markers() {
    let cleaned = clean_model_output("<|im_start|><|im_end|><|endoftext|>");
    assert_eq!(cleaned, "");
}

// ============================================================================
// PreparedTokens
// ============================================================================

#[test]
fn test_prepared_tokens_clone() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![10, 20, 30]);
    let prepared = prepare_tokens(&config, &ModelFormat::Gguf).expect("should prepare tokens");
    let cloned = prepared.clone();
    assert_eq!(cloned.tokens(), prepared.tokens());
    assert_eq!(cloned.input_count(), prepared.input_count());
}

#[test]
fn test_prepared_tokens_debug() {
    use crate::format::ModelFormat;

    let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![1, 2]);
    let prepared = prepare_tokens(&config, &ModelFormat::Gguf).expect("should prepare tokens");
    let debug = format!("{:?}", prepared);
    assert!(debug.contains("PreparedTokens"));
}

// ============================================================================
// InferenceConfig defaults
// ============================================================================

#[test]
fn test_inference_config_defaults() {
    let config = InferenceConfig::new("/model.gguf");
    assert_eq!(config.max_tokens, 32);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(!config.no_gpu);
    assert!(!config.trace);
    assert!(!config.trace_verbose);
    assert!(config.trace_output.is_none());
    assert!(config.trace_steps.is_none());
    assert!(!config.verbose);
    assert!(!config.use_mock_backend);
    assert!(config.prompt.is_none());
    assert!(config.input_tokens.is_none());
}

// ============================================================================
// VALID_MODEL_EXTENSIONS
// ============================================================================

#[test]
fn test_valid_model_extensions_list() {
    assert!(VALID_MODEL_EXTENSIONS.contains(&"gguf"));
    assert!(VALID_MODEL_EXTENSIONS.contains(&"safetensors"));
    assert!(VALID_MODEL_EXTENSIONS.contains(&"apr"));
    assert!(VALID_MODEL_EXTENSIONS.contains(&"bin"));
    assert!(VALID_MODEL_EXTENSIONS.contains(&"json"));
    assert!(!VALID_MODEL_EXTENSIONS.contains(&"txt"));
    assert!(!VALID_MODEL_EXTENSIONS.contains(&"exe"));
}

// ============================================================================
// log_cpu_backend
// ============================================================================

#[test]
fn test_log_cpu_backend_no_panic() {
    // These should not panic even if they write to stderr
    log_cpu_backend(false, false);
    log_cpu_backend(false, true);
    log_cpu_backend(true, false);
    log_cpu_backend(true, true);
}
