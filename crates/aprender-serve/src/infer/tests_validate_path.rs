//! T-COV-95 Deep Coverage Bridge: infer/mod.rs
//!
//! Targets: validate_model_path, qtype_to_dtype_str, run_mock_inference,
//! clean_model_output, InferenceConfig builder chain, InferenceResult construction.

use super::*;
use std::path::Path;

// ============================================================================
// validate_model_path coverage
// ============================================================================

#[test]
fn test_validate_path_traversal_double_dot() {
    let path = Path::new("/tmp/../etc/passwd.gguf");
    let result = validate_model_path(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("traversal"), "got: {err}");
}

#[test]
fn test_validate_path_traversal_middle() {
    let path = Path::new("/home/user/../secret/model.gguf");
    let result = validate_model_path(path);
    assert!(result.is_err());
}

#[test]
fn test_validate_path_wrong_extension_txt() {
    let path = Path::new("/tmp/model.txt");
    // Create the file so it exists
    let _ = std::fs::write(path, b"fake");
    let result = validate_model_path(path);
    let _ = std::fs::remove_file(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("extension") || err.contains("Invalid"),
        "got: {err}"
    );
}

#[test]
fn test_validate_path_no_extension() {
    let path = Path::new("/tmp/modelfile");
    let _ = std::fs::write(path, b"fake");
    let result = validate_model_path(path);
    let _ = std::fs::remove_file(path);
    assert!(result.is_err());
}

#[test]
fn test_validate_path_nonexistent_file() {
    let path = Path::new("/tmp/nonexistent_model_12345.gguf");
    let result = validate_model_path(path);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not found") || err.contains("File"),
        "got: {err}"
    );
}

#[test]
fn test_validate_path_directory_not_file() {
    let path = Path::new("/tmp");
    // /tmp exists but isn't a .gguf - will fail on extension
    let result = validate_model_path(path);
    assert!(result.is_err());
}

#[test]
fn test_validate_path_valid_extensions() {
    // Test all valid extensions recognized by the validator
    for ext in &["gguf", "safetensors", "apr", "bin"] {
        let p = format!("/tmp/test_validate_{}.{}", ext, ext);
        let path = Path::new(&p);
        let _ = std::fs::write(path, b"fake model data here");
        let result = validate_model_path(path);
        let _ = std::fs::remove_file(path);
        // Should not fail on extension (may fail on other checks)
        if let Err(e) = &result {
            let msg = e.to_string();
            assert!(
                !msg.contains("extension"),
                "extension {} should be valid: {msg}",
                ext
            );
        }
    }
}

// ============================================================================
// qtype_to_dtype_str coverage (all 16 match arms)
// ============================================================================

#[test]
fn test_qtype_to_dtype_str_f32() {
    assert_eq!(qtype_to_dtype_str(0), "F32");
}

#[test]
fn test_qtype_to_dtype_str_f16() {
    assert_eq!(qtype_to_dtype_str(1), "F16");
}

#[test]
fn test_qtype_to_dtype_str_q4_0() {
    assert_eq!(qtype_to_dtype_str(2), "Q4_0");
}

#[test]
fn test_qtype_to_dtype_str_q4_1() {
    assert_eq!(qtype_to_dtype_str(3), "Q4_1");
}

#[test]
fn test_qtype_to_dtype_str_q5_0() {
    assert_eq!(qtype_to_dtype_str(6), "Q5_0");
}

#[test]
fn test_qtype_to_dtype_str_q5_1() {
    assert_eq!(qtype_to_dtype_str(7), "Q5_1");
}

#[test]
fn test_qtype_to_dtype_str_q8_0() {
    assert_eq!(qtype_to_dtype_str(8), "Q8_0");
}

#[test]
fn test_qtype_to_dtype_str_q8_1() {
    assert_eq!(qtype_to_dtype_str(9), "Q8_1");
}

#[test]
fn test_qtype_to_dtype_str_q2_k() {
    assert_eq!(qtype_to_dtype_str(10), "Q2_K");
}

#[test]
fn test_qtype_to_dtype_str_q3_k() {
    assert_eq!(qtype_to_dtype_str(11), "Q3_K");
}

#[test]
fn test_qtype_to_dtype_str_q4_k() {
    assert_eq!(qtype_to_dtype_str(12), "Q4_K");
}

#[test]
fn test_qtype_to_dtype_str_q5_k() {
    assert_eq!(qtype_to_dtype_str(13), "Q5_K");
}

#[test]
fn test_qtype_to_dtype_str_q6_k() {
    assert_eq!(qtype_to_dtype_str(14), "Q6_K");
}

#[test]
fn test_qtype_to_dtype_str_iq2_xxs() {
    assert_eq!(qtype_to_dtype_str(16), "IQ2_XXS");
}

#[test]
fn test_qtype_to_dtype_str_iq2_xs() {
    assert_eq!(qtype_to_dtype_str(17), "IQ2_XS");
}

#[test]
fn test_qtype_to_dtype_str_bf16() {
    assert_eq!(qtype_to_dtype_str(30), "BF16");
}

#[test]
fn test_qtype_to_dtype_str_unknown() {
    assert_eq!(qtype_to_dtype_str(99), "Unknown");
    assert_eq!(qtype_to_dtype_str(255), "Unknown");
    assert_eq!(qtype_to_dtype_str(4), "Unknown"); // gap value
    assert_eq!(qtype_to_dtype_str(5), "Unknown"); // gap value
    assert_eq!(qtype_to_dtype_str(15), "Unknown"); // gap value
}

// ============================================================================
// run_mock_inference coverage
// ============================================================================

#[test]
fn test_mock_inference_with_prompt() {
    let config = mock_config("Hello world test");
    let result = run_mock_inference(&config).unwrap();
    assert_eq!(result.format, "Mock");
    assert!(!result.used_gpu);
    assert!(result.text.contains("mock response for: Hello world test"));
    assert_eq!(result.input_token_count, 3); // 3 words
    assert!(result.generated_token_count > 0);
    assert!(result.tok_per_sec > 0.0);
    assert!((result.load_ms - 10.0).abs() < f64::EPSILON);
}

#[test]
fn test_mock_inference_with_input_tokens() {
    let config = InferenceConfig::new("/dev/null")
        .with_input_tokens(vec![1, 2, 3, 4, 5])
        .with_max_tokens(8)
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    assert_eq!(result.input_token_count, 5);
    assert_eq!(result.generated_token_count, 8);
    // Input tokens + generated tokens
    assert_eq!(result.tokens.len(), 13);
    // Generated tokens start at 100
    assert_eq!(result.tokens[5], 100);
    assert_eq!(result.tokens[12], 107);
}

#[test]
fn test_mock_inference_no_prompt_no_tokens() {
    let config = InferenceConfig::new("/dev/null")
        .with_max_tokens(4)
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    assert_eq!(result.input_token_count, 1); // BOS token
    assert!(result.text.contains("(no prompt)"));
}

#[test]
fn test_mock_inference_negative_temperature_error() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_temperature(-1.0)
        .with_max_tokens(4)
        .with_mock_backend();
    let result = run_mock_inference(&config);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("temperature") || err.contains("negative"),
        "got: {err}"
    );
}

#[test]
fn test_mock_inference_zero_max_tokens_error() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(0)
        .with_mock_backend();
    let result = run_mock_inference(&config);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("max_tokens") || err.contains("> 0"),
        "got: {err}"
    );
}

#[test]
fn test_mock_inference_with_trace_output() {
    let trace_path = std::path::PathBuf::from("/tmp/test_mock_trace_output_cov95.json");
    let _ = std::fs::remove_file(&trace_path);

    let config = InferenceConfig::new("/dev/null")
        .with_prompt("trace test")
        .with_max_tokens(4)
        .with_trace_output(trace_path.clone())
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    assert!(result.generated_token_count > 0);

    // Verify trace file was written
    assert!(trace_path.exists(), "trace file should be created");
    let content = std::fs::read_to_string(&trace_path).unwrap();
    assert!(content.contains("\"mock\": true"));
    assert!(content.contains("\"input_tokens\""));
    let _ = std::fs::remove_file(&trace_path);
}

#[test]
fn test_mock_inference_max_tokens_capped_at_32() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("test")
        .with_max_tokens(100)
        .with_mock_backend();
    let result = run_mock_inference(&config).unwrap();
    // Mock caps at 32 tokens
    assert_eq!(result.generated_token_count, 32);
}

// ============================================================================
// run_inference dispatches to mock
// ============================================================================

#[test]
fn test_run_inference_mock_dispatch() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("dispatch test")
        .with_max_tokens(4)
        .with_mock_backend();
    let result = run_inference(&config).unwrap();
    assert_eq!(result.format, "Mock");
}

// ============================================================================
// clean_model_output coverage
// ============================================================================

#[test]
fn test_clean_model_output_im_start_assistant() {
    let raw = "<|im_start|>assistant\nHello there!";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Hello there!");
}

#[test]
fn test_clean_model_output_im_end() {
    let raw = "Hello<|im_end|>";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Hello");
}

#[test]
fn test_clean_model_output_endoftext() {
    let raw = "Hello world<|endoftext|>";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Hello world");
}

#[test]
fn test_clean_model_output_all_markers() {
    let raw = "<|im_start|>assistant\nHello<|im_end|><|endoftext|>";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Hello");
}

#[test]
fn test_clean_model_output_no_markers() {
    let raw = "Just plain text";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Just plain text");
}

#[test]
fn test_clean_model_output_empty() {
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
fn test_clean_model_output_im_start_without_newline() {
    let raw = "<|im_start|>assistantHello";
    let cleaned = clean_model_output(raw);
    assert_eq!(cleaned, "Hello");
}

// ============================================================================
// InferenceConfig builder chain coverage
// ============================================================================

#[test]
fn test_inference_config_full_builder_chain() {
    let config = InferenceConfig::new("/path/to/model.gguf")
        .with_prompt("test prompt")
        .with_max_tokens(64)
        .with_temperature(0.7)
        .with_top_k(40)
        .without_gpu()
        .with_verbose(true)
        .with_trace(true)
        .with_trace_output("/tmp/trace.json");

    assert_eq!(config.model_path.to_str().unwrap(), "/path/to/model.gguf");
    assert_eq!(config.prompt.as_deref(), Some("test prompt"));
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 40);
    assert!(config.no_gpu);
    assert!(config.verbose);
    assert!(config.trace);
    assert_eq!(
        config.trace_output.as_ref().unwrap().to_str().unwrap(),
        "/tmp/trace.json"
    );
}

#[test]
fn test_inference_config_defaults() {
    let config = InferenceConfig::new("model.gguf");
    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 1);
    assert!(!config.no_gpu);
    assert!(!config.verbose);
    assert!(!config.trace);
    assert!(!config.trace_verbose);
    assert!(config.trace_output.is_none());
    assert!(config.trace_steps.is_none());
    assert!(config.prompt.is_none());
    assert!(config.input_tokens.is_none());
    assert!(!config.use_mock_backend);
}

#[test]
fn test_inference_config_with_input_tokens() {
    let config = InferenceConfig::new("model.gguf").with_input_tokens(vec![1, 2, 3]);
    assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
}

#[test]
fn test_inference_config_with_mock_backend() {
    let config = InferenceConfig::new("model.gguf").with_mock_backend();
    assert!(config.use_mock_backend);
}

#[test]
fn test_inference_config_debug_impl() {
    let config = InferenceConfig::new("model.gguf").with_prompt("test");
    let debug = format!("{:?}", config);
    assert!(debug.contains("model.gguf"));
    assert!(debug.contains("test"));
}

#[test]
fn test_inference_config_clone_impl() {
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("test")
        .with_max_tokens(64);
    let cloned = config.clone();
    assert_eq!(cloned.max_tokens, 64);
    assert_eq!(cloned.prompt.as_deref(), Some("test"));
}

include!("tests_inference.rs");
