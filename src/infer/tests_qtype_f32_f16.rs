//! T-COV-95 Synthetic Falsification: infer/mod.rs additional coverage
//!
//! Tests for qtype_to_dtype_str, InferenceConfig builder methods, and InferenceResult fields.

use super::*;
use std::path::PathBuf;

// ============================================================================
// qtype_to_dtype_str tests - all match arms
// ============================================================================

#[test]
fn test_qtype_f32() {
    assert_eq!(qtype_to_dtype_str(0), "F32");
}

#[test]
fn test_qtype_f16() {
    assert_eq!(qtype_to_dtype_str(1), "F16");
}

#[test]
fn test_qtype_q4_0() {
    assert_eq!(qtype_to_dtype_str(2), "Q4_0");
}

#[test]
fn test_qtype_q4_1() {
    assert_eq!(qtype_to_dtype_str(3), "Q4_1");
}

#[test]
fn test_qtype_q5_0() {
    assert_eq!(qtype_to_dtype_str(6), "Q5_0");
}

#[test]
fn test_qtype_q5_1() {
    assert_eq!(qtype_to_dtype_str(7), "Q5_1");
}

#[test]
fn test_qtype_q8_0() {
    assert_eq!(qtype_to_dtype_str(8), "Q8_0");
}

#[test]
fn test_qtype_q8_1() {
    assert_eq!(qtype_to_dtype_str(9), "Q8_1");
}

#[test]
fn test_qtype_q2_k() {
    assert_eq!(qtype_to_dtype_str(10), "Q2_K");
}

#[test]
fn test_qtype_q3_k() {
    assert_eq!(qtype_to_dtype_str(11), "Q3_K");
}

#[test]
fn test_qtype_q4_k() {
    assert_eq!(qtype_to_dtype_str(12), "Q4_K");
}

#[test]
fn test_qtype_q5_k() {
    assert_eq!(qtype_to_dtype_str(13), "Q5_K");
}

#[test]
fn test_qtype_q6_k() {
    assert_eq!(qtype_to_dtype_str(14), "Q6_K");
}

#[test]
fn test_qtype_iq2_xxs() {
    assert_eq!(qtype_to_dtype_str(16), "IQ2_XXS");
}

#[test]
fn test_qtype_iq2_xs() {
    assert_eq!(qtype_to_dtype_str(17), "IQ2_XS");
}

#[test]
fn test_qtype_bf16() {
    assert_eq!(qtype_to_dtype_str(30), "BF16");
}

#[test]
fn test_qtype_unknown() {
    assert_eq!(qtype_to_dtype_str(99), "Unknown");
    assert_eq!(qtype_to_dtype_str(255), "Unknown");
    assert_eq!(qtype_to_dtype_str(4), "Unknown");
    assert_eq!(qtype_to_dtype_str(5), "Unknown");
    assert_eq!(qtype_to_dtype_str(15), "Unknown");
}

// ============================================================================
// InferenceConfig builder tests
// ============================================================================

#[test]
fn test_inference_config_new() {
    let config = InferenceConfig::new("test.gguf");
    assert_eq!(config.model_path, PathBuf::from("test.gguf"));
    assert!(config.prompt.is_none());
    assert!(config.input_tokens.is_none());
    assert_eq!(config.max_tokens, 32);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(!config.no_gpu);
    assert!(!config.trace);
    assert!(!config.verbose);
}

#[test]
fn test_inference_config_with_prompt() {
    let config = InferenceConfig::new("test.gguf").with_prompt("Hello, world!");
    assert_eq!(config.prompt, Some("Hello, world!".to_string()));
}

#[test]
fn test_inference_config_with_input_tokens() {
    let config = InferenceConfig::new("test.gguf").with_input_tokens(vec![1, 2, 3, 4, 5]);
    assert_eq!(config.input_tokens, Some(vec![1, 2, 3, 4, 5]));
}

#[test]
fn test_inference_config_with_max_tokens() {
    let config = InferenceConfig::new("test.gguf").with_max_tokens(100);
    assert_eq!(config.max_tokens, 100);
}

#[test]
fn test_inference_config_with_temperature() {
    let config = InferenceConfig::new("test.gguf").with_temperature(0.7);
    assert!((config.temperature - 0.7).abs() < 0.001);
}

#[test]
fn test_inference_config_with_top_k() {
    let config = InferenceConfig::new("test.gguf").with_top_k(40);
    assert_eq!(config.top_k, 40);
}

#[test]
fn test_inference_config_without_gpu() {
    let config = InferenceConfig::new("test.gguf").without_gpu();
    assert!(config.no_gpu);
}

#[test]
fn test_inference_config_with_verbose() {
    let config = InferenceConfig::new("test.gguf").with_verbose(true);
    assert!(config.verbose);
}

#[test]
fn test_inference_config_with_trace() {
    let config = InferenceConfig::new("test.gguf").with_trace(true);
    assert!(config.trace);
}

#[test]
fn test_inference_config_with_trace_output() {
    let config = InferenceConfig::new("test.gguf").with_trace_output("trace.json");
    assert_eq!(config.trace_output, Some(PathBuf::from("trace.json")));
}

#[test]
fn test_inference_config_builder_chain() {
    let config = InferenceConfig::new("model.gguf")
        .with_prompt("Test prompt")
        .with_max_tokens(64)
        .with_temperature(0.5)
        .with_top_k(20)
        .without_gpu()
        .with_verbose(true)
        .with_trace(true)
        .with_trace_output("output.json");

    assert_eq!(config.model_path, PathBuf::from("model.gguf"));
    assert_eq!(config.prompt, Some("Test prompt".to_string()));
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.5).abs() < 0.001);
    assert_eq!(config.top_k, 20);
    assert!(config.no_gpu);
    assert!(config.verbose);
    assert!(config.trace);
    assert_eq!(config.trace_output, Some(PathBuf::from("output.json")));
}

// ============================================================================
// InferenceResult tests
// ============================================================================

#[test]
fn test_inference_result_fields() {
    let result = InferenceResult {
        text: "Generated output".to_string(),
        tokens: vec![1, 2, 3, 4, 5, 6],
        input_token_count: 3,
        generated_token_count: 3,
        inference_ms: 100.0,
        tok_per_sec: 30.0,
        load_ms: 50.0,
        format: "GGUF".to_string(),
        used_gpu: false,
    };

    assert_eq!(result.text, "Generated output");
    assert_eq!(result.tokens.len(), 6);
    assert_eq!(result.input_token_count, 3);
    assert_eq!(result.generated_token_count, 3);
    assert!((result.inference_ms - 100.0).abs() < 0.1);
    assert!((result.tok_per_sec - 30.0).abs() < 0.1);
    assert!((result.load_ms - 50.0).abs() < 0.1);
    assert_eq!(result.format, "GGUF");
    assert!(!result.used_gpu);
}

#[test]
fn test_inference_result_with_gpu() {
    let result = InferenceResult {
        text: "GPU generated".to_string(),
        tokens: vec![10, 20, 30],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 10.0,
        tok_per_sec: 200.0,
        load_ms: 100.0,
        format: "APR".to_string(),
        used_gpu: true,
    };

    assert!(result.used_gpu);
    assert_eq!(result.format, "APR");
}

#[test]
fn test_inference_result_clone() {
    let result = InferenceResult {
        text: "test".to_string(),
        tokens: vec![1],
        input_token_count: 0,
        generated_token_count: 1,
        inference_ms: 1.0,
        tok_per_sec: 1000.0,
        load_ms: 1.0,
        format: "SafeTensors".to_string(),
        used_gpu: false,
    };

    let cloned = result.clone();
    assert_eq!(cloned.text, result.text);
    assert_eq!(cloned.tokens, result.tokens);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_inference_config_empty_prompt() {
    let config = InferenceConfig::new("test.gguf").with_prompt("");
    assert_eq!(config.prompt, Some(String::new()));
}

#[test]
fn test_inference_config_zero_max_tokens() {
    let config = InferenceConfig::new("test.gguf").with_max_tokens(0);
    assert_eq!(config.max_tokens, 0);
}

#[test]
fn test_inference_config_negative_temperature_equivalent() {
    // Temperature should generally be non-negative, but the API accepts any f32
    let config = InferenceConfig::new("test.gguf").with_temperature(-0.1);
    assert!((config.temperature - (-0.1)).abs() < 0.001);
}

#[test]
fn test_inference_config_path_with_spaces() {
    let config = InferenceConfig::new("/path/with spaces/model.gguf");
    assert_eq!(
        config.model_path,
        PathBuf::from("/path/with spaces/model.gguf")
    );
}

#[test]
fn test_inference_config_unicode_path() {
    let config = InferenceConfig::new("/路径/模型.gguf");
    assert_eq!(config.model_path, PathBuf::from("/路径/模型.gguf"));
}

#[test]
fn test_inference_result_empty_tokens() {
    let result = InferenceResult {
        text: String::new(),
        tokens: vec![],
        input_token_count: 0,
        generated_token_count: 0,
        inference_ms: 0.0,
        tok_per_sec: 0.0,
        load_ms: 0.0,
        format: "Unknown".to_string(),
        used_gpu: false,
    };

    assert!(result.tokens.is_empty());
    assert_eq!(result.generated_token_count, 0);
}

#[test]
fn test_inference_result_high_throughput() {
    let result = InferenceResult {
        text: "high speed".to_string(),
        tokens: vec![1, 2, 3],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 0.1,
        tok_per_sec: 20000.0, // Very high throughput
        load_ms: 0.1,
        format: "GGUF".to_string(),
        used_gpu: true,
    };

    assert!(result.tok_per_sec > 10000.0);
}
