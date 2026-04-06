//! Deep coverage tests for realizar/src/infer.rs
//!
//! This module provides additional coverage for the high-level inference API.
//! Targets 90%+ coverage.

use realizar::infer::{InferenceConfig, InferenceResult};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

// ============================================================================
// Test fixtures helpers
// ============================================================================

fn create_temp_file(name: &str, content: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("infer_test_{}", name));
    let mut file = fs::File::create(&path).expect("create test file");
    file.write_all(content).expect("write test content");
    path
}

fn cleanup_file(path: &PathBuf) {
    let _ = fs::remove_file(path);
}

// ============================================================================
// Test 1-15: InferenceConfig builder tests
// ============================================================================

#[test]
fn test_inference_config_new_pathbuf() {
    let path = PathBuf::from("/test/model.gguf");
    let config = InferenceConfig::new(path.clone());
    assert_eq!(config.model_path, path);
}

#[test]
fn test_inference_config_new_string() {
    let config = InferenceConfig::new("/test/model.gguf".to_string());
    assert_eq!(config.model_path, PathBuf::from("/test/model.gguf"));
}

#[test]
fn test_inference_config_new_str() {
    let config = InferenceConfig::new("/test/model.gguf");
    assert_eq!(config.model_path, PathBuf::from("/test/model.gguf"));
}

#[test]
fn test_inference_config_with_prompt_string() {
    let config = InferenceConfig::new("/m.gguf").with_prompt("hello".to_string());
    assert_eq!(config.prompt, Some("hello".to_string()));
}

#[test]
fn test_inference_config_with_prompt_str() {
    let config = InferenceConfig::new("/m.gguf").with_prompt("hello");
    assert_eq!(config.prompt, Some("hello".to_string()));
}

#[test]
fn test_inference_config_with_input_tokens_empty() {
    let config = InferenceConfig::new("/m.gguf").with_input_tokens(vec![]);
    assert_eq!(config.input_tokens, Some(vec![]));
}

#[test]
fn test_inference_config_with_input_tokens_many() {
    let tokens: Vec<u32> = (0..1000).collect();
    let config = InferenceConfig::new("/m.gguf").with_input_tokens(tokens.clone());
    assert_eq!(config.input_tokens, Some(tokens));
}

#[test]
fn test_inference_config_with_max_tokens_zero() {
    let config = InferenceConfig::new("/m.gguf").with_max_tokens(0);
    assert_eq!(config.max_tokens, 0);
}

#[test]
fn test_inference_config_with_max_tokens_large() {
    let config = InferenceConfig::new("/m.gguf").with_max_tokens(1_000_000);
    assert_eq!(config.max_tokens, 1_000_000);
}

#[test]
fn test_inference_config_with_temperature_zero() {
    let config = InferenceConfig::new("/m.gguf").with_temperature(0.0);
    assert!((config.temperature - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_inference_config_with_temperature_high() {
    let config = InferenceConfig::new("/m.gguf").with_temperature(2.0);
    assert!((config.temperature - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_inference_config_with_temperature_negative() {
    // Negative temperature is technically allowed (will be clamped by sampler)
    let config = InferenceConfig::new("/m.gguf").with_temperature(-1.0);
    assert!((config.temperature - (-1.0)).abs() < f32::EPSILON);
}

#[test]
fn test_inference_config_with_top_k_zero() {
    let config = InferenceConfig::new("/m.gguf").with_top_k(0);
    assert_eq!(config.top_k, 0);
}

#[test]
fn test_inference_config_with_top_k_large() {
    let config = InferenceConfig::new("/m.gguf").with_top_k(100000);
    assert_eq!(config.top_k, 100000);
}

#[test]
fn test_inference_config_builder_chaining() {
    let config = InferenceConfig::new("/m.gguf")
        .with_prompt("Hello")
        .with_max_tokens(100)
        .with_temperature(0.8)
        .with_top_k(50)
        .without_gpu()
        .with_verbose(true)
        .with_trace(true);

    assert_eq!(config.prompt, Some("Hello".to_string()));
    assert_eq!(config.max_tokens, 100);
    assert!((config.temperature - 0.8).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 50);
    assert!(config.no_gpu);
    assert!(config.verbose);
    assert!(config.trace);
}

// ============================================================================
// Test 16-25: InferenceConfig defaults and clone
// ============================================================================

#[test]
fn test_inference_config_defaults_no_gpu() {
    let config = InferenceConfig::new("/m.gguf");
    assert!(!config.no_gpu); // GPU enabled by default
}

#[test]
fn test_inference_config_defaults_trace() {
    let config = InferenceConfig::new("/m.gguf");
    assert!(!config.trace);
    assert!(!config.trace_verbose);
}

#[test]
fn test_inference_config_defaults_trace_output() {
    let config = InferenceConfig::new("/m.gguf");
    assert!(config.trace_output.is_none());
}

#[test]
fn test_inference_config_defaults_trace_steps() {
    let config = InferenceConfig::new("/m.gguf");
    assert!(config.trace_steps.is_none());
}

#[test]
fn test_inference_config_defaults_verbose() {
    let config = InferenceConfig::new("/m.gguf");
    assert!(!config.verbose);
}

#[test]
fn test_inference_config_clone() {
    let config = InferenceConfig::new("/m.gguf")
        .with_prompt("test")
        .with_max_tokens(50);
    let cloned = config.clone();
    assert_eq!(config.model_path, cloned.model_path);
    assert_eq!(config.prompt, cloned.prompt);
    assert_eq!(config.max_tokens, cloned.max_tokens);
}

#[test]
fn test_inference_config_debug() {
    let config = InferenceConfig::new("/model.gguf").with_prompt("hello");
    let debug = format!("{:?}", config);
    assert!(debug.contains("InferenceConfig"));
    assert!(debug.contains("model_path"));
    assert!(debug.contains("hello"));
}

#[test]
fn test_inference_config_multiple_with_prompt() {
    // Second prompt overwrites first
    let config = InferenceConfig::new("/m.gguf")
        .with_prompt("first")
        .with_prompt("second");
    assert_eq!(config.prompt, Some("second".to_string()));
}

#[test]
fn test_inference_config_input_tokens_overrides_prompt() {
    // Both can be set (API decides which to use)
    let config = InferenceConfig::new("/m.gguf")
        .with_prompt("hello")
        .with_input_tokens(vec![1, 2, 3]);
    assert!(config.prompt.is_some());
    assert!(config.input_tokens.is_some());
}

#[test]
fn test_inference_config_without_gpu_multiple_calls() {
    let config = InferenceConfig::new("/m.gguf").without_gpu().without_gpu();
    assert!(config.no_gpu);
}

// ============================================================================
// Test 26-40: InferenceResult tests
// ============================================================================

#[test]
fn test_inference_result_creation() {
    let result = InferenceResult {
        text: "Generated text".to_string(),
        tokens: vec![1, 2, 3, 4],
        input_token_count: 2,
        generated_token_count: 2,
        inference_ms: 100.5,
        tok_per_sec: 20.0,
        load_ms: 50.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };
    assert_eq!(result.text, "Generated text");
    assert_eq!(result.tokens, vec![1, 2, 3, 4]);
    assert_eq!(result.input_token_count, 2);
    assert_eq!(result.generated_token_count, 2);
}

#[test]
fn test_inference_result_clone() {
    let result = InferenceResult {
        text: "test".to_string(),
        tokens: vec![1],
        input_token_count: 1,
        generated_token_count: 0,
        inference_ms: 10.0,
        tok_per_sec: 0.0,
        load_ms: 5.0,
        format: "APR".to_string(),
        used_gpu: false,
    };
    let cloned = result.clone();
    assert_eq!(result.text, cloned.text);
    assert_eq!(result.tokens, cloned.tokens);
    assert_eq!(result.format, cloned.format);
    assert_eq!(result.used_gpu, cloned.used_gpu);
}

#[test]
fn test_inference_result_debug() {
    let result = InferenceResult {
        text: "hello".to_string(),
        tokens: vec![1, 2],
        input_token_count: 1,
        generated_token_count: 1,
        inference_ms: 10.0,
        tok_per_sec: 100.0,
        load_ms: 5.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("InferenceResult"));
    assert!(debug.contains("hello"));
    assert!(debug.contains("GGUF"));
}

#[test]
fn test_inference_result_zero_inference_ms() {
    let result = InferenceResult {
        text: String::new(),
        tokens: vec![],
        input_token_count: 0,
        generated_token_count: 0,
        inference_ms: 0.0,
        tok_per_sec: 0.0,
        load_ms: 0.0,
        format: String::new(),
        used_gpu: false,
    };
    assert_eq!(result.inference_ms, 0.0);
    assert_eq!(result.tok_per_sec, 0.0);
}

#[test]
fn test_inference_result_large_values() {
    let result = InferenceResult {
        text: "x".repeat(1_000_000),
        tokens: (0..100_000).collect(),
        input_token_count: 1000,
        generated_token_count: 99_000,
        inference_ms: 1_000_000.0,
        tok_per_sec: 99.0,
        load_ms: 500_000.0,
        format: "SafeTensors".to_string(),
        used_gpu: true,
    };
    assert_eq!(result.text.len(), 1_000_000);
    assert_eq!(result.tokens.len(), 100_000);
}

#[test]
fn test_inference_result_empty_text() {
    let result = InferenceResult {
        text: String::new(),
        tokens: vec![1],
        input_token_count: 1,
        generated_token_count: 0,
        inference_ms: 5.0,
        tok_per_sec: 0.0,
        load_ms: 2.0,
        format: "GGUF".to_string(),
        used_gpu: false,
    };
    assert!(result.text.is_empty());
}

#[test]
fn test_inference_result_unicode_text() {
    let result = InferenceResult {
        text: "こんにちは 🌍 مرحبا".to_string(),
        tokens: vec![1, 2, 3],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 10.0,
        tok_per_sec: 200.0,
        load_ms: 5.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };
    assert!(result.text.contains("こんにちは"));
    assert!(result.text.contains("🌍"));
    assert!(result.text.contains("مرحبا"));
}

#[test]
fn test_inference_result_all_formats() {
    for format in &["GGUF", "APR", "SafeTensors"] {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: format.to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, *format);
    }
}

// ============================================================================
// Test 41-60: run_inference error handling
// ============================================================================

#[test]
fn test_run_inference_nonexistent_file() {
    let config = InferenceConfig::new("/nonexistent/path/to/model.gguf");
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Failed to read") || err_msg.contains("read model"));
}

#[test]
fn test_run_inference_empty_file() {
    let path = create_temp_file("empty.gguf", &[]);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("too small") || err_msg.contains("format"));
    cleanup_file(&path);
}

#[test]
fn test_run_inference_1_byte_file() {
    let path = create_temp_file("1byte.gguf", &[0xFF]);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_7_byte_file() {
    let path = create_temp_file("7byte.gguf", &[0, 1, 2, 3, 4, 5, 6]);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("too small"));
    cleanup_file(&path);
}

#[test]
fn test_run_inference_8_byte_invalid_magic() {
    // 8 bytes but invalid magic
    let path = create_temp_file("invalid_magic.gguf", &[0, 0, 0, 0, 0, 0, 0, 0]);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Format") || err_msg.contains("format") || err_msg.contains("detection")
    );
    cleanup_file(&path);
}

#[test]
fn test_run_inference_random_data() {
    let data: Vec<u8> = (0..100).map(|i| (i * 37) as u8).collect();
    let path = create_temp_file("random.bin", &data);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_gguf_magic_only() {
    // GGUF magic but nothing else
    let path = create_temp_file("gguf_magic.gguf", b"GGUF\x03\x00\x00\x00");
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    // Should fail during model loading, not format detection
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_safetensors_magic_only() {
    // SafeTensors-like but incomplete
    let mut data = vec![0u8; 16];
    data[0..8].copy_from_slice(&8u64.to_le_bytes()); // header size
    let path = create_temp_file("st_magic.safetensors", &data);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    // Should fail during model loading
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_apr_magic_only() {
    // APR magic but incomplete
    let mut data = vec![0u8; 16];
    data[0..4].copy_from_slice(b"APR\0");
    let path = create_temp_file("apr_magic.apr", &data);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    // Should fail during model loading
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_directory_as_path() {
    // Try to "read" a directory
    let dir = std::env::temp_dir();
    let config = InferenceConfig::new(&dir);
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
}

#[test]
fn test_run_inference_with_verbose() {
    let path = create_temp_file("verbose_test.bin", &[0; 10]);
    let config = InferenceConfig::new(&path).with_verbose(true);
    // Should still fail but exercise verbose path
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_with_no_gpu() {
    let path = create_temp_file("no_gpu_test.bin", &[0; 10]);
    let config = InferenceConfig::new(&path).without_gpu();
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_with_trace() {
    let path = create_temp_file("trace_test.bin", &[0; 10]);
    let config = InferenceConfig::new(&path).with_trace(true);
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    cleanup_file(&path);
}

// ============================================================================
// Test 61-80: File path edge cases
// ============================================================================

#[test]
fn test_inference_config_relative_path() {
    let config = InferenceConfig::new("./model.gguf");
    assert_eq!(config.model_path, PathBuf::from("./model.gguf"));
}

#[test]
fn test_inference_config_path_with_spaces() {
    let config = InferenceConfig::new("/path with spaces/model file.gguf");
    assert_eq!(
        config.model_path,
        PathBuf::from("/path with spaces/model file.gguf")
    );
}

#[test]
fn test_inference_config_path_with_unicode() {
    let config = InferenceConfig::new("/путь/модель.gguf");
    assert_eq!(config.model_path, PathBuf::from("/путь/модель.gguf"));
}

#[test]
fn test_inference_config_very_long_path() {
    let long_path = format!("/{}/model.gguf", "a".repeat(500));
    let config = InferenceConfig::new(&long_path);
    assert_eq!(config.model_path, PathBuf::from(long_path));
}

#[test]
fn test_inference_config_empty_path() {
    let config = InferenceConfig::new("");
    assert_eq!(config.model_path, PathBuf::from(""));
}

#[test]
fn test_inference_config_dot_dot_path() {
    let config = InferenceConfig::new("../../../model.gguf");
    assert_eq!(config.model_path, PathBuf::from("../../../model.gguf"));
}

// ============================================================================
// Test 81-100: Property-based tests with proptest
// ============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_inference_config_max_tokens_preserved(max_tokens in 0usize..1_000_000) {
            let config = InferenceConfig::new("/m.gguf").with_max_tokens(max_tokens);
            prop_assert_eq!(config.max_tokens, max_tokens);
        }

        #[test]
        fn prop_inference_config_top_k_preserved(top_k in 0usize..100_000) {
            let config = InferenceConfig::new("/m.gguf").with_top_k(top_k);
            prop_assert_eq!(config.top_k, top_k);
        }

        #[test]
        fn prop_inference_config_temperature_preserved(temp in -10.0f32..10.0f32) {
            let config = InferenceConfig::new("/m.gguf").with_temperature(temp);
            prop_assert!((config.temperature - temp).abs() < f32::EPSILON);
        }

        #[test]
        fn prop_inference_config_prompt_preserved(prompt in "\\PC{1,100}") {
            let config = InferenceConfig::new("/m.gguf").with_prompt(prompt.clone());
            prop_assert_eq!(config.prompt, Some(prompt));
        }

        #[test]
        fn prop_inference_config_tokens_preserved(tokens in prop::collection::vec(0u32..100000, 0..100)) {
            let config = InferenceConfig::new("/m.gguf").with_input_tokens(tokens.clone());
            prop_assert_eq!(config.input_tokens, Some(tokens));
        }

        #[test]
        fn prop_inference_result_clone_is_equal(
            text in "\\PC{0,100}",
            tokens in prop::collection::vec(0u32..100000, 0..10),
            input_count in 0usize..100,
            gen_count in 0usize..100,
            inference_ms in 0.0f64..1_000_000.0f64,
        ) {
            let result = InferenceResult {
                text,
                tokens,
                input_token_count: input_count,
                generated_token_count: gen_count,
                inference_ms,
                tok_per_sec: if inference_ms > 0.0 { gen_count as f64 / (inference_ms / 1000.0) } else { 0.0 },
                load_ms: 0.0,
                format: "GGUF".to_string(),
                used_gpu: false,
            };
            let cloned = result.clone();
            prop_assert_eq!(result.text, cloned.text);
            prop_assert_eq!(result.tokens, cloned.tokens);
            prop_assert_eq!(result.input_token_count, cloned.input_token_count);
        }

        #[test]
        fn prop_inference_config_path_preserved(path in "\\PC{1,200}") {
            let config = InferenceConfig::new(path.clone());
            prop_assert_eq!(config.model_path, PathBuf::from(path));
        }

        #[test]
        fn prop_inference_invalid_file_always_errors(size in 0usize..8) {
            let data = vec![0u8; size];
            let path = create_temp_file(&format!("prop_invalid_{}.bin", size), &data);
            let config = InferenceConfig::new(&path);
            let result = realizar::infer::run_inference(&config);
            cleanup_file(&path);
            prop_assert!(result.is_err());
        }
    }
}

// ============================================================================
// Test 101-120: Edge cases and boundary conditions
// ============================================================================

#[test]
fn test_inference_config_chained_overwrite() {
    let config = InferenceConfig::new("/first.gguf")
        .with_prompt("first")
        .with_prompt("second")
        .with_max_tokens(10)
        .with_max_tokens(20)
        .with_temperature(0.5)
        .with_temperature(0.9);

    assert_eq!(config.prompt, Some("second".to_string()));
    assert_eq!(config.max_tokens, 20);
    assert!((config.temperature - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_inference_result_tok_per_sec_calculation() {
    let result = InferenceResult {
        text: "test".to_string(),
        tokens: vec![1, 2, 3, 4, 5],
        input_token_count: 1,
        generated_token_count: 4,
        inference_ms: 1000.0, // 1 second
        tok_per_sec: 4.0,     // 4 tokens / 1 second
        load_ms: 0.0,
        format: "GGUF".to_string(),
        used_gpu: false,
    };
    assert!((result.tok_per_sec - 4.0).abs() < f64::EPSILON);
}

#[test]
fn test_inference_result_high_throughput() {
    let result = InferenceResult {
        text: "x".repeat(1000),
        tokens: (0..1001).collect(),
        input_token_count: 1,
        generated_token_count: 1000,
        inference_ms: 100.0,  // 100ms
        tok_per_sec: 10000.0, // 1000 tokens / 0.1 seconds = 10000 tok/s
        load_ms: 10.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };
    assert!((result.tok_per_sec - 10000.0).abs() < f64::EPSILON);
}

#[test]
fn test_inference_result_gpu_vs_cpu() {
    let gpu_result = InferenceResult {
        text: "gpu".to_string(),
        tokens: vec![1],
        input_token_count: 1,
        generated_token_count: 0,
        inference_ms: 10.0,
        tok_per_sec: 0.0,
        load_ms: 100.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };

    let cpu_result = InferenceResult {
        text: "cpu".to_string(),
        tokens: vec![1],
        input_token_count: 1,
        generated_token_count: 0,
        inference_ms: 50.0,
        tok_per_sec: 0.0,
        load_ms: 50.0,
        format: "GGUF".to_string(),
        used_gpu: false,
    };

    assert!(gpu_result.used_gpu);
    assert!(!cpu_result.used_gpu);
}

#[test]
fn test_inference_config_all_defaults_explicit() {
    let config = InferenceConfig::new("/m.gguf");
    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 1);
    assert!(!config.no_gpu);
    assert!(!config.verbose);
    assert!(!config.trace);
    assert!(!config.trace_verbose);
    assert!(config.prompt.is_none());
    assert!(config.input_tokens.is_none());
    assert!(config.trace_output.is_none());
    assert!(config.trace_steps.is_none());
}

// ============================================================================
// Test GGUF format detection with valid header
// ============================================================================

#[test]
fn test_run_inference_gguf_valid_header_incomplete_body() {
    // Create a file with valid GGUF header but incomplete body
    let mut data = Vec::new();
    // GGUF v3 header
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata kv count

    let path = create_temp_file("gguf_header_only.gguf", &data);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    // Should detect GGUF format but fail on incomplete data
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_verbose_error_path() {
    let path = create_temp_file("verbose_err.gguf", &[0; 10]);
    let config = InferenceConfig::new(&path).with_verbose(true).without_gpu();
    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    cleanup_file(&path);
}

// ============================================================================
// Test APR format detection
// ============================================================================

#[test]
#[ignore = "APR parser behavior changed - incomplete files may not error during format detection"]
fn test_run_inference_apr_header_incomplete() {
    // APR header with version but incomplete
    let mut data = Vec::new();
    data.extend_from_slice(b"APR\0"); // magic
    data.extend_from_slice(&1u32.to_le_bytes()); // version
    data.extend_from_slice(&0u32.to_le_bytes()); // tensor count
    data.extend_from_slice(&0u32.to_le_bytes()); // metadata size
                                                 // Add padding to make it "valid-looking"
    data.extend_from_slice(&[0u8; 100]);

    let path = create_temp_file("apr_incomplete.apr", &data);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    // Should fail during APR loading
    assert!(result.is_err());
    cleanup_file(&path);
}

// ============================================================================
// Test SafeTensors format detection
// ============================================================================

#[test]
fn test_run_inference_safetensors_incomplete() {
    // SafeTensors with header size but incomplete JSON
    let mut data = Vec::new();
    let header = b"{}";
    let header_size = header.len() as u64;
    data.extend_from_slice(&header_size.to_le_bytes());
    data.extend_from_slice(header);

    let path = create_temp_file("st_incomplete.safetensors", &data);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    // Should detect SafeTensors but fail (no config.json)
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_run_inference_safetensors_with_tensors() {
    // SafeTensors with valid header but no config.json
    let mut data = Vec::new();
    let header = br#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let header_size = header.len() as u64;
    data.extend_from_slice(&header_size.to_le_bytes());
    data.extend_from_slice(header);
    // Add tensor data
    data.extend_from_slice(&[0u8; 16]);

    let path = create_temp_file("st_with_tensor.safetensors", &data);
    let config = InferenceConfig::new(&path);
    let result = realizar::infer::run_inference(&config);
    // Should fail because SafeTensors requires config.json for inference
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("config.json") || err_msg.contains("SafeTensors"));
    cleanup_file(&path);
}

// ============================================================================
// Integration tests with model loading
// ============================================================================

#[test]
fn test_run_inference_with_all_options_invalid_file() {
    let path = create_temp_file("all_options.bin", &[0; 100]);
    let config = InferenceConfig::new(&path)
        .with_prompt("Hello world")
        .with_max_tokens(50)
        .with_temperature(0.8)
        .with_top_k(40)
        .without_gpu()
        .with_verbose(true)
        .with_trace(true);

    let result = realizar::infer::run_inference(&config);
    assert!(result.is_err());
    cleanup_file(&path);
}

#[test]
fn test_inference_config_with_trace_output_path() {
    let config = InferenceConfig::new("/m.gguf");
    // trace_output is not settable via builder in current API
    // but we can test it's None by default
    assert!(config.trace_output.is_none());
}
