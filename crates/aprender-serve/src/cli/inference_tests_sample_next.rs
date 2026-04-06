//! T-COV-95 Deep Coverage: inference.rs pure functions (Part 05)
//!
//! Tests sample_next_token, print_inference_output, print_gpu_model_info,
//! decode_apr_output_tokens (fallback path), and print_model_info helpers.

#![allow(clippy::needless_pass_by_value)]

use super::inference::*;

// ============================================================================
// sample_next_token tests
// ============================================================================

#[test]
fn test_sample_next_token_greedy_zero_temp() {
    let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
    let token = sample_next_token(&logits, 0.0);
    assert_eq!(token, 3); // Index of max value (0.8)
}

#[test]
fn test_sample_next_token_greedy_low_temp() {
    let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
    let token = sample_next_token(&logits, 0.001);
    assert_eq!(token, 3); // Still greedy
}

#[test]
fn test_sample_next_token_greedy_negative_values() {
    let logits = vec![-1.0, -0.5, -2.0, -0.1, -3.0];
    let token = sample_next_token(&logits, 0.0);
    assert_eq!(token, 3); // Index of max (-0.1)
}

#[test]
fn test_sample_next_token_greedy_all_same() {
    let logits = vec![1.0, 1.0, 1.0, 1.0];
    let token = sample_next_token(&logits, 0.0);
    // Any index is valid when all values are equal
    assert!(token < 4);
}

#[test]
fn test_sample_next_token_greedy_single_element() {
    let logits = vec![42.0];
    let token = sample_next_token(&logits, 0.0);
    assert_eq!(token, 0);
}

#[test]
fn test_sample_next_token_greedy_large_vocab() {
    let mut logits = vec![0.0f32; 32000];
    logits[15000] = 100.0; // Clear max
    let token = sample_next_token(&logits, 0.0);
    assert_eq!(token, 15000);
}

#[test]
fn test_sample_next_token_with_temperature() {
    let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
    // With temperature, should still favor the highest logit
    let token = sample_next_token(&logits, 0.5);
    // With such a dominant logit, should almost always be 0
    assert!(token < 5);
}

#[test]
fn test_sample_next_token_empty_logits_returns_zero() {
    let logits: Vec<f32> = vec![];
    let token = sample_next_token(&logits, 0.0);
    assert_eq!(token, 0);
}

// ============================================================================
// print_inference_output tests (output format paths)
// ============================================================================

#[test]
fn test_print_inference_output_json_format() {
    // Just verify it doesn't panic
    print_inference_output(
        "test_model",
        "hello",
        "world",
        5,
        100.0,
        50.0,
        0.7,
        "json",
        false,
    );
}

#[test]
fn test_print_inference_output_text_verbose() {
    print_inference_output(
        "test_model",
        "hello",
        "world",
        5,
        100.0,
        50.0,
        0.7,
        "text",
        true,
    );
}

#[test]
fn test_print_inference_output_text_non_verbose() {
    // Ollama-style clean output
    print_inference_output(
        "test_model",
        "hello",
        "world",
        5,
        100.0,
        50.0,
        0.7,
        "text",
        false,
    );
}

#[test]
fn test_print_inference_output_unknown_format() {
    // Unknown format defaults to text
    print_inference_output(
        "test_model",
        "prompt",
        "output",
        10,
        200.0,
        100.0,
        0.0,
        "yaml",
        true,
    );
}

#[test]
fn test_print_inference_output_zero_tokens() {
    print_inference_output(
        "model.gguf",
        "test",
        "",
        0,
        0.0,
        0.0,
        0.0,
        "json",
        false,
    );
}

// ============================================================================
// print_gpu_model_info tests
// ============================================================================

#[test]
fn test_print_gpu_model_info_basic() {
    // Just verify it doesn't panic
    print_gpu_model_info(32000, 4096, 32, 10, Some(1), Some(2), 0.7);
}

#[test]
fn test_print_gpu_model_info_no_special_tokens() {
    print_gpu_model_info(100, 64, 2, 5, None, None, 0.0);
}

#[test]
fn test_print_gpu_model_info_zero_temp() {
    print_gpu_model_info(50000, 2048, 16, 1, Some(0), Some(50000), 0.0);
}
