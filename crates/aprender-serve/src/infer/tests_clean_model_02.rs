
// ============================================================================
// clean_model_output: partial markers and special strings
// ============================================================================

#[test]
fn test_clean_model_output_partial_im_start() {
    // Partial marker should NOT be stripped
    let cleaned = clean_model_output("<|im_sta");
    assert_eq!(cleaned, "<|im_sta");
}

#[test]
fn test_clean_model_output_only_whitespace() {
    let cleaned = clean_model_output("   \n\n\t  ");
    assert_eq!(cleaned, "");
}

#[test]
fn test_clean_model_output_unicode_content() {
    let cleaned =
        clean_model_output("<|im_start|>assistant\n\u{4f60}\u{597d}\u{4e16}\u{754c}<|im_end|>");
    assert_eq!(cleaned, "\u{4f60}\u{597d}\u{4e16}\u{754c}");
}

#[test]
fn test_clean_model_output_repeated_endoftext() {
    let cleaned = clean_model_output("Hello<|endoftext|><|endoftext|><|endoftext|>");
    assert_eq!(cleaned, "Hello");
}

// ============================================================================
// mock_config: edge cases
// ============================================================================

#[test]
fn test_mock_config_unicode_prompt() {
    let config = mock_config("\u{1f600} emoji prompt");
    assert_eq!(config.prompt, Some("\u{1f600} emoji prompt".to_string()));
    assert!(config.use_mock_backend);
}

#[test]
fn test_mock_config_very_long_prompt() {
    let long = "x".repeat(10_000);
    let config = mock_config(&long);
    assert_eq!(config.prompt.as_deref().map(str::len), Some(10_000));
}

// ============================================================================
// run_mock_inference: edge cases for token counting
// ============================================================================

#[test]
fn test_mock_inference_single_word_prompt() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("Hello")
        .with_max_tokens(3)
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("should succeed");
    assert_eq!(result.input_token_count, 1, "single word = 1 token");
    assert_eq!(result.generated_token_count, 3);
    assert_eq!(result.tokens.len(), 4); // 1 input + 3 generated
}

#[test]
fn test_mock_inference_whitespace_only_prompt() {
    let config = InferenceConfig::new("/dev/null")
        .with_prompt("   \t  \n  ")
        .with_max_tokens(2)
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("should succeed");
    // split_whitespace on whitespace-only yields 0 tokens
    assert_eq!(result.input_token_count, 0);
    assert_eq!(result.generated_token_count, 2);
}

#[test]
fn test_mock_inference_generated_token_ids_are_sequential() {
    let config = InferenceConfig::new("/dev/null")
        .with_input_tokens(vec![1])
        .with_max_tokens(5)
        .with_mock_backend();

    let result = run_mock_inference(&config).expect("should succeed");
    // Generated tokens should be [100, 101, 102, 103, 104]
    let generated = &result.tokens[1..];
    for (i, &tok) in generated.iter().enumerate() {
        assert_eq!(
            tok,
            100 + i as u32,
            "generated token {} should be {}",
            i,
            100 + i
        );
    }
}

// ============================================================================
// VALID_MODEL_EXTENSIONS: comprehensive check
// ============================================================================

#[test]
fn test_valid_model_extensions_count() {
    assert_eq!(
        VALID_MODEL_EXTENSIONS.len(),
        5,
        "should have exactly 5 valid extensions: gguf, safetensors, apr, bin, json"
    );
}

#[test]
fn test_valid_model_extensions_does_not_include_dangerous() {
    for ext in &["sh", "py", "exe", "bat", "cmd", "ps1", "rb", "pl"] {
        assert!(
            !VALID_MODEL_EXTENSIONS.contains(ext),
            "dangerous extension {} should not be in valid list",
            ext
        );
    }
}
