
// ============================================================================
// InferenceResult construction and accessors
// ============================================================================

#[test]
fn test_inference_result_debug() {
    let result = InferenceResult {
        text: "hello".to_string(),
        tokens: vec![1, 100],
        input_token_count: 1,
        generated_token_count: 1,
        inference_ms: 50.0,
        tok_per_sec: 20.0,
        load_ms: 10.0,
        format: "Test".to_string(),
        used_gpu: false,
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("hello"));
    assert!(debug.contains("Test"));
}

#[test]
fn test_inference_result_clone() {
    let result = InferenceResult {
        text: "world".to_string(),
        tokens: vec![1, 2, 3],
        input_token_count: 1,
        generated_token_count: 2,
        inference_ms: 100.0,
        tok_per_sec: 20.0,
        load_ms: 5.0,
        format: "GGUF".to_string(),
        used_gpu: true,
    };
    let cloned = result.clone();
    assert_eq!(cloned.text, "world");
    assert_eq!(cloned.tokens.len(), 3);
    assert!(cloned.used_gpu);
}

// ============================================================================
// mock_config helper coverage
// ============================================================================

#[test]
fn test_mock_config_helper() {
    let config = mock_config("hello");
    assert_eq!(config.prompt.as_deref(), Some("hello"));
    assert_eq!(config.max_tokens, 16);
    assert!(config.use_mock_backend);
}

#[test]
fn test_mock_config_empty_prompt() {
    let config = mock_config("");
    let result = run_mock_inference(&config).unwrap();
    assert!(result.text.contains("mock response for:"));
}
