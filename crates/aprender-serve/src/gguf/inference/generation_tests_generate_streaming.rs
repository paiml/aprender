
#[test]
fn test_generate_streaming_callback_receives_tokens() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(3);
    let prompt = vec![1];

    let mut received_tokens = Vec::new();
    let result = model
        .generate_with_cache_streaming(&prompt, &gen_config, |token| {
            received_tokens.push(token);
            true
        })
        .expect("expected value");

    // Callback should receive each generated token
    if result.len() > prompt.len() {
        assert!(
            !received_tokens.is_empty(),
            "Callback should receive generated tokens"
        );
        // Received tokens should match generated portion
        let generated = &result[prompt.len()..];
        assert_eq!(
            received_tokens.len(),
            generated.len(),
            "Should receive one callback per generated token"
        );
    }
}

#[test]
fn test_generate_streaming_callback_can_stop() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(10);
    let prompt = vec![1];

    let mut count = 0;
    let _result = model
        .generate_with_cache_streaming(&prompt, &gen_config, |_| {
            count += 1;
            count < 2 // Stop after 2 tokens
        })
        .expect("expected value");

    assert!(
        count <= 2,
        "Callback should be able to stop generation, got {} callbacks",
        count
    );
}

// =============================================================================
// Predict Next Tests
// =============================================================================

#[test]
fn test_predict_next_basic() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);

    let result = model.predict_next(&[1, 2, 3]);
    assert!(result.is_ok(), "predict_next should succeed");

    let token = result.expect("token");
    assert!(
        token < cfg.vocab_size as u32,
        "Token should be in vocab range"
    );
}

#[test]
fn test_predict_next_deterministic() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);

    let result1 = model.predict_next(&[1, 2]).expect("result1");
    let result2 = model.predict_next(&[1, 2]).expect("result2");

    assert_eq!(result1, result2, "predict_next should be deterministic");
}

// =============================================================================
// GH-167: Context Limit Exceeded Tests (F-QUAL-037)
// =============================================================================

/// GH-167: Test that prompt exceeding context_length returns clear error
#[test]
fn test_gh167_context_limit_exceeded_returns_clean_error() {
    // Create model with small context_length
    let mut cfg = make_test_config();
    cfg.context_length = 10; // Very small context window
    let model = create_test_model_with_config(&cfg);

    // Create prompt that exceeds context limit
    let oversized_prompt: Vec<u32> = (0..15).collect(); // 15 tokens > 10 context
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.generate(&oversized_prompt, &gen_config);
    assert!(
        result.is_err(),
        "Should error when prompt exceeds context limit"
    );

    let err = result.unwrap_err();
    let err_str = err.to_string();

    // Error message should be user-friendly, not CUDA_ERROR_UNKNOWN
    assert!(
        err_str.contains("Context")
            || err_str.contains("context")
            || err_str.contains("limit")
            || err_str.contains("exceeded"),
        "Error should mention context limit, got: {}",
        err_str
    );
    assert!(
        !err_str.contains("CUDA_ERROR"),
        "Error should NOT be a cryptic CUDA error, got: {}",
        err_str
    );
}

/// GH-167: Test that prompt exactly at context_length is allowed
#[test]
fn test_gh167_context_limit_exact_allowed() {
    let mut cfg = make_test_config();
    cfg.context_length = 10;
    let model = create_test_model_with_config(&cfg);

    // Prompt at exactly context limit (no room for generation, but should not error on limit check)
    let prompt: Vec<u32> = (0..10).collect();
    let gen_config = QuantizedGenerateConfig::deterministic(0); // No new tokens

    // This should succeed (or fail for other reasons, not context limit)
    let result = model.generate(&prompt, &gen_config);
    // Note: May succeed or fail, but should not give cryptic CUDA error
    if let Err(e) = result {
        let err_str = e.to_string();
        assert!(
            !err_str.contains("CUDA_ERROR"),
            "Should not give CUDA error for edge case"
        );
    }
}

/// GH-167: Test generate_with_cache also checks context limit
#[test]
fn test_gh167_generate_with_cache_checks_context_limit() {
    let mut cfg = make_test_config();
    cfg.context_length = 8;
    let model = create_test_model_with_config(&cfg);

    let oversized_prompt: Vec<u32> = (0..12).collect();
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.generate_with_cache(&oversized_prompt, &gen_config);
    assert!(
        result.is_err(),
        "generate_with_cache should check context limit"
    );

    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("Context") || err_str.contains("context") || err_str.contains("exceeded"),
        "Should give clear context error, got: {}",
        err_str
    );
}
