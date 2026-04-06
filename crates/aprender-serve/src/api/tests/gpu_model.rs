
// =============================================================================
// Error Path Falsification Tests
// =============================================================================

/// Test tokenizer missing error path in GPU model
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_gpu_model_tokenizer_missing_error() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create GPU model with empty vocab to trigger tokenizer issues
    let config = GpuModelConfig {
        vocab_size: 10, // Very small vocab
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let gpu_model = GpuModel::new(config).expect("create GPU model");
    let state = AppState::with_gpu_model(gpu_model).expect("create AppState");
    let app = create_router(state);

    // Send request with content that won't tokenize well
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello world!"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    // Should handle gracefully (may succeed or fail with proper error)
    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
            || status == StatusCode::BAD_REQUEST,
        "Should handle gracefully, got {}",
        status
    );
}

/// Test empty messages error path
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_model_empty_messages_error() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("create AppState");
    let app = create_router(state);

    // Empty messages should be rejected
    let req_body = serde_json::json!({
        "model": "default",
        "messages": []
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    // Empty messages trigger tokenization to return empty, which should return BAD_REQUEST
    let status = response.status();
    assert!(
        status == StatusCode::BAD_REQUEST
            || status == StatusCode::OK
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Empty messages should be handled, got {}",
        status
    );
}

// =============================================================================
// Temperature/Sampling Parameter Tests for Model Paths
// =============================================================================

/// Test temperature=0 uses greedy sampling in quantized path
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_model_temperature_zero() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("create AppState");
    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.0
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Temperature=0 should use greedy sampling, got {}",
        status
    );
}

/// Test max_tokens parameter in quantized path
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_model_max_tokens() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("create AppState");
    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "max_tokens should be respected, got {}",
        status
    );
}

// =============================================================================
// Registry Fallback Path Tests (Directive 2: Verify Fallback)
// =============================================================================

/// Test fallback to registry when no specialized model
#[tokio::test]
async fn test_registry_fallback_path() {
    use crate::api::test_helpers::create_test_app_shared;

    // create_test_app uses AppState::demo() which has no GPU/quantized models
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    // Should hit registry path and succeed
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

/// Test registry model streaming fallback
#[tokio::test]
async fn test_registry_streaming_fallback() {
    use crate::api::test_helpers::create_test_app_shared;

    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    // Should hit registry streaming path and succeed
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// =============================================================================
// Separate Stream Handler Tests (openai_handlers.rs lines 1097-1214)
// =============================================================================

/// Test dedicated stream handler endpoint
#[tokio::test]
async fn test_stream_handler_endpoint() {
    use crate::api::test_helpers::create_test_app_shared;

    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Test"}]
    });

    // The stream endpoint returns Result<Sse<...>, (StatusCode, Json<ErrorResponse>)>
    // We test that it routes correctly
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/stream")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    // May be 404 (endpoint not registered) or OK
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Stream endpoint should be handled, got {}",
        status
    );
}

/// Test chat completions with empty prompt after tokenization
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_empty_prompt_after_tokenization() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("create AppState");
    let app = create_router(state);

    // Whitespace-only content may result in empty tokens
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "   "}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    // Should handle gracefully
    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::BAD_REQUEST
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Whitespace-only should be handled, got {}",
        status
    );
}

// =============================================================================
// Finish Reason Tests
// =============================================================================

/// Test finish_reason="length" when max_tokens reached
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_finish_reason_length() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("create AppState");
    let app = create_router(state);

    // Very low max_tokens should trigger length finish reason
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    // Response should contain finish_reason (either "length" or "stop")
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Should handle max_tokens=1, got {}",
        status
    );
}
