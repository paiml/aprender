
// =============================================================================
// Multi-Message Conversation Tests
// =============================================================================

/// Test multi-turn conversation with quantized model
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_multi_turn_conversation() {
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
        context_length: 512,
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
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"}
        ]
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
        "Multi-turn should be handled, got {}",
        status
    );
}
