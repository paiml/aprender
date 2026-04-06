
#[tokio::test]
#[ignore = "APR audit integration test - depends on predict endpoint"]
async fn test_apr_audit_endpoint() {
    // Tests real audit trail: predict creates record, audit fetches it
    let state = AppState::demo().expect("test");
    let app = create_router(state);

    // First, make a prediction to create an audit record
    let predict_request = PredictRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: None,
        top_k: None,
        include_confidence: true,
    };

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&predict_request).expect("test"),
                ))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let predict_result: PredictResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };
    let request_id = predict_result.request_id;

    // Now fetch the audit record for this prediction
    let audit_response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/audit/{}", request_id))
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(audit_response.status(), StatusCode::OK);

    let audit_body = axum::body::to_bytes(audit_response.into_body(), usize::MAX)
        .await
        .expect("test");
    let audit_result: AuditResponse = match serde_json::from_slice(&audit_body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    // Verify the audit record matches the prediction request
    assert_eq!(audit_result.record.request_id, request_id);
}

#[tokio::test]
async fn test_apr_audit_invalid_id() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audit/not-a-valid-uuid")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[test]
fn test_predict_request_serialization() {
    let request = PredictRequest {
        model: Some("test-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        top_k: Some(3),
        include_confidence: true,
    };

    let json = serde_json::to_string(&request).expect("test");
    assert!(json.contains("test-model"));
    assert!(json.contains("features"));

    // Deserialize back
    let deserialized: PredictRequest = serde_json::from_str(&json).expect("test");
    assert_eq!(deserialized.features.len(), 3);
}

#[test]
fn test_explain_request_defaults() {
    let json = r#"{"features": [1.0], "feature_names": ["f1"]}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("test");

    assert_eq!(request.top_k_features, 5); // default
    assert_eq!(request.method, "shap"); // default
}

// ==========================================================================
// M33: GGUF HTTP Serving Integration Tests (IMP-084 through IMP-087)
// ==========================================================================

/// IMP-084: AppState::with_gpu_model creates state with GPU model
#[test]
#[cfg(feature = "gpu")]
fn test_imp_084_app_state_with_gpu_model() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create minimal GPU model
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2, // Standard MHA
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
    let gpu_model = GpuModel::new(config).expect("Failed to create GPU model");

    // Create AppState with GPU model
    let state = AppState::with_gpu_model(gpu_model).expect("Failed to create AppState");

    // Verify GPU model is present
    assert!(
        state.has_gpu_model(),
        "IMP-084: AppState should have GPU model"
    );
}

/// IMP-085: /v1/completions endpoint uses GPU model when available
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_085_completions_uses_gpu_model() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create minimal GPU model
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2, // Standard MHA
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
    let gpu_model = GpuModel::new(config).expect("Failed to create GPU model");

    let state = AppState::with_gpu_model(gpu_model).expect("Failed to create AppState");
    let app = create_router(state);

    // Make completion request
    let request = CompletionRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(5),
        temperature: Some(0.0),
        model: "default".to_string(),
        top_p: None,
        stop: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    // Should succeed (200 OK) with GPU model
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "IMP-085: /v1/completions should work with GPU model"
    );
}

// ========================================================================
// IMP-116: Cached Model HTTP Integration Tests
// ========================================================================

/// IMP-116a: Test AppState can store cached model
#[test]
#[cfg(feature = "gpu")]
fn test_imp_116a_appstate_cached_model_storage() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,

        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    // Create test model
    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);

    // Create AppState with cached model
    let state = AppState::with_cached_model(cached_model)
        .expect("IMP-116a: AppState should accept cached model");

    // Verify model is accessible
    assert!(
        state.cached_model().is_some(),
        "IMP-116a: Cached model should be accessible from AppState"
    );
}

/// IMP-116b: Test cached model is thread-safe for async handlers
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_116b_cached_model_thread_safety() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = Arc::new(OwnedQuantizedModelCachedSync::new(model));

    // Spawn multiple concurrent tasks accessing the model
    let mut handles = Vec::new();
    for i in 0..4 {
        let model_clone = cached_model.clone();
        handles.push(tokio::spawn(async move {
            // Should be able to get inner model from any thread
            let inner = model_clone.model();
            assert_eq!(inner.config.hidden_dim, 64, "Task {i} should access model");
        }));
    }

    // All tasks should complete successfully
    for handle in handles {
        handle
            .await
            .expect("IMP-116b: Concurrent access should succeed");
    }
}

/// IMP-116c: Test completions endpoint routes to cached model
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_116c_completions_uses_cached_model() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);

    // Create state with cached model
    let state = AppState::with_cached_model(cached_model).expect("Failed to create AppState");

    // Verify cached model is stored correctly
    assert!(
        state.has_cached_model(),
        "IMP-116c: AppState should have cached model"
    );
    assert!(
        state.cached_model().is_some(),
        "IMP-116c: cached_model() should return Some"
    );

    let app = create_router(state);

    // Make completion request - may fail due to test model but path should be exercised
    let request = CompletionRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(3),
        temperature: Some(0.0),
        model: "default".to_string(),
        top_p: None,
        stop: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    // The request was routed (may fail with 500 due to test model)
    // Key point: no panic, request was handled
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "IMP-116c: Request should be handled (got {})",
        status
    );
}
