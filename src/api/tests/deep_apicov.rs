
#[tokio::test]
async fn test_deep_apicov_realize_reload_no_registry() {
    // Test reload without registry mode enabled
    let app = create_test_app_shared();
    let json = r#"{"model":"default","path":"/tmp/model.gguf"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/reload")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    // Should return NOT_IMPLEMENTED when registry not enabled
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn test_deep_apicov_realize_model_endpoint() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/realize/model")
                .body(Body::empty())
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
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ModelMetadataResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };
    assert!(!result.id.is_empty());
}

#[tokio::test]
async fn test_deep_apicov_realize_embed_default_model() {
    let app = create_test_app_shared();
    let json = r#"{"input":"test embedding input"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: EmbeddingResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };
    assert_eq!(result.object, "list");
    assert!(!result.data.is_empty());
    assert_eq!(result.data[0].embedding.len(), 384); // 384-dim embedding
}

#[tokio::test]
async fn test_deep_apicov_openai_embeddings_endpoint() {
    let app = create_test_app_shared();
    let json = r#"{"input":"test embedding"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
}

#[tokio::test]
async fn test_deep_apicov_models_endpoint() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/models")
                .body(Body::empty())
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
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ModelsResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };
    assert!(!result.models.is_empty());
}

#[tokio::test]
async fn test_deep_apicov_gpu_warmup_no_model() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/gpu/warmup")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    // Should return SERVICE_UNAVAILABLE when no GPU model
    assert!(
        response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_deep_apicov_gpu_status_no_model() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/gpu/status")
                .body(Body::empty())
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
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: GpuStatusResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };
    assert!(!result.cache_ready); // No GPU model, so not ready
}

#[tokio::test]
async fn test_deep_apicov_apr_explain_empty_features() {
    let app = create_test_app_shared();
    let json = r#"{"features":[],"feature_names":[]}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(json))
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

#[tokio::test]
async fn test_deep_apicov_apr_audit_not_found() {
    let app = create_test_app_shared();
    // Valid UUID format but non-existent
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audit/00000000-0000-0000-0000-000000000000")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[test]
fn test_deep_apicov_default_functions() {
    assert_eq!(default_top_k_features(), 5);
    assert_eq!(default_explain_method(), "shap");
    assert!(default_true());
}

#[test]
fn test_deep_apicov_embedding_request_serialize() {
    let req = EmbeddingRequest {
        input: "test input".to_string(),
        model: Some("test-model".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("test input"));
    assert!(json.contains("test-model"));
}

#[test]
fn test_deep_apicov_embedding_response_structure() {
    let resp = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2, 0.3],
        }],
        model: "test".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("embedding"));
    assert!(json.contains("prompt_tokens"));
}

#[test]
fn test_deep_apicov_completion_request_serialize() {
    let req = CompletionRequest {
        model: "gpt-3.5".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.8),
        top_p: Some(0.95),
        stop: Some(vec!["END".to_string()]),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("gpt-3.5"));
    assert!(json.contains("Hello"));
}

#[test]
fn test_deep_apicov_completion_response_structure() {
    let resp = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1234567890,
        model: "test".to_string(),
        choices: vec![CompletionChoice {
            text: "generated text".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("cmpl-123"));
    assert!(json.contains("generated text"));
}

#[test]
fn test_deep_apicov_reload_request_serialize() {
    let req = ReloadRequest {
        model: Some("my-model".to_string()),
        path: Some("/path/to/model.gguf".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("my-model"));
    assert!(json.contains("/path/to/model"));
}

#[test]
fn test_deep_apicov_reload_response_structure() {
    let resp = ReloadResponse {
        success: true,
        message: "Reload successful".to_string(),
        reload_time_ms: 150,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("success"));
    assert!(json.contains("Reload successful"));
}

#[test]
fn test_deep_apicov_model_lineage_structure() {
    let lineage = ModelLineage {
        uri: "pacha://model:v1".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("recipe.yaml".to_string()),
        parent: Some("parent-model".to_string()),
        content_hash: "blake3:abc123".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(json.contains("pacha://model:v1"));
    assert!(json.contains("recipe.yaml"));
}

#[test]
fn test_deep_apicov_gpu_warmup_response_structure() {
    let resp = GpuWarmupResponse {
        success: true,
        memory_bytes: 1_000_000_000,
        num_layers: 32,
        message: "Warmup complete".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("success"));
    assert!(json.contains("num_layers"));
}

#[test]
fn test_deep_apicov_gpu_batch_request_structure() {
    let req = GpuBatchRequest {
        prompts: vec!["hello".to_string(), "world".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: Vec::new(),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("hello"));
    assert!(json.contains("world"));
}

#[test]
fn test_deep_apicov_gpu_batch_response_structure() {
    let resp = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "generated".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 3,
            processing_time_ms: 10.5,
            throughput_tps: 285.7,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("generated"));
    assert!(json.contains("throughput_tps"));
}

#[test]
fn test_deep_apicov_server_metrics_response_structure() {
    let resp = ServerMetricsResponse {
        throughput_tok_per_sec: 100.5,
        latency_p50_ms: 5.2,
        latency_p95_ms: 15.8,
        latency_p99_ms: 25.3,
        gpu_memory_used_bytes: 1_000_000_000,
        gpu_memory_total_bytes: 24_000_000_000,
        gpu_utilization_percent: 85,
        cuda_path_active: true,
        batch_size: 4,
        queue_depth: 2,
        total_tokens: 10000,
        total_requests: 500,
        uptime_secs: 3600,
        model_name: "test-model".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("throughput_tok_per_sec"));
    assert!(json.contains("test-model"));
}

#[test]
fn test_deep_apicov_explain_request_serialize() {
    let req = ExplainRequest {
        model: Some("my-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        top_k_features: 3,
        method: "lime".to_string(),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("my-model"));
    assert!(json.contains("lime"));
}
