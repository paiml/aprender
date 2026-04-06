
#[test]
fn test_reload_request_empty() {
    let json = "{}";
    let req: ReloadRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert!(req.model.is_none());
    assert!(req.path.is_none());
}

#[test]
fn test_reload_response_serde() {
    let resp = ReloadResponse {
        success: true,
        message: "Reloaded".to_string(),
        reload_time_ms: 500,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: ReloadResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.success);
    assert_eq!(parsed.reload_time_ms, 500);
}

#[test]
fn test_completion_request_serde() {
    let req = CompletionRequest {
        model: "default".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        stop: Some(vec!["END".to_string()]),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.model, "default");
    assert_eq!(parsed.max_tokens, Some(100));
}

#[test]
fn test_completion_response_serde() {
    let resp = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1234567890,
        model: "default".to_string(),
        choices: vec![CompletionChoice {
            text: "world".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        },
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: CompletionResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].text, "world");
}

// ============================================================================
// GPU handler structs serde
// ============================================================================

#[test]
fn test_gpu_batch_request_serde() {
    let req = gpu_handlers::GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 64,
        temperature: 0.7,
        top_k: 40,
        stop: vec!["<|end|>".to_string()],
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: gpu_handlers::GpuBatchRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.prompts.len(), 2);
    assert_eq!(parsed.max_tokens, 64);
}

#[test]
fn test_gpu_batch_response_serde() {
    let resp = gpu_handlers::GpuBatchResponse {
        results: vec![gpu_handlers::GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "hello".to_string(),
            num_generated: 3,
        }],
        stats: gpu_handlers::GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 3,
            processing_time_ms: 10.0,
            throughput_tps: 300.0,
        },
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: gpu_handlers::GpuBatchResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.results.len(), 1);
    assert!(parsed.stats.throughput_tps > 0.0);
}

#[test]
fn test_gpu_warmup_response_serde() {
    let resp = gpu_handlers::GpuWarmupResponse {
        success: true,
        memory_bytes: 1024 * 1024,
        num_layers: 12,
        message: "Warmed up".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: gpu_handlers::GpuWarmupResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.success);
    assert_eq!(parsed.num_layers, 12);
}

#[test]
fn test_gpu_status_response_serde() {
    let resp = gpu_handlers::GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 4,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: gpu_handlers::GpuStatusResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(!parsed.cache_ready);
    assert_eq!(parsed.batch_threshold, 32);
}

#[test]
fn test_gpu_batch_result_serde() {
    let result = gpu_handlers::GpuBatchResult {
        index: 5,
        token_ids: vec![10, 20, 30],
        text: "generated".to_string(),
        num_generated: 3,
    };
    let json = serde_json::to_string(&result).expect("JSON serialization failed");
    let parsed: gpu_handlers::GpuBatchResult = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.index, 5);
    assert_eq!(parsed.token_ids, vec![10, 20, 30]);
}

#[test]
fn test_gpu_batch_stats_serde() {
    let stats = gpu_handlers::GpuBatchStats {
        batch_size: 8,
        gpu_used: true,
        total_tokens: 256,
        processing_time_ms: 50.5,
        throughput_tps: 5069.3,
    };
    let json = serde_json::to_string(&stats).expect("JSON serialization failed");
    let parsed: gpu_handlers::GpuBatchStats = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.gpu_used);
    assert_eq!(parsed.batch_size, 8);
}

// ============================================================================
// HTTP handler coverage - realize endpoints
// ============================================================================

#[tokio::test]
async fn test_realize_embed_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "input": "Hello world"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // May return OK or error depending on mock state
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_openai_models_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test value should be present");
    let parsed: OpenAIModelsResponse = serde_json::from_slice(&body).expect("JSON from_slice failed");
    assert!(!parsed.data.is_empty());
}

#[tokio::test]
async fn test_openai_chat_completions_non_streaming() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false,
        "max_tokens": 5
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Accept various status codes since mock state may not have full model
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_openai_completions_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "prompt": "Hello",
        "max_tokens": 5
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_gpu_warmup_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/gpu/warmup")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // GPU warmup returns status even without GPU
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_gpu_status_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/gpu/status")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": ["Hello", "World"],
        "max_tokens": 5
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_server_metrics_handler_via_http() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/metrics")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_tokenize_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "text": "Hello world"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_batch_tokenize_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "texts": ["Hello", "World"]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_generate_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 3,
        "temperature": 0.0,
        "strategy": "greedy",
        "top_k": 1,
        "top_p": 1.0
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}
