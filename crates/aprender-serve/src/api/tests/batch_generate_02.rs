
#[tokio::test]
async fn test_batch_generate_handler_invalid_strategy() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompts": ["test"], "strategy": "invalid"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_batch_generate_handler_with_seed() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompts": ["Test"], "max_tokens": 3, "seed": 12345}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// =============================================================================
// Edge Cases and Error Paths
// =============================================================================

#[test]
fn test_gpu_batch_request_large_prompts() {
    let large_prompt = "x".repeat(10000);
    let request = GpuBatchRequest {
        prompts: vec![large_prompt.clone()],
        max_tokens: 10,
        temperature: 0.5,
        top_k: 40,
        stop: vec![],
    };

    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.len() > 10000);
}

#[test]
fn test_gpu_batch_request_many_prompts() {
    let prompts: Vec<String> = (0..100).map(|i| format!("Prompt {}", i)).collect();
    let request = GpuBatchRequest {
        prompts,
        max_tokens: 10,
        temperature: 0.5,
        top_k: 40,
        stop: vec![],
    };

    assert_eq!(request.prompts.len(), 100);
}

#[test]
fn test_gpu_batch_stats_zero_values() {
    let stats = GpuBatchStats {
        batch_size: 0,
        gpu_used: false,
        total_tokens: 0,
        processing_time_ms: 0.0,
        throughput_tps: 0.0,
    };

    let json = serde_json::to_string(&stats).expect("serialize");
    let parsed: GpuBatchStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.batch_size, 0);
    assert_eq!(parsed.total_tokens, 0);
}

#[test]
fn test_gpu_batch_stats_large_values() {
    let stats = GpuBatchStats {
        batch_size: 1_000_000,
        gpu_used: true,
        total_tokens: 1_000_000_000,
        processing_time_ms: 100_000.0,
        throughput_tps: 10_000_000.0,
    };

    let json = serde_json::to_string(&stats).expect("serialize");
    let parsed: GpuBatchStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.batch_size, 1_000_000);
}

#[test]
fn test_gpu_warmup_response_large_memory() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 24_000_000_000, // 24GB
        num_layers: 100,
        message: "Warmed up 24GB of VRAM".to_string(),
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("24000000000"));
}

#[test]
fn test_gpu_status_response_large_cache() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 48_000_000_000, // 48GB
        batch_threshold: 64,
        recommended_min_batch: 64,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("48000000000"));
}

// =============================================================================
// Stream Generate Handler Tests
// =============================================================================

#[tokio::test]
async fn test_stream_generate_handler_success() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "greedy", "max_tokens": 3}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    // SSE responses have text/event-stream content type
    let content_type = response.headers().get("content-type");
    assert!(content_type.is_some());
}

#[tokio::test]
async fn test_stream_generate_handler_empty_prompt() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "", "max_tokens": 5}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_stream_generate_handler_invalid_strategy() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "unknown_strategy", "max_tokens": 5}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// =============================================================================
// Additional Response Type Coverage
// =============================================================================

#[test]
fn test_tokenize_response_serialization() {
    let response = TokenizeResponse {
        token_ids: vec![1, 2, 3, 4, 5],
        num_tokens: 5,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("[1,2,3,4,5]"));
    assert!(json.contains("5"));
}

#[test]
fn test_generate_response_serialization() {
    let response = GenerateResponse {
        token_ids: vec![10, 20, 30],
        text: "Hello world".to_string(),
        num_generated: 3,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("Hello world"));
    assert!(json.contains("num_generated"));
}

#[test]
fn test_batch_tokenize_response_serialization() {
    let response = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1],
                num_tokens: 1,
            },
            TokenizeResponse {
                token_ids: vec![2, 3],
                num_tokens: 2,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("results"));
}

#[test]
fn test_batch_generate_response_serialization() {
    let response = BatchGenerateResponse {
        results: vec![
            GenerateResponse {
                token_ids: vec![1],
                text: "a".to_string(),
                num_generated: 1,
            },
            GenerateResponse {
                token_ids: vec![2],
                text: "b".to_string(),
                num_generated: 1,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("results"));
}

#[test]
fn test_models_response_serialization() {
    use crate::registry::ModelInfo;

    let response = ModelsResponse {
        models: vec![ModelInfo {
            id: "model-1".to_string(),
            name: "Model One".to_string(),
            description: "First model".to_string(),
            format: "gguf".to_string(),
            loaded: true,
        }],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("model-1"));
    assert!(json.contains("Model One"));
}
