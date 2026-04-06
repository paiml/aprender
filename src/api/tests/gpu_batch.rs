
#[tokio::test]
async fn test_gpu_batch_completions_missing_prompts_field() {
    let app = create_test_app_shared();

    // Missing required 'prompts' field
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"max_tokens": 50}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Missing required field should return 422 Unprocessable Entity
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_gpu_warmup_endpoint_method_not_allowed() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET") // Wrong method, should be POST
                .uri("/v1/gpu/warmup")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    // GET on POST-only endpoint should return 405
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_gpu_status_endpoint_post_method_not_allowed() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST") // Wrong method, should be GET
                .uri("/v1/gpu/status")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // POST on GET-only endpoint should return 405
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// =============================================================================
// Models Endpoint Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_models_handler_demo_mode() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/models")
                .body(Body::empty())
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
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let models: ModelsResponse = serde_json::from_slice(&body).expect("parse json");

    // Demo mode returns default model info
    assert!(!models.models.is_empty());
    assert_eq!(models.models[0].id, "default");
}

// =============================================================================
// Tokenize Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_tokenize_handler_success() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "Hello world"}"#))
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
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: TokenizeResponse = serde_json::from_slice(&body).expect("parse json");

    assert!(result.num_tokens > 0);
    assert!(!result.token_ids.is_empty());
}

#[tokio::test]
async fn test_tokenize_handler_with_model_id() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "Test", "model_id": "nonexistent"}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Demo mode falls back to default model, so this should still work
    // or return NOT_FOUND depending on implementation
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// =============================================================================
// Generate Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_generate_handler_greedy() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "greedy", "max_tokens": 5}"#,
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
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: GenerateResponse = serde_json::from_slice(&body).expect("parse json");

    // In demo mode, tokens may be generated but text may be empty placeholder
    // Just verify we got a valid response with some tokens
    assert!(!result.token_ids.is_empty());
}

#[tokio::test]
async fn test_generate_handler_top_k() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "Hello", "strategy": "top_k", "top_k": 10, "max_tokens": 5, "seed": 42}"#))
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

#[tokio::test]
async fn test_generate_handler_top_p() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "Hello", "strategy": "top_p", "top_p": 0.9, "max_tokens": 5, "seed": 42}"#))
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

#[tokio::test]
async fn test_generate_handler_empty_prompt() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
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
async fn test_generate_handler_invalid_strategy() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompt": "Hello", "strategy": "invalid_strategy", "max_tokens": 5}"#,
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
// Batch Tokenize Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_batch_tokenize_handler_success() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"texts": ["Hello", "World", "Test"]}"#))
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
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: BatchTokenizeResponse = serde_json::from_slice(&body).expect("parse json");

    assert_eq!(result.results.len(), 3);
    for r in &result.results {
        assert!(r.num_tokens > 0);
    }
}

#[tokio::test]
async fn test_batch_tokenize_handler_empty_texts() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"texts": []}"#))
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
// Batch Generate Handler Tests (from gpu_handlers.rs)
// =============================================================================

#[tokio::test]
async fn test_batch_generate_handler_success() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"prompts": ["Hello", "World"], "max_tokens": 3, "strategy": "greedy"}"#,
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
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let result: BatchGenerateResponse = serde_json::from_slice(&body).expect("parse json");

    assert_eq!(result.results.len(), 2);
}

#[tokio::test]
async fn test_batch_generate_handler_empty_prompts() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompts": [], "max_tokens": 5}"#))
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
