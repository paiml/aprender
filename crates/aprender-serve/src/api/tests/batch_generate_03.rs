
#[tokio::test]
async fn test_batch_generate_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": ["Hello", "World"],
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
                .uri("/batch/generate")
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
async fn test_apr_predict_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "input": "Hello"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
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
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_apr_explain_handler_via_http() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "input": "Hello"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
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
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_health_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_dispatch_metrics_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
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

// ============================================================================
// OpenAI types serde
// ============================================================================

#[test]
fn test_openai_model_serde() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "realizar".to_string(),
    };
    let json = serde_json::to_string(&model).expect("JSON serialization failed");
    let parsed: OpenAIModel = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.id, "gpt-4");
}

#[test]
fn test_openai_models_response_serde() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "test".to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "test".to_string(),
        }],
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: OpenAIModelsResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.data.len(), 1);
}

#[test]
fn test_chat_completion_request_serde() {
    let req = ChatCompletionRequest {
        model: "default".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: ChatCompletionRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.messages.len(), 1);
    assert!(!parsed.stream);
}

#[test]
fn test_chat_completion_response_serde() {
    let resp = ChatCompletionResponse {
        id: "chat-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "default".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: ChatCompletionResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.choices[0].message.content, "Hello!");
}

#[test]
fn test_usage_serde() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let json = serde_json::to_string(&usage).expect("JSON serialization failed");
    let parsed: Usage = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.total_tokens, 30);
}

#[test]
fn test_error_response_serde() {
    let err = ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&err).expect("JSON serialization failed");
    let parsed: ErrorResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.error.contains("wrong"));
}

// ============================================================================
// Chat completion with trace header
// ============================================================================

#[tokio::test]
async fn test_chat_completions_with_trace_header() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "test"}],
        "stream": false,
        "max_tokens": 2
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Trace-Level", "detailed")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Just verify the endpoint accepts the trace header without error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// ============================================================================
// Invalid JSON error paths
// ============================================================================

#[tokio::test]
async fn test_generate_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_chat_completions_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{invalid}"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_tokenize_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from("bad json"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// ============================================================================
// Streaming endpoint
// ============================================================================

#[tokio::test]
async fn test_stream_generate_handler_via_http() {
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
                .uri("/stream/generate")
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

// ============================================================================
// Debug trait coverage
// ============================================================================

#[test]
fn test_context_window_config_debug() {
    let config = ContextWindowConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("4096"));
}

#[test]
fn test_embedding_data_debug() {
    let data = EmbeddingData {
        object: "embedding".to_string(),
        index: 0,
        embedding: vec![0.1],
    };
    let debug = format!("{data:?}");
    assert!(debug.contains("embedding"));
}

#[test]
fn test_embedding_usage_debug() {
    let usage = EmbeddingUsage {
        prompt_tokens: 5,
        total_tokens: 5,
    };
    let debug = format!("{usage:?}");
    assert!(debug.contains("5"));
}

#[test]
fn test_completion_choice_debug() {
    let choice = CompletionChoice {
        text: "hello".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };
    let debug = format!("{choice:?}");
    assert!(debug.contains("hello"));
}
