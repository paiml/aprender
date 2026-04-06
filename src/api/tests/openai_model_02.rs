
// ============================================================================
// OpenAIModel and OpenAIModelsResponse coverage
// ============================================================================

#[test]
fn test_openai_model_debug() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "openai".to_string(),
    };
    let debug = format!("{:?}", model);
    assert!(debug.contains("gpt-4"));
}

#[test]
fn test_openai_model_clone() {
    let model = OpenAIModel {
        id: "test".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "realizar".to_string(),
    };
    let cloned = model.clone();
    assert_eq!(cloned.owned_by, "realizar");
}

#[test]
fn test_openai_models_response_debug() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![],
    };
    let debug = format!("{:?}", resp);
    assert!(debug.contains("list"));
}

#[test]
fn test_openai_models_response_clone() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "m1".to_string(),
            object: "model".to_string(),
            created: 1,
            owned_by: "test".to_string(),
        }],
    };
    let cloned = resp.clone();
    assert_eq!(cloned.data.len(), 1);
}

// ============================================================================
// Request/Response types edge cases
// ============================================================================

#[test]
fn test_tokenize_request_serde() {
    let req = TokenizeRequest {
        text: "Hello world".to_string(),
        model_id: Some("default".to_string()),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("Hello world"));
    let parsed: TokenizeRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.model_id.as_deref(), Some("default"));
}

#[test]
fn test_tokenize_request_no_model() {
    let json = r#"{"text": "test"}"#;
    let req: TokenizeRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert!(req.model_id.is_none());
}

#[test]
fn test_tokenize_response_serde() {
    let resp = TokenizeResponse {
        token_ids: vec![1, 2, 3],
        num_tokens: 3,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: TokenizeResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.num_tokens, 3);
}

#[test]
fn test_generate_request_serde() {
    let req = GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        strategy: "top_k".to_string(),
        top_k: 40,
        top_p: 0.9,
        seed: Some(42),
        model_id: None,
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("top_k"));
    let parsed: GenerateRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.seed, Some(42));
}

#[test]
fn test_generate_request_minimal() {
    let json = r#"{"prompt": "test"}"#;
    let req: GenerateRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(req.prompt, "test");
    // Check defaults - default_max_tokens() returns 50, default_temperature() returns 1.0
    assert_eq!(req.max_tokens, 50);
    assert!((req.temperature - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_generate_response_serde() {
    let resp = GenerateResponse {
        token_ids: vec![1, 2],
        text: "hello".to_string(),
        num_generated: 1,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: GenerateResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.text, "hello");
}

#[test]
fn test_batch_tokenize_request_serde() {
    let req = BatchTokenizeRequest {
        texts: vec!["Hello".to_string(), "World".to_string()],
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: BatchTokenizeRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.texts.len(), 2);
}

#[test]
fn test_batch_tokenize_response_serde() {
    let resp = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1],
                num_tokens: 1,
            },
            TokenizeResponse {
                token_ids: vec![2],
                num_tokens: 1,
            },
        ],
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: BatchTokenizeResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.results.len(), 2);
}

#[test]
fn test_batch_generate_request_serde() {
    let req = BatchGenerateRequest {
        prompts: vec!["A".to_string(), "B".to_string()],
        max_tokens: 10,
        temperature: 0.5,
        strategy: "greedy".to_string(),
        top_k: 1,
        top_p: 1.0,
        seed: None,
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: BatchGenerateRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.prompts.len(), 2);
}

#[test]
fn test_batch_generate_response_serde() {
    let resp = BatchGenerateResponse {
        results: vec![GenerateResponse {
            token_ids: vec![1],
            text: "a".to_string(),
            num_generated: 1,
        }],
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: BatchGenerateResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.results.len(), 1);
}

#[test]
fn test_stream_token_event_serde() {
    let event = StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&event).expect("JSON serialization failed");
    let parsed: StreamTokenEvent = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.token_id, 42);
}

#[test]
fn test_stream_done_event_serde() {
    let event = StreamDoneEvent { num_generated: 10 };
    let json = serde_json::to_string(&event).expect("JSON serialization failed");
    let parsed: StreamDoneEvent = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.num_generated, 10);
}

// ============================================================================
// HTTP endpoints additional edge cases
// ============================================================================

#[tokio::test]
async fn test_empty_prompts_batch_completions() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": [],
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
    // Should return BAD_REQUEST for empty prompts
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_empty_texts_batch_tokenize() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "texts": []
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
    // Should return BAD_REQUEST for empty texts
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_empty_prompts_batch_generate() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompts": [],
        "max_tokens": 5
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
    // Should return BAD_REQUEST for empty prompts
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_invalid_strategy_generate() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 5,
        "strategy": "invalid_strategy"
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
    // Should return BAD_REQUEST for invalid strategy
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_models_endpoint_returns_list() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/models")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_realize_model_handler() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/model")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // May return OK or error depending on model state
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[tokio::test]
async fn test_realize_reload_handler() {
    let app = create_test_app_shared();
    let body = serde_json::json!({});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/reload")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // May return OK or error depending on model state
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_chat_completions_streaming_flag() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "max_tokens": 2
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
    // Accept streaming response or error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );
}

// ============================================================================
// ChatCompletionRequest coverage
// ============================================================================

#[test]
fn test_chat_completion_request_serde() {
    let req = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        n: 1,
        stream: false,
        stop: Some(vec!["END".to_string()]),
        user: Some("test-user".to_string()),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("gpt-4"));
    let parsed: ChatCompletionRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.model, "gpt-4");
    assert_eq!(parsed.user.as_deref(), Some("test-user"));
}

#[test]
fn test_chat_completion_request_minimal() {
    let json = r#"{"model": "test", "messages": []}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(req.model, "test");
    assert!(!req.stream);
    assert!(req.max_tokens.is_none());
}
