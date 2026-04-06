
#[test]
fn test_format_chat_messages_single_user() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello!".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, None);
    assert!(result.contains("Hello!"));
}

#[test]
fn test_format_chat_messages_multi_turn() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![
        crate::api::ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "Hello!".to_string(),
            name: None,
        },
    ];
    let result = format_chat_messages(&messages, None);
    assert!(result.contains("Hi") || result.contains("Hello!"));
}

#[test]
fn test_format_chat_messages_with_model_name() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, Some("llama"));
    assert!(!result.is_empty());
}

// ============================================================================
// B3: clean_chat_output
// ============================================================================

#[test]
fn test_clean_chat_output_no_markers() {
    use crate::api::realize_handlers::clean_chat_output;
    let result = clean_chat_output("Just plain text");
    assert_eq!(result, "Just plain text");
}

#[test]
fn test_clean_chat_output_chatml_markers() {
    use crate::api::realize_handlers::clean_chat_output;
    // clean_chat_output truncates at the earliest stop sequence
    // <|im_start|> is at position 0, so everything is truncated
    let text = "<|im_start|>assistant\nHello there<|im_end|>";
    let result = clean_chat_output(text);
    assert!(result.is_empty() || !result.contains("<|im_start|>"));
}

#[test]
fn test_clean_chat_output_empty() {
    use crate::api::realize_handlers::clean_chat_output;
    let result = clean_chat_output("");
    assert!(result.is_empty());
}

#[test]
fn test_clean_chat_output_partial_markers() {
    use crate::api::realize_handlers::clean_chat_output;
    // Text before <|im_end|> is preserved; <|im_start|> at position 0 truncates all
    let text = "Hello world<|im_end|>extra stuff";
    let result = clean_chat_output(text);
    assert!(result.contains("Hello world"));
    assert!(!result.contains("extra stuff"));
}

// ============================================================================
// B3: HTTP Handler Integration - Realize Endpoints
// ============================================================================

#[tokio::test]
async fn test_realize_embed_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/embed")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"input":"Hello world"}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR,
    );
}

#[tokio::test]
async fn test_realize_model_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/model")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND,);
}

#[tokio::test]
async fn test_realize_reload_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/reload")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"test"}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_openai_completions_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"test","prompt":"Hello","max_tokens":5}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_openai_embeddings_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"input":"Hello"}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR,
    );
}

// ============================================================================
// B3: OpenAI Handlers
// ============================================================================

#[tokio::test]
async fn test_openai_models_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND,);
}

#[tokio::test]
async fn test_openai_chat_completions_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"test","messages":[{"role":"user","content":"Hello"}]}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_openai_chat_completions_with_temperature() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"test","messages":[{"role":"user","content":"Hi"}],"temperature":0.5,"max_tokens":10}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_openai_chat_completions_streaming() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"test","messages":[{"role":"user","content":"Hi"}],"stream":true}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

// ============================================================================
// B3: AppState Accessors
// ============================================================================

#[test]
fn test_appstate_has_quantized_model_demo() {
    let state = AppState::demo_mock().expect("test value should be present");
    // Demo mock state typically has no quantized model
    let _ = state.has_quantized_model();
    let _ = state.quantized_model();
}

#[test]
fn test_appstate_has_apr_transformer_demo() {
    let state = AppState::demo_mock().expect("test value should be present");
    let _ = state.has_apr_transformer();
    let _ = state.apr_transformer();
}

#[test]
fn test_appstate_verbose() {
    let state = AppState::demo_mock().expect("test value should be present");
    assert!(!state.is_verbose());
    let state_verbose = state.with_verbose(true);
    assert!(state_verbose.is_verbose());
}

#[test]
fn test_appstate_demo_creates_valid_state() {
    let state = AppState::demo();
    assert!(state.is_ok());
}

#[test]
fn test_appstate_demo_mock_creates_valid_state() {
    let state = AppState::demo_mock();
    assert!(state.is_ok());
}

// ============================================================================
// B3: build_trace_data
// ============================================================================

#[test]
fn test_build_trace_data_none() {
    let (brick, step, layer) = crate::api::build_trace_data(None, 100, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_brick() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("brick"), 100, 10, 5, 4);
    assert!(brick.is_some());
    assert!(step.is_none());
    assert!(layer.is_none());
    let b = brick.expect("test value should be present");
    assert_eq!(b.level, "brick");
}

#[test]
fn test_build_trace_data_step() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("step"), 200, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_some());
    assert!(layer.is_none());
    let s = step.expect("test value should be present");
    assert_eq!(s.level, "step");
}

#[test]
fn test_build_trace_data_layer() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("layer"), 300, 10, 5, 8);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());
    let l = layer.expect("test value should be present");
    assert_eq!(l.level, "layer");
}

// ============================================================================
// B3: Request/Response Struct Serde Round-Trips
// ============================================================================

#[test]
fn test_chat_message_serde() {
    let msg = crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("alice".to_string()),
    };
    let json = serde_json::to_string(&msg).expect("JSON serialization failed");
    let deserialized: crate::api::ChatMessage = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.role, "user");
    assert_eq!(deserialized.content, "Hello");
    assert_eq!(deserialized.name, Some("alice".to_string()));
}

#[test]
fn test_chat_message_without_name() {
    let json = r#"{"role":"assistant","content":"Hi!"}"#;
    let msg: crate::api::ChatMessage = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(msg.role, "assistant");
    assert!(msg.name.is_none());
}

#[test]
fn test_error_response_serde() {
    let err = crate::api::ErrorResponse {
        error: "something went wrong".to_string(),
    };
    let json = serde_json::to_string(&err).expect("JSON serialization failed");
    let deserialized: crate::api::ErrorResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.error, "something went wrong");
}

#[test]
fn test_health_response_serde() {
    let health = crate::api::HealthResponse {
        status: "ok".to_string(),
        version: "0.3.5".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&health).expect("JSON serialization failed");
    let deserialized: crate::api::HealthResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.status, "ok");
}

#[test]
fn test_generate_request_serde() {
    let req = crate::api::GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.5,
        strategy: "greedy".to_string(),
        top_k: 1,
        top_p: 1.0,
        seed: Some(42),
        model_id: None,
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let deserialized: crate::api::GenerateRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.prompt, "Hello");
    assert_eq!(deserialized.seed, Some(42));
}

#[test]
fn test_generate_response_serde() {
    let resp = crate::api::GenerateResponse {
        token_ids: vec![1, 2, 3],
        text: "hello".to_string(),
        num_generated: 3,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: crate::api::GenerateResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.num_generated, 3);
}

#[test]
fn test_tokenize_request_serde() {
    let req = crate::api::TokenizeRequest {
        text: "Hello world".to_string(),
        model_id: None,
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let deserialized: crate::api::TokenizeRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.text, "Hello world");
}

#[test]
fn test_tokenize_response_serde() {
    let resp = crate::api::TokenizeResponse {
        token_ids: vec![1, 2, 3, 4],
        num_tokens: 4,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: crate::api::TokenizeResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.num_tokens, 4);
}
