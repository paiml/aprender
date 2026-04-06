
// ============================================================================
// format_chat_messages Tests
// ============================================================================

#[test]
fn test_format_chat_messages_chatml() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hi!".to_string(),
            name: None,
        },
    ];

    let formatted = format_chat_messages(&messages, Some("qwen"));
    assert!(formatted.contains("<|im_start|>"));
    assert!(formatted.contains("system"));
    assert!(formatted.contains("You are helpful."));
    assert!(formatted.contains("user"));
    assert!(formatted.contains("Hi!"));
}

#[test]
fn test_format_chat_messages_alpaca() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let formatted = format_chat_messages(&messages, Some("alpaca"));
    assert!(formatted.contains("### Instruction:"));
}

#[test]
fn test_format_chat_messages_empty() {
    let messages: Vec<ChatMessage> = vec![];
    let formatted = format_chat_messages(&messages, None);
    assert!(formatted.is_empty() || formatted.contains("assistant"));
}

// ============================================================================
// build_trace_data Tests
// ============================================================================

#[test]
fn test_build_trace_data_none() {
    let (brick, step, layer) = build_trace_data(None, 1000, 10, 5, 12);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_brick() {
    let (brick, step, layer) = build_trace_data(Some("brick"), 5000, 20, 10, 28);
    assert!(brick.is_some());
    let trace = brick.expect("test value should be present");
    assert_eq!(trace.level, "brick");
    assert!(trace.total_time_us > 0);
    assert!(!trace.breakdown.is_empty());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_step() {
    let (brick, step, layer) = build_trace_data(Some("step"), 5000, 20, 10, 28);
    // "step" level only returns step trace
    assert!(brick.is_none());
    assert!(step.is_some());
    let step_trace = step.expect("test value should be present");
    assert_eq!(step_trace.level, "step");
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_layer() {
    let (brick, step, layer) = build_trace_data(Some("layer"), 5000, 20, 10, 28);
    // "layer" level only returns layer trace
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());
    let layer_trace = layer.expect("test value should be present");
    assert_eq!(layer_trace.level, "layer");
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = build_trace_data(Some("unknown"), 5000, 20, 10, 28);
    // Unknown levels should return None for all
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

// ============================================================================
// OpenAIModel Tests
// ============================================================================

#[test]
fn test_openai_model_structure() {
    let model = OpenAIModel {
        id: "qwen2.5-coder-1.5b".to_string(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "realizar".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    assert!(json.contains("qwen2.5-coder-1.5b"));
    assert!(json.contains("1700000000"));
}

// ============================================================================
// Invalid Request Tests (Bad Request 400)
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
                .expect("test"),
        )
        .await
        .expect("test");

    // Should get a 400 or 422 for invalid JSON
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_generate_missing_required_field() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"temperature": 0.5}"#))
                .expect("test"),
        )
        .await
        .expect("test");

    // Missing required 'prompt' field
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// ============================================================================
// Method Not Allowed Tests (405)
// ============================================================================

#[tokio::test]
async fn test_generate_get_method_not_allowed() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/generate")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    // POST-only endpoint should reject GET
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_chat_completions_get_method_not_allowed() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/chat/completions")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ============================================================================
// Not Found Tests (404)
// ============================================================================

#[tokio::test]
async fn test_unknown_endpoint_404() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/unknown/endpoint")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_v1_unknown_404() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/unknown")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ============================================================================
// Content-Type Tests
// ============================================================================

#[tokio::test]
async fn test_generate_wrong_content_type() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "text/plain")
                .body(Body::from("prompt=test"))
                .expect("test"),
        )
        .await
        .expect("test");

    // Should get error for wrong content type
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// ============================================================================
// Deep Coverage: ChatDelta and ChatChunkChoice
// ============================================================================

#[test]
fn test_chunk_delta_role_only() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("assistant"));
}

#[test]
fn test_chunk_delta_content_only() {
    let delta = ChatDelta {
        role: None,
        content: Some("test content".to_string()),
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("test content"));
}

#[test]
fn test_chunk_choice_structure() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("hi".to_string()),
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("delta"));
    assert!(json.contains("hi"));
}
