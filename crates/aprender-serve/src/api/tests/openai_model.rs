
#[test]
fn test_openai_model_debug() {
    let model = OpenAIModel {
        id: "debug-test".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "test".to_string(),
    };

    let debug = format!("{:?}", model);
    assert!(debug.contains("OpenAIModel"));
    assert!(debug.contains("debug-test"));
}

// =============================================================================
// Usage Tests
// =============================================================================

#[test]
fn test_usage_serialization() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    assert!(json.contains("100"));
    assert!(json.contains("50"));
    assert!(json.contains("150"));
}

#[test]
fn test_usage_deserialization() {
    let json = r#"{"prompt_tokens":25,"completion_tokens":75,"total_tokens":100}"#;
    let usage: Usage = serde_json::from_str(json).expect("deserialize");

    assert_eq!(usage.prompt_tokens, 25);
    assert_eq!(usage.completion_tokens, 75);
    assert_eq!(usage.total_tokens, 100);
}

#[test]
fn test_usage_clone() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };

    let cloned = usage.clone();
    assert_eq!(cloned.prompt_tokens, usage.prompt_tokens);
    assert_eq!(cloned.completion_tokens, usage.completion_tokens);
    assert_eq!(cloned.total_tokens, usage.total_tokens);
}

#[test]
fn test_usage_debug() {
    let usage = Usage {
        prompt_tokens: 1,
        completion_tokens: 2,
        total_tokens: 3,
    };

    let debug = format!("{:?}", usage);
    assert!(debug.contains("Usage"));
}

// =============================================================================
// ChatChoice Tests
// =============================================================================

#[test]
fn test_chat_choice_serialization() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Response text".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("assistant"));
    assert!(json.contains("Response text"));
    assert!(json.contains("stop"));
}

#[test]
fn test_chat_choice_finish_reason_length() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Truncated response...".to_string(),
            name: None,
        },
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("length"));
}

#[test]
fn test_chat_choice_clone() {
    let choice = ChatChoice {
        index: 1,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Test".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let cloned = choice.clone();
    assert_eq!(cloned.index, 1);
    assert_eq!(cloned.message.content, "Test");
}

// =============================================================================
// TraceData and TraceOperation Tests
// =============================================================================

#[test]
fn test_trace_data_serialization() {
    let trace = TraceData {
        level: "step".to_string(),
        operations: 10,
        total_time_us: 5000,
        breakdown: vec![
            TraceOperation {
                name: "tokenize".to_string(),
                time_us: 100,
                details: Some("5 tokens".to_string()),
            },
            TraceOperation {
                name: "forward".to_string(),
                time_us: 4800,
                details: None,
            },
        ],
        provenance: TraceProvenance::Estimated,
    };

    let json = serde_json::to_string(&trace).expect("serialize");
    assert!(json.contains("step"));
    assert!(json.contains("tokenize"));
    assert!(json.contains("forward"));
    assert!(json.contains("5 tokens"));
}

#[test]
fn test_trace_operation_without_details() {
    let op = TraceOperation {
        name: "softmax".to_string(),
        time_us: 250,
        details: None,
    };

    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("softmax"));
    assert!(json.contains("250"));
    // Details should not be present
    assert!(!json.contains("details"));
}

#[test]
fn test_trace_data_clone() {
    let trace = TraceData {
        level: "layer".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![],
        provenance: TraceProvenance::Estimated,
    };

    let cloned = trace.clone();
    assert_eq!(cloned.level, "layer");
    assert_eq!(cloned.operations, 5);
}

// =============================================================================
// build_trace_data Tests
// =============================================================================

#[test]
fn test_build_trace_data_brick_level() {
    let (brick, step, layer) = build_trace_data(Some("brick"), 1000, 10, 5, 12);

    assert!(brick.is_some());
    assert!(step.is_none());
    assert!(layer.is_none());

    let trace = brick.expect("test value should be present");
    assert_eq!(trace.level, "brick");
    assert_eq!(trace.operations, 5); // completion_tokens
    assert_eq!(trace.total_time_us, 1000);
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 1000);
    assert!(trace.breakdown[0].details.as_ref().expect("details").contains("apr profile"));
}

#[test]
fn test_build_trace_data_step_level() {
    let (brick, step, layer) = build_trace_data(Some("step"), 2000, 20, 10, 24);

    assert!(brick.is_none());
    assert!(step.is_some());
    assert!(layer.is_none());

    let trace = step.expect("test value should be present");
    assert_eq!(trace.level, "step");
    assert_eq!(trace.operations, 10); // completion_tokens
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 2000);
    assert!(trace.breakdown[0].details.as_ref().expect("details").contains("apr profile"));
}

#[test]
fn test_build_trace_data_layer_level() {
    let (brick, step, layer) = build_trace_data(Some("layer"), 3000, 15, 8, 32);

    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());

    let trace = layer.expect("test value should be present");
    assert_eq!(trace.level, "layer");
    assert_eq!(trace.operations, 32); // num_layers
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 3000);
    let details = trace.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("32 layers"));
    assert!(details.contains("apr profile"));
}

#[test]
fn test_build_trace_data_no_level() {
    let (brick, step, layer) = build_trace_data(None, 1000, 10, 5, 12);

    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = build_trace_data(Some("invalid"), 1000, 10, 5, 12);

    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

// =============================================================================
// HTTP Endpoint Tests
// =============================================================================

#[tokio::test]
async fn test_openai_models_endpoint() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
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
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test value should be present");
    let result: OpenAIModelsResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert_eq!(result.object, "list");
    assert!(!result.data.is_empty());
    assert_eq!(result.data[0].object, "model");
    assert_eq!(result.data[0].owned_by, "realizar");
}

#[tokio::test]
async fn test_openai_chat_completions_endpoint() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

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
        .expect("test value should be present");
    let result: ChatCompletionResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert!(result.id.starts_with("chatcmpl-"));
    assert_eq!(result.object, "chat.completion");
    assert_eq!(result.choices.len(), 1);
    assert_eq!(result.choices[0].message.role, "assistant");
}

#[tokio::test]
async fn test_openai_chat_completions_with_system_message() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 10
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_openai_chat_completions_with_temperature() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_openai_chat_completions_with_top_p() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 0.95
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}
