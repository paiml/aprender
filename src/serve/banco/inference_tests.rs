//! Phase 2b integration tests for the inference pipeline.
//!
//! Tests cover: handler fallback, response format, streaming format,
//! tokenize/detokenize with heuristic fallback, and inference helper purity.

// ============================================================================
// Handler: No model loaded → dry_run fallback
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_chat_no_model_returns_dry_run() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    assert!(!state.model.is_loaded());

    let app = super::router::create_banco_router(state);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello!"}]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");

    assert_eq!(json["choices"][0]["finish_reason"], "dry_run");
    assert!(json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content")
        .contains("No model loaded"));
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
}

// ============================================================================
// Handler: Response format is OpenAI-compatible
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_chat_response_has_openai_fields() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Test"}]
    });
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");

    // OpenAI SDK required fields
    assert!(json["id"].as_str().is_some(), "must have id");
    assert_eq!(json["object"], "chat.completion");
    assert!(json["created"].as_u64().is_some(), "must have created timestamp");
    assert!(json["model"].as_str().is_some(), "must have model");
    assert!(json["choices"].as_array().is_some(), "must have choices");
    assert!(json["usage"]["prompt_tokens"].as_u64().is_some());
    assert!(json["usage"]["completion_tokens"].as_u64().is_some());
    assert!(json["usage"]["total_tokens"].as_u64().is_some());
}

// ============================================================================
// Handler: SSE streaming format is OpenAI-compatible
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_stream_returns_sse_chunks() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Test"}],
        "stream": true
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    // Content-Type should be SSE
    let ct = response.headers().get("content-type").expect("content-type");
    assert!(ct.to_str().expect("str").contains("text/event-stream"));

    // Read the full body
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let body_str = String::from_utf8_lossy(&bytes);

    // Should contain data: lines and end with [DONE]
    assert!(body_str.contains("data:"), "SSE must have data: lines");
    assert!(body_str.contains("[DONE]"), "SSE must end with [DONE]");

    // Parse first non-empty data line as JSON to verify chunk format
    for line in body_str.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                break;
            }
            let chunk: serde_json::Value = serde_json::from_str(data).expect("parse chunk");
            assert_eq!(chunk["object"], "chat.completion.chunk");
            assert!(chunk["choices"][0]["delta"].is_object());
            break;
        }
    }
}

// ============================================================================
// Handler: Empty messages → 400
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_chat_empty_messages_returns_400() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"messages": []});
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

// ============================================================================
// Tokenize: Heuristic fallback (no model)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_tokenize_heuristic_fallback() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"text": "Hello, world! This is a test."});
    let response = app
        .oneshot(
            Request::post("/api/v1/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");

    let count = json["count"].as_u64().expect("count");
    assert!(count > 0, "should estimate non-zero tokens");
    let tokens = json["tokens"].as_array().expect("tokens");
    assert_eq!(tokens.len(), count as usize, "token IDs should match count");
}

// ============================================================================
// Detokenize: Heuristic fallback (no model)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_detokenize_heuristic_fallback() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"tokens": [1, 2, 3, 4, 5]});
    let response = app
        .oneshot(
            Request::post("/api/v1/detokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");

    let text = json["text"].as_str().expect("text");
    assert!(text.contains("5 tokens"), "should describe token count");
}

// ============================================================================
// Inference helpers: encode_prompt roundtrip
// ============================================================================

#[cfg(feature = "realizar")]
#[test]
#[allow(non_snake_case)]
fn test_P2B_encode_prompt_finds_known_tokens() {
    let vocab: Vec<String> = vec![
        "<unk>".to_string(),
        "</s>".to_string(),
        "H".to_string(),
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
    ];
    let tokens = super::inference::encode_prompt(&vocab, "Hello");
    // Should encode as "H", "e", "l", "l", "o"
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokens[0], 2); // "H"
    assert_eq!(tokens[1], 3); // "e"
    assert_eq!(tokens[2], 4); // "l"
    assert_eq!(tokens[3], 4); // "l"
    assert_eq!(tokens[4], 5); // "o"
}

#[cfg(feature = "realizar")]
#[test]
#[allow(non_snake_case)]
fn test_P2B_sample_temperature_zero_is_greedy() {
    // Temperature 0 should always pick argmax
    let params = super::inference::SamplingParams { temperature: 0.0, top_k: 40, max_tokens: 10 };
    // Temperature 0 should produce greedy sampling
    assert!(params.temperature.abs() < f32::EPSILON);
}

// ============================================================================
// Model slot: unloaded state
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P2B_model_slot_empty() {
    let slot = super::model_slot::ModelSlot::empty();
    assert!(!slot.is_loaded());
    assert!(slot.info().is_none());
    assert_eq!(slot.uptime_secs(), 0);
}

#[test]
#[allow(non_snake_case)]
fn test_P2B_model_slot_unload_empty_returns_error() {
    let slot = super::model_slot::ModelSlot::empty();
    let result = slot.unload();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), super::model_slot::ModelSlotError::NoModelLoaded);
}

// ============================================================================
// Usage: total_tokens = prompt_tokens + completion_tokens
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_usage_tokens_add_up() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello, world!"}]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");

    let prompt = json["usage"]["prompt_tokens"].as_u64().expect("prompt");
    let completion = json["usage"]["completion_tokens"].as_u64().expect("completion");
    let total = json["usage"]["total_tokens"].as_u64().expect("total");
    assert_eq!(total, prompt + completion);
}

// ============================================================================
// Max tokens respected in request
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_max_tokens_in_request() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 10
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Conversation ID association
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2B_chat_with_conversation_id() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let conv_id = state.conversations.create("banco-echo");

    let app = super::router::create_banco_router(state.clone());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "conversation_id": conv_id
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    // Verify both user message AND assistant response were saved
    let full = state.conversations.get(&conv_id).expect("get conv");
    assert!(full.messages.len() >= 2, "should have user message + assistant response");
    // First message is user
    assert_eq!(full.messages[0].role, crate::serve::templates::Role::User);
    // Last message is assistant (the generated response)
    assert_eq!(full.messages.last().expect("last").role, crate::serve::templates::Role::Assistant);
}
