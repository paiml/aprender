//! Text completions + model detail endpoint tests.

use super::handlers_completions::{CompletionRequest, CompletionResponse, PromptInput};

// ============================================================================
// Type tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_CMPL_001_prompt_single_string() {
    let json = serde_json::json!({"prompt": "Hello world"});
    let req: CompletionRequest = serde_json::from_value(json).expect("parse");
    matches!(req.prompt, PromptInput::Single(ref s) if s == "Hello world");
}

#[test]
#[allow(non_snake_case)]
fn test_CMPL_002_prompt_array() {
    let json = serde_json::json!({"prompt": ["Hello", "World"]});
    let req: CompletionRequest = serde_json::from_value(json).expect("parse");
    matches!(req.prompt, PromptInput::Multiple(ref v) if v.len() == 2);
}

#[test]
#[allow(non_snake_case)]
fn test_CMPL_003_response_serializes() {
    let resp = CompletionResponse {
        id: "cmpl-1".to_string(),
        object: "text_completion".to_string(),
        created: 123,
        model: "test".to_string(),
        choices: vec![super::handlers_completions::CompletionChoice {
            text: "Hello".to_string(),
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: super::handlers_completions::CompletionUsage {
            prompt_tokens: 2,
            completion_tokens: 1,
            total_tokens: 3,
        },
    };
    let json = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["text"], "Hello");
}

#[test]
#[allow(non_snake_case)]
fn test_CMPL_004_request_with_options() {
    let json = serde_json::json!({
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": ["\n"]
    });
    let req: CompletionRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.max_tokens, Some(100));
    assert!((req.temperature.unwrap() - 0.5).abs() < f32::EPSILON);
    assert_eq!(req.stop, Some(vec!["\n".to_string()]));
}

// ============================================================================
// Endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CMPL_HDL_001_text_completion() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "prompt": "The capital of France is",
        "max_tokens": 50
    });
    let response = app
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["object"], "text_completion");
    assert!(json["choices"][0]["text"].as_str().is_some());
    assert!(json["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CMPL_HDL_002_model_detail_not_found() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/v1/models/nonexistent").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CMPL_HDL_003_model_detail_backend() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    // "realizar" is a default backend
    let response = app
        .oneshot(Request::get("/v1/models/realizar").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["object"], "model");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CMPL_HDL_004_openai_audio_compat() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"audio_data": "SGVsbG8=", "format": "wav"});
    let response = app
        .oneshot(
            Request::post("/v1/audio/transcriptions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CMPL_HDL_005_prompt_array_completion() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "prompt": ["Hello", "World"],
        "max_tokens": 20
    });
    let response = app
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}
