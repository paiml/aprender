//! PMAT-923 falsifier: Ollama HTTP compat routes (`/api/chat`, `/api/generate`).
//!
//! GAP: `apr serve` exposed only the OpenAI `/v1/*` API, so an Ollama client
//! could not use it as a drop-in HTTP replacement — POSTing `/api/chat` or
//! `/api/generate` returned the axum 404 `not_found` fallback.
//!
//! These tests are the falsifier for OBLIG-OLLAMA-API-CHAT-GENERATE-ROUTED in
//! `contracts/apr-serve-openai-compat-v1.yaml`:
//!   - RED on the unwired router: the route is absent → 404 / no Ollama fields.
//!   - GREEN once the handlers are wired: 200 + an Ollama-shaped body
//!     (`done` present; `message.role`/`message.content` for chat; flat
//!     `response` for generate).
//!
//! The router is built from `AppState::demo()` (a tiny in-memory model), so no
//! real model download is needed.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use realizar::api::{create_router, AppState};
use tower::ServiceExt;

fn create_test_app() -> axum::Router {
    let state = AppState::demo().expect("demo state should create");
    create_router(state)
}

fn json_post(uri: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

/// FALSIFIER: POST /api/chat must be ROUTED and return an Ollama-shaped body.
///
/// RED (unwired): axum's fallback returns 404 with `{"error":"not_found"}`,
/// which has no `done`/`message` fields → the assertions below fail.
/// GREEN (wired): the Ollama handler returns 200 + the Ollama chat schema.
#[tokio::test]
async fn api_chat_is_routed_and_returns_ollama_shape() {
    let app = create_test_app();
    let req = json_post(
        "/api/chat",
        serde_json::json!({
            "model": "apr",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }),
    );

    let response = app.oneshot(req).await.unwrap();

    // The route MUST exist — anything other than the axum 404 fallback proves
    // it is wired. With the Ollama handler in place this is a 200.
    assert_ne!(
        response.status(),
        StatusCode::NOT_FOUND,
        "POST /api/chat returned 404 — route is not wired"
    );
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("Ollama JSON body");

    // Ollama chat schema: {model, message:{role,content}, done, ...}
    assert_eq!(json["done"], true, "Ollama chat body must carry done:true");
    assert_eq!(
        json["message"]["role"], "assistant",
        "Ollama chat body must carry message.role"
    );
    assert!(
        json["message"].get("content").is_some(),
        "Ollama chat body must carry message.content, got: {json}"
    );
    assert_eq!(json["model"], "apr");
}

/// FALSIFIER: POST /api/generate must be ROUTED and return Ollama's flat
/// `{response, done}` shape (no nested `message`).
#[tokio::test]
async fn api_generate_is_routed_and_returns_ollama_shape() {
    let app = create_test_app();
    let req = json_post(
        "/api/generate",
        serde_json::json!({
            "model": "apr",
            "prompt": "2+2=",
            "stream": false
        }),
    );

    let response = app.oneshot(req).await.unwrap();

    assert_ne!(
        response.status(),
        StatusCode::NOT_FOUND,
        "POST /api/generate returned 404 — route is not wired"
    );
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("Ollama JSON body");

    assert_eq!(
        json["done"], true,
        "Ollama generate body must carry done:true"
    );
    assert!(
        json.get("response").is_some(),
        "Ollama generate body must carry a flat `response` field, got: {json}"
    );
    assert!(
        json.get("message").is_none(),
        "Ollama generate body uses a flat `response`, not a nested message"
    );
    assert_eq!(json["model"], "apr");
}
