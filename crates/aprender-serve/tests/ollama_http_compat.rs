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

/// PMAT-SERVE-MULTITURN-001: a 3-turn /api/chat conversation must reach the
/// model in full — not just its last message.
///
/// The two tests above are ROUTING/SHAPE falsifiers: one request, one message,
/// and assertions on `done`/`message.role`/the presence of `message.content`.
/// They cannot see a conversation bug, and they are weaker than they look:
/// `chat_response_to_parts` folds an upstream error INTO `message.content`, so
/// the handler still answers 200 with `done:true` when generation fails
/// outright. Every assertion in those tests is satisfied by total failure.
///
/// So this one does not assert on shape. It uses `prompt_eval_count` — the
/// count of PROMPT tokens the backend actually consumed — as an oracle:
/// a 3-turn conversation must consume strictly more prompt tokens than its
/// last turn alone. That is a property of the whole pipeline (handler →
/// `to_chat_request` → template → tokenizer), and it is exactly what breaks if
/// the handler ever drops all-but-last, which is the RED-turning mutation the
/// roadmap names.
#[tokio::test]
async fn api_chat_three_turn_conversation_reaches_the_model_in_full() {
    let last_turn_only = serde_json::json!({
        "model": "apr",
        "messages": [{"role": "user", "content": "And what is the third?"}],
        "stream": false
    });
    let three_turns = serde_json::json!({
        "model": "apr",
        "messages": [
            {"role": "user",      "content": "What is the first letter of the alphabet?"},
            {"role": "assistant", "content": "The first letter of the alphabet is A."},
            {"role": "user",      "content": "And what is the third?"}
        ],
        "stream": false
    });

    async fn prompt_tokens(body: serde_json::Value) -> (u64, serde_json::Value) {
        let app = create_test_app();
        let resp = app.oneshot(json_post("/api/chat", body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("Ollama JSON body");
        let n = json["prompt_eval_count"]
            .as_u64()
            .unwrap_or_else(|| panic!("prompt_eval_count must be present and numeric: {json}"));
        (n, json)
    }

    let (single, single_json) = prompt_tokens(last_turn_only).await;
    let (multi, multi_json) = prompt_tokens(three_turns).await;

    // Guard the oracle itself: if the backend reports 0 prompt tokens for both,
    // the comparison below is vacuous and would pass no matter what the handler
    // did with the message list.
    assert!(
        multi > 0,
        "prompt_eval_count is 0 for a 3-turn conversation — the oracle is not \
         measuring anything, so this test cannot detect a dropped turn.\n{multi_json}"
    );

    assert!(
        multi > single,
        "a 3-turn conversation consumed {multi} prompt tokens but its LAST TURN ALONE \
         consumed {single} — the earlier turns never reached the model.\n\
         3-turn: {multi_json}\nlast-only: {single_json}"
    );
}
