//! #3718 vs PMAT-4616 (D5, #4614): the over-length refusal in `apr serve`'s OWN
//! SafeTensors handlers. D5 maps `aprender-serve`'s api errors to 4xx; it does not
//! touch these apr-cli handlers, so on `main` (with or without D5) this request is a
//! 500 "Generation failed". Driven through the REAL `safetensors_app` router.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::tests_st_serve_session_4269::tiny_transformer;
use super::{safetensors_app, SafeTensorsState};
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

fn router() -> axum::Router {
    let state = SafeTensorsState {
        transformer: Some(Arc::new(Mutex::new(tiny_transformer()))), // context_length 64
        tokenizer_info: None, // prompt chars become token ids, one per char
        model_path: "tiny.safetensors".into(),
    };
    safetensors_app(serde_json::json!({"count": 0}), true, state, "tiny".into())
}

async fn post(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| panic!("{status}: {}", String::from_utf8_lossy(&bytes)));
    (status, v)
}

/// 200 prompt tokens cannot fit a 64-token window: a client error, named as such.
fn assert_context_refusal(path: &str, status: StatusCode, v: &serde_json::Value) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {v}");
    assert_eq!(v["error"]["code"], "context_length_exceeded", "{path}: {v}");
}

#[tokio::test]
async fn st_router_generate_over_length_is_a_400_context_length_exceeded() {
    let body = serde_json::json!({"prompt": "a".repeat(200), "max_tokens": 4});
    let (st, v) = post("/generate", body).await;
    assert_context_refusal("/generate", st, &v);
}

#[tokio::test]
async fn st_router_chat_over_length_is_a_400_context_length_exceeded() {
    let body = serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "a".repeat(200)}],
        "max_tokens": 4,
    });
    let (st, v) = post("/v1/chat/completions", body).await;
    assert_context_refusal("/v1/chat/completions", st, &v);
}
