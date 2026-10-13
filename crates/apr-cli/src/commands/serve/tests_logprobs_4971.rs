//! #4971 (ASOC-INV-021): the `apr serve` chat routes realizar's router does not
//! mount compute no logprobs, so a request for them is refused (501 naming the
//! backend, 400/422 for a bad field) and never answered 200 without them.
//! Driven through the REAL router builders, with no model loaded.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

async fn chat(router: &axum::Router, body: serde_json::Value) -> (StatusCode, String) {
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// Every refusal `router` must answer, naming `backend` on the 501; and a
/// request that does not ask is not refused for logprobs.
async fn refuses_logprobs(router: &axum::Router, backend: &str) {
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
    // (logprobs, top_logprobs, status, words the reply must carry)
    let table: &[(serde_json::Value, serde_json::Value, StatusCode, &str)] = &[
        (
            true.into(),
            serde_json::Value::Null,
            StatusCode::NOT_IMPLEMENTED,
            backend,
        ),
        (true.into(), 3.into(), StatusCode::NOT_IMPLEMENTED, backend),
        (
            serde_json::Value::Null,
            3.into(),
            StatusCode::BAD_REQUEST,
            "requires logprobs",
        ),
        (
            true.into(),
            21.into(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "top_logprobs must be",
        ),
        (
            "yes".into(),
            serde_json::Value::Null,
            StatusCode::UNPROCESSABLE_ENTITY,
            "logprobs must be",
        ),
    ];
    for (logprobs, top, want, words) in table {
        let body = serde_json::json!({
            "messages": messages, "max_tokens": 2, "logprobs": logprobs, "top_logprobs": top
        });
        let (status, reply) = chat(router, body.clone()).await;
        assert_eq!(status, *want, "{backend} {body}: {reply}");
        assert!(reply.contains(words), "{backend} {body}: {reply}");
    }
    let plain = serde_json::json!({"messages": messages, "max_tokens": 2, "logprobs": false});
    let (status, reply) = chat(router, plain).await;
    assert_ne!(status, StatusCode::NOT_IMPLEMENTED, "{backend}: {reply}");
    assert!(
        !reply.contains("`logprobs` is not supported"),
        "{backend}: {reply}"
    );
}

#[tokio::test]
async fn the_apr_cpu_chat_route_refuses_logprobs() {
    let r = super::handlers::build_demo_apr_cpu_router_for_test();
    refuses_logprobs(&r, "APR CPU (AprTransformer)").await;
}

#[tokio::test]
async fn the_safetensors_f32_chat_route_refuses_logprobs() {
    let state = super::safetensors::SafeTensorsState {
        transformer: None,
        tokenizer_info: None,
        model_path: "m.safetensors".into(),
    };
    let r = super::safetensors::safetensors_app(
        serde_json::json!({"count": 0}),
        true,
        state,
        "m".into(),
    );
    refuses_logprobs(&r, "SafeTensors F32 CPU").await;
}
