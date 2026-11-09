//! #3568 PR 4: `response_format` on apr-cli's own chat servers is refused by name, never
//! dropped. Driven through the REAL router builders, so a handler that stops calling
//! `refuse_response_format` serves the request and fails here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

async fn chat(router: &axum::Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

fn asked(format: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": "Give me a JSON object."}],
        "max_tokens": 4,
        "response_format": format,
    })
}

/// A constrained request answers 400 naming `path`; the same request asking for `text`
/// reaches the loop (which, with no model loaded, answers something else).
async fn refused_by_name_and_text_is_served(router: axum::Router, path: &str) {
    let schema = serde_json::json!({"type": "object", "required": ["a"]});
    for format in [
        serde_json::json!({"type": "json_object"}),
        serde_json::json!({"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}),
    ] {
        let (status, body) = chat(&router, asked(format.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{format}: {body}");
        assert_eq!(body["type"], "schema_unsupported_path", "{format}: {body}");
        let error = body["error"].as_str().unwrap_or_default();
        assert!(
            error.contains(&format!("the {path} generation path")),
            "{error}"
        );
    }
    let (status, body) = chat(&router, asked(serde_json::json!({"type": "text"}))).await;
    assert_ne!(body["type"], "schema_unsupported_path", "{status}: {body}");
    assert_ne!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn the_apr_cpu_server_refuses_a_constraint_by_name() {
    let router = super::handlers::build_demo_apr_cpu_router_for_test();
    refused_by_name_and_text_is_served(router, super::response_format::APR_CPU).await;
}

#[tokio::test]
async fn the_safetensors_server_refuses_a_constraint_by_name() {
    let state = super::safetensors::SafeTensorsState {
        transformer: None,
        tokenizer_info: None,
        model_path: "m.safetensors".into(),
    };
    let router = super::safetensors::safetensors_app(
        serde_json::json!({"count": 0}),
        true,
        state,
        "m".into(),
    );
    refused_by_name_and_text_is_served(router, super::response_format::SAFETENSORS_CPU).await;
}
