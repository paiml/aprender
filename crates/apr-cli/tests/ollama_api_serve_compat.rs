//! PMAT-923 — `apr serve` answers Ollama's native `/api/chat` + `/api/generate`.
//!
//! The PROVEN GAP this falsifier closes: `apr serve <model>` does NOT mount
//! realizar's `create_router`; it builds its OWN bespoke axum routers (here the
//! APR-CPU router from `handlers::build_apr_cpu_router`). Wiring the Ollama
//! routes only into realizar's router left a live Ollama client getting a 404
//! from `apr serve`.
//!
//! This test exercises the REAL apr-cli serve router (via the
//! `apr_cli::serve_test_support::build_demo_apr_cpu_router_for_test` seam — the
//! exact router `apr serve <model.apr>` mounts), POSTs Ollama requests, and
//! asserts:
//!   * status is NOT 404 (the route is wired, not the axum fallback), and
//!   * the body has Ollama wire shape (`done:true`; `/api/chat` carries a nested
//!     `message:{role,content}`; `/api/generate` carries a flat `response`).
//!
//! RED on the unwired router (404, no `done`); GREEN once the routes are wired.
//! Mutation-verified: deleting the two `.route("/api/...")` calls in
//! `build_apr_cpu_router` flips both assertions RED (404).
//!
//! Contract: OBLIG-OLLAMA-API-ROUTED-ON-APR-SERVE in
//! `contracts/apr-serve-openai-compat-v1.yaml`.

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

#[cfg(feature = "inference")]
mod tests {
    use apr_cli::serve_test_support::build_demo_apr_cpu_router_for_test;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use serde_json::Value;
    use tower::ServiceExt;

    /// POST `body` to `path` on the REAL apr-cli APR-CPU serve router.
    async fn post_json(path: &str, body: Value) -> (StatusCode, Value) {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = build_demo_apr_cpu_router_for_test()
            .oneshot(req)
            .await
            .unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, json)
    }

    #[tokio::test]
    async fn api_chat_is_routed_and_returns_ollama_shape() {
        let (status, body) = post_json(
            "/api/chat",
            serde_json::json!({
                "model": "apr",
                "messages": [{"role": "user", "content": "2+2?"}],
                "stream": false
            }),
        )
        .await;

        // The PROVEN GAP: an unwired route falls through to the axum 404 handler.
        assert_ne!(
            status,
            StatusCode::NOT_FOUND,
            "/api/chat must be wired into the apr serve router, not hit the 404 fallback. body={body}"
        );
        assert_eq!(status, StatusCode::OK, "/api/chat returns 200. body={body}");

        // Ollama chat wire shape: nested message + terminal done flag.
        assert_eq!(
            body["done"], true,
            "Ollama body must carry done:true (the 404 fallback has no `done`). body={body}"
        );
        assert_eq!(
            body["message"]["role"], "assistant",
            "Ollama chat carries message.role=assistant. body={body}"
        );
        assert!(
            body["message"]["content"].is_string(),
            "Ollama chat carries message.content string. body={body}"
        );
        // Generate's flat field must NOT be present on a chat body.
        assert!(
            body.get("response").is_none(),
            "/api/chat uses nested message, not a flat response. body={body}"
        );
    }

    #[tokio::test]
    async fn api_generate_is_routed_and_returns_ollama_shape() {
        let (status, body) = post_json(
            "/api/generate",
            serde_json::json!({
                "model": "apr",
                "prompt": "2+2?",
                "stream": false
            }),
        )
        .await;

        assert_ne!(
            status,
            StatusCode::NOT_FOUND,
            "/api/generate must be wired into the apr serve router, not hit the 404 fallback. body={body}"
        );
        assert_eq!(
            status,
            StatusCode::OK,
            "/api/generate returns 200. body={body}"
        );

        // Ollama generate wire shape: flat response + terminal done flag.
        assert_eq!(
            body["done"], true,
            "Ollama body must carry done:true. body={body}"
        );
        assert!(
            body["response"].is_string(),
            "/api/generate carries a flat response string. body={body}"
        );
        // Chat's nested message must NOT be present on a generate body.
        assert!(
            body.get("message").is_none(),
            "/api/generate uses a flat response, not a nested message. body={body}"
        );
    }

    #[tokio::test]
    async fn unknown_api_route_still_404s() {
        // Guard rail: the fallback IS still a 404 — proving the two assertions
        // above distinguish "wired" from "fallback", not "everything is 200".
        let (status, _body) = post_json("/api/does-not-exist", serde_json::json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
