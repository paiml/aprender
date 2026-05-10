//! FALSIFY-AUTH-001 — every protected route returns HTTP 401 with a JSON
//! envelope (`{"error":"unauthorized", ...}`) when no `Authorization: Bearer
//! <key>` header is presented.
//!
//! Contract: `contracts/apr-serve-api-key-auth-v1.yaml`.
//!
//! Discharge strategy: build a minimal axum router that wraps any path
//! with the `apr_cli::serve_auth::layer` middleware, then drive it with
//! `tower::ServiceExt::oneshot` for each shipped route shape (GET / POST /
//! HEAD-not-allowed). The auth layer is generic over the inner router's
//! state, so this same code path is what `routes::create_router`,
//! `handlers::build_apr_cpu_router`, and `handlers_include_01::build_gpu_router`
//! all use in production.

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

#[cfg(feature = "inference")]
mod tests {
    use apr_cli::serve_auth::{layer, AuthGate};
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
        routing::{get, post},
        Json, Router,
    };
    use serde_json::Value;
    use tower::ServiceExt;

    fn protected_router() -> Router {
        let router = Router::new()
            .route("/", get(|| async { "ok" }))
            .route(
                "/health",
                get(|| async { Json(serde_json::json!({"ok": true})) }),
            )
            .route(
                "/predict",
                post(|| async { Json(serde_json::json!({"ok": true})) }),
            )
            .route(
                "/v1/chat/completions",
                post(|| async { Json(serde_json::json!({"ok": true})) }),
            );
        layer(AuthGate::from_plain_key("the-correct-key"), router)
    }

    async fn fire(method: &str, path: &str, header: Option<&str>) -> (StatusCode, Value) {
        let mut builder = Request::builder().method(method).uri(path);
        if let Some(value) = header {
            builder = builder.header(axum::http::header::AUTHORIZATION, value);
        }
        let req = builder.body(Body::empty()).unwrap();
        let resp = protected_router().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let body: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, body)
    }

    #[tokio::test]
    async fn missing_bearer_returns_401_on_every_route() {
        for (method, path) in [
            ("GET", "/"),
            ("GET", "/health"),
            ("POST", "/predict"),
            ("POST", "/v1/chat/completions"),
        ] {
            let (status, body) = fire(method, path, None).await;
            assert_eq!(
                status,
                StatusCode::UNAUTHORIZED,
                "{method} {path}: expected 401 with no Authorization header, got {status:?}",
            );
            assert_eq!(
                body["error"].as_str(),
                Some("unauthorized"),
                "{method} {path}: response body must carry error=unauthorized — got {body:?}",
            );
        }
    }

    #[tokio::test]
    async fn wrong_scheme_returns_401() {
        let (status, body) = fire("GET", "/health", Some("Basic dXNlcjpwYXNz")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"].as_str(), Some("unauthorized"));
    }

    #[tokio::test]
    async fn wrong_key_returns_401() {
        let (status, body) = fire("GET", "/health", Some("Bearer the-wrong-key")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"].as_str(), Some("unauthorized"));
    }

    #[tokio::test]
    async fn unauthenticated_response_carries_www_authenticate_header() {
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = protected_router().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let www_auth = resp
            .headers()
            .get(axum::http::header::WWW_AUTHENTICATE)
            .and_then(|v| v.to_str().ok());
        assert_eq!(
            www_auth,
            Some("Bearer"),
            "401 must advertise scheme via WWW-Authenticate per RFC 7235",
        );
    }
}

#[cfg(not(feature = "inference"))]
#[test]
fn auth_layer_only_compiled_with_inference_feature() {
    // The middleware lives behind `feature = "inference"` because it
    // depends on axum, which is itself feature-gated. The test crate
    // builds either way — this stub keeps the test target valid when
    // `cargo test --no-default-features` runs without inference.
}
