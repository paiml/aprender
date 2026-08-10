//! Falsifiers for the `RouterConfig` hardening flags behind
//! `apr serve run --no-metrics` / `--no-cors`.
//!
//! These assert observable HTTP behaviour (status code, response headers), not
//! the presence of a config field: an operator who passes `--no-metrics` must
//! get a 404 from `/metrics`, and one who passes `--no-cors` must see no
//! `access-control-*` header on the wire.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::*;

fn router_with(cors: bool, metrics: bool) -> axum::Router {
    let state = AppState::with_cache(10);
    create_router_with_config(
        state,
        RouterConfig {
            openai_api: true,
            cors,
            metrics,
        },
    )
}

async fn get(app: axum::Router, uri: &str) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri(uri)
            .header("origin", "http://example.invalid")
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("router responds")
}

async fn body_string(response: axum::http::Response<Body>) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Baseline: with metrics enabled the endpoint really does serve Prometheus
/// text. Without this, a "fix" that deleted the route would look correct.
#[tokio::test]
async fn metrics_enabled_serves_prometheus_exposition() {
    let response = get(router_with(true, true), "/metrics").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_string(response).await;
    assert!(
        body.contains("realizar_requests_total"),
        "enabled /metrics must expose the counter set, got: {body}"
    );
}

/// `--no-metrics` must withhold telemetry, not merely hide the banner line.
#[tokio::test]
async fn metrics_disabled_makes_metrics_endpoints_404() {
    for uri in ["/metrics", "/metrics/dispatch"] {
        let response = get(router_with(true, false), uri).await;
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "{uri} must 404 when metrics are disabled"
        );

        let body = body_string(response).await;
        assert!(
            !body.contains("realizar_requests_total"),
            "{uri} leaked metrics while disabled: {body}"
        );
    }
}

/// Disabling metrics must not take the rest of the server down with it.
#[tokio::test]
async fn metrics_disabled_keeps_health_serving() {
    let response = get(router_with(true, false), "/health").await;
    assert_eq!(response.status(), StatusCode::OK);
}

/// Baseline: CORS enabled really does advertise a permissive origin.
#[tokio::test]
async fn cors_enabled_sends_allow_origin_header() {
    let response = get(router_with(true, true), "/health").await;
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .map(|v| v.to_str().unwrap_or_default().to_owned()),
        Some("*".to_owned()),
        "CORS-enabled server must advertise a permissive origin"
    );
}

/// `--no-cors` must strip every `access-control-*` header from the response.
#[tokio::test]
async fn cors_disabled_sends_no_access_control_headers() {
    let response = get(router_with(false, true), "/health").await;
    assert_eq!(response.status(), StatusCode::OK);

    let leaked: Vec<String> = response
        .headers()
        .keys()
        .map(|k| k.as_str().to_owned())
        .filter(|k| k.starts_with("access-control"))
        .collect();
    assert!(
        leaked.is_empty(),
        "CORS-disabled server still sent: {leaked:?}"
    );
}

/// A browser preflight against a CORS-disabled server must not be answered
/// with permissive method/header allowances either.
#[tokio::test]
async fn cors_disabled_does_not_answer_preflight_permissively() {
    let response = router_with(false, true)
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/health")
                .header("origin", "http://example.invalid")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router responds");

    let leaked: Vec<String> = response
        .headers()
        .keys()
        .map(|k| k.as_str().to_owned())
        .filter(|k| k.starts_with("access-control"))
        .collect();
    assert!(
        leaked.is_empty(),
        "CORS-disabled preflight still sent: {leaked:?}"
    );
}
