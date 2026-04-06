//! Tests for Banco privacy middleware.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware,
    routing::get,
    Router,
};
use tower::ServiceExt;

use super::middleware::privacy_layer;
use crate::serve::backends::PrivacyTier;

/// Minimal handler for middleware testing.
async fn ok_handler() -> &'static str {
    "ok"
}

/// Build a test router with the privacy middleware.
fn test_router(tier: PrivacyTier) -> Router {
    Router::new()
        .route("/test", get(ok_handler))
        .layer(middleware::from_fn(move |req, next| privacy_layer(tier, req, next)))
}

// ============================================================================
// BANCO_MID_001: Privacy header injection
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_001_standard_header() {
    let app = test_router(PrivacyTier::Standard);
    let response = app
        .oneshot(Request::builder().uri("/test").body(Body::empty()).expect("request"))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-privacy-tier").expect("header").to_str().expect("str"),
        "standard"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_001_sovereign_header() {
    let app = test_router(PrivacyTier::Sovereign);
    let response = app
        .oneshot(Request::builder().uri("/test").body(Body::empty()).expect("request"))
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-privacy-tier").expect("header").to_str().expect("str"),
        "sovereign"
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_001_private_header() {
    let app = test_router(PrivacyTier::Private);
    let response = app
        .oneshot(Request::builder().uri("/test").body(Body::empty()).expect("request"))
        .await
        .expect("response");

    assert_eq!(
        response.headers().get("x-privacy-tier").expect("header").to_str().expect("str"),
        "private"
    );
}

// ============================================================================
// BANCO_MID_002: Sovereign mode rejects external backend hints
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_002_sovereign_rejects_external_backend() {
    let app = test_router(PrivacyTier::Sovereign);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("x-banco-backend", "openai")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        response.headers().get("x-privacy-tier").expect("header").to_str().expect("str"),
        "sovereign"
    );
}

// ============================================================================
// BANCO_MID_003: Sovereign mode allows local backend hints
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_003_sovereign_allows_local_backend() {
    let app = test_router(PrivacyTier::Sovereign);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("x-banco-backend", "realizar")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// BANCO_MID_004: CORS headers
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_004_cors_headers_present() {
    let app = test_router(PrivacyTier::Standard);
    let response = app
        .oneshot(Request::builder().uri("/test").body(Body::empty()).expect("request"))
        .await
        .expect("response");

    assert!(response.headers().get("access-control-allow-origin").is_some());
    assert_eq!(
        response.headers().get("access-control-allow-origin").expect("cors").to_str().expect("str"),
        "*"
    );
    assert!(response.headers().get("access-control-allow-methods").is_some());
    assert!(response.headers().get("access-control-expose-headers").is_some());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_004_cors_preflight() {
    let app = test_router(PrivacyTier::Standard);
    let response = app
        .oneshot(
            Request::builder().method("OPTIONS").uri("/test").body(Body::empty()).expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        response.headers().get("access-control-allow-origin").expect("cors").to_str().expect("str"),
        "*"
    );
    assert!(response
        .headers()
        .get("access-control-max-age")
        .expect("max-age")
        .to_str()
        .expect("str")
        .contains("86400"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_MID_003_standard_allows_any_backend() {
    let app = test_router(PrivacyTier::Standard);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/test")
                .header("x-banco-backend", "openai")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
}
