//! Browser UI serving tests.

// ============================================================================
// UI endpoint tests (zero-JS SSR)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_UI_HDL_001_index_returns_html() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response =
        app.oneshot(Request::get("/").body(Body::empty()).expect("req")).await.expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let ct = response.headers().get("content-type").expect("ct").to_str().expect("str");
    assert!(ct.contains("text/html"), "expected HTML content-type, got: {ct}");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let html = String::from_utf8_lossy(&bytes);
    assert!(html.contains("Banco"));
    // Zero-JS UI uses <form> instead of <script>
    assert!(html.contains("<form"), "Should have a form element");
    assert!(!html.contains("<script>"), "Zero-JS: no inline script tags");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_UI_HDL_002_chat_form_post() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::post("/ui/chat")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("message=Hello+Banco"))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let html = String::from_utf8_lossy(&bytes);
    assert!(html.contains("Hello Banco"), "User message should appear");
    assert!(!html.contains("<script>"), "Zero-JS: response has no script tags");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_UI_HDL_003_assets_404() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/assets/nonexistent.wasm").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_UI_HDL_004_index_has_model_status() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response =
        app.oneshot(Request::get("/").body(Body::empty()).expect("req")).await.expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let html = String::from_utf8_lossy(&bytes);
    // Should show model status (no model loaded)
    assert!(html.contains("No model"), "Should indicate no model loaded");
}
