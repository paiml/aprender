//! Metrics collector + endpoint tests.

use super::metrics::MetricsCollector;

// ============================================================================
// MetricsCollector unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_METRICS_001_initial_state() {
    let m = MetricsCollector::new();
    assert_eq!(m.total_requests.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(m.chat_requests.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(m.tokens_generated.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(m.errors.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
#[allow(non_snake_case)]
fn test_METRICS_002_increment() {
    let m = MetricsCollector::new();
    m.inc_requests();
    m.inc_requests();
    m.inc_chat();
    m.add_tokens(100);
    m.inc_errors();
    assert_eq!(m.total_requests.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(m.chat_requests.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(m.tokens_generated.load(std::sync::atomic::Ordering::Relaxed), 100);
    assert_eq!(m.errors.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
#[allow(non_snake_case)]
fn test_METRICS_003_render_prometheus() {
    let m = MetricsCollector::new();
    m.inc_requests();
    m.inc_chat();
    m.add_tokens(42);

    let output = m.render(true, 85);
    assert!(output.contains("banco_requests_total 1"));
    assert!(output.contains("banco_chat_requests_total 1"));
    assert!(output.contains("banco_tokens_generated_total 42"));
    assert!(output.contains("banco_model_loaded 1"));
    assert!(output.contains("banco_endpoints_total 85"));
    assert!(output.contains("# TYPE banco_requests_total counter"));
}

#[test]
#[allow(non_snake_case)]
fn test_METRICS_004_render_no_model() {
    let m = MetricsCollector::new();
    let output = m.render(false, 85);
    assert!(output.contains("banco_model_loaded 0"));
}

// ============================================================================
// Metrics endpoint test
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_METRICS_HDL_001_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/metrics").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let ct = response.headers().get("content-type").expect("ct").to_str().expect("str");
    assert!(ct.contains("text/plain"), "expected text/plain, got: {ct}");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("banco_requests_total"));
    assert!(body.contains("banco_uptime_seconds"));
}
