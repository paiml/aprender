//! Model registry (pacha) endpoint tests.

use super::handlers_registry::{CachedModelInfo, PullRequest, PullResult};

// ============================================================================
// Type unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_REG_001_pull_request_serde() {
    let json = serde_json::json!({"model_ref": "llama3:8b-q4"});
    let req: PullRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.model_ref, "llama3:8b-q4");
}

#[test]
#[allow(non_snake_case)]
fn test_REG_002_pull_result_serializes() {
    let result = PullResult {
        model_ref: "test".to_string(),
        status: "cached".to_string(),
        path: Some("/path/to/model".to_string()),
        size_bytes: Some(1024),
        cache_hit: true,
        format: Some("gguf".to_string()),
    };
    let json = serde_json::to_value(&result).expect("serialize");
    assert_eq!(json["status"], "cached");
    assert!(json["cache_hit"].as_bool().expect("bool"));
}

#[test]
#[allow(non_snake_case)]
fn test_REG_003_cached_model_info() {
    let info = CachedModelInfo {
        name: "llama3".to_string(),
        version: "1.0.0".to_string(),
        path: "/cache/llama3.gguf".to_string(),
        size_bytes: 4_000_000_000,
        format: "gguf".to_string(),
    };
    let json = serde_json::to_value(&info).expect("serialize");
    assert_eq!(json["name"], "llama3");
    assert_eq!(json["size_bytes"], 4_000_000_000_u64);
}

// ============================================================================
// Registry endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_REG_HDL_001_pull_model() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"model_ref": "test-model:latest"});
    let response = app
        .oneshot(
            Request::post("/api/v1/models/pull")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    // May return OK (dry-run/cache) or NOT_FOUND (not in registry)
    let status = response.status().as_u16();
    assert!(status == 200 || status == 404, "expected 200 or 404, got {status}");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_REG_HDL_002_list_registry() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/models/registry").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    // models array should exist (may be empty)
    assert!(json["models"].is_array());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_REG_HDL_003_remove_nonexistent() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::delete("/api/v1/models/registry/nonexistent-model")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    // Either 204 (dry-run) or 404 (not found in real registry)
    let status = response.status().as_u16();
    assert!(status == 204 || status == 404, "expected 204 or 404, got {status}");
}
