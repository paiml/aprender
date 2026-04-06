//! P1 cross-cutting tests: embeddings endpoint + audit logging.

// ============================================================================
// Embeddings Endpoint
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_embeddings_single_input() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"input": "Hello, world!"});
    let response = app
        .oneshot(
            Request::post("/api/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let embedding = data[0]["embedding"].as_array().expect("embedding array");
    assert_eq!(embedding.len(), 128, "heuristic embeddings should be 128-dim");
    assert!(json["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens") > 0);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_embeddings_batch_input() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"input": ["Hello", "World", "Foo"]});
    let response = app
        .oneshot(
            Request::post("/api/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 3, "batch should return 3 embeddings");
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    assert_eq!(data[2]["index"], 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_embeddings_deterministic() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    // Same input should produce same embedding
    let state = super::state::BancoStateInner::with_defaults();
    let body = serde_json::json!({"input": "deterministic test"});

    let app1 = super::router::create_banco_router(state.clone());
    let resp1 = app1
        .oneshot(
            Request::post("/api/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    let bytes1 = axum::body::to_bytes(resp1.into_body(), 1_048_576).await.expect("body");
    let json1: serde_json::Value = serde_json::from_slice(&bytes1).expect("parse");

    let app2 = super::router::create_banco_router(state);
    let resp2 = app2
        .oneshot(
            Request::post("/api/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    let bytes2 = axum::body::to_bytes(resp2.into_body(), 1_048_576).await.expect("body");
    let json2: serde_json::Value = serde_json::from_slice(&bytes2).expect("parse");

    assert_eq!(json1["data"][0]["embedding"], json2["data"][0]["embedding"]);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_embeddings_v1_compat_route() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"input": "test"});
    let response = app
        .oneshot(
            Request::post("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Audit Logging
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_audit_log_captures_requests() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let audit_log = super::audit::AuditLog::new();
    let app = super::router::create_banco_router_with_audit(
        super::state::BancoStateInner::with_defaults(),
        audit_log.clone(),
    );

    assert!(audit_log.is_empty());

    // Make a request
    let _ =
        app.oneshot(Request::get("/health").body(Body::empty()).expect("req")).await.expect("resp");

    assert_eq!(audit_log.len(), 1);
    let entries = audit_log.recent(10);
    assert_eq!(entries[0].method, "GET");
    assert_eq!(entries[0].path, "/health");
    assert_eq!(entries[0].status, 200);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_audit_log_multiple_requests() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let audit_log = super::audit::AuditLog::new();
    let state = super::state::BancoStateInner::with_defaults();

    // Need to create a new router for each oneshot (consumed by oneshot)
    let app = super::router::create_banco_router_with_audit(state.clone(), audit_log.clone());
    let _ = app.oneshot(Request::get("/health").body(Body::empty()).expect("req")).await;

    let app = super::router::create_banco_router_with_audit(state.clone(), audit_log.clone());
    let _ = app.oneshot(Request::get("/api/v1/system").body(Body::empty()).expect("req")).await;

    let app = super::router::create_banco_router_with_audit(state, audit_log.clone());
    let _ = app.oneshot(Request::get("/api/v1/models").body(Body::empty()).expect("req")).await;

    assert_eq!(audit_log.len(), 3);
    let recent = audit_log.recent(2);
    assert_eq!(recent.len(), 2);
    // Most recent first
    assert_eq!(recent[0].path, "/api/v1/models");
    assert_eq!(recent[1].path, "/api/v1/system");
}

#[test]
#[allow(non_snake_case)]
fn test_P1_audit_log_ring_buffer() {
    let log = super::audit::AuditLog::new();
    // Fill beyond capacity
    for i in 0..10_001 {
        log.push(super::audit::AuditEntry {
            ts: format!("{i}"),
            method: "GET".to_string(),
            path: format!("/test/{i}"),
            status: 200,
            latency_ms: 1,
        });
    }
    assert_eq!(log.len(), 10_000, "ring buffer should cap at 10,000");
    let recent = log.recent(1);
    assert_eq!(recent[0].path, "/test/10000", "most recent should be last pushed");
}

// ============================================================================
// EmbeddingsInput type tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P1_embeddings_input_single() {
    let input: super::types::EmbeddingsInput =
        serde_json::from_str(r#""hello""#).expect("parse single");
    assert_eq!(input.texts(), vec!["hello"]);
}

#[test]
#[allow(non_snake_case)]
fn test_P1_embeddings_input_batch() {
    let input: super::types::EmbeddingsInput =
        serde_json::from_str(r#"["a","b","c"]"#).expect("parse batch");
    assert_eq!(input.texts(), vec!["a", "b", "c"]);
}
