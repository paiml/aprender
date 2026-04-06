//! Batch inference tests.

use crate::serve::templates::ChatMessage;

use super::batch::{BatchItem, BatchItemResult, BatchStatus};

// ============================================================================
// BatchStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BATCH_001_run_batch() {
    let store = super::batch::BatchStore::new();
    let items = vec![
        BatchItem {
            id: "req-1".to_string(),
            messages: vec![ChatMessage::user("Hello")],
            max_tokens: 10,
        },
        BatchItem {
            id: "req-2".to_string(),
            messages: vec![ChatMessage::user("World")],
            max_tokens: 10,
        },
    ];
    let job = store.run(items, |item| BatchItemResult {
        id: item.id.clone(),
        content: format!("echo: {}", item.messages[0].content),
        finish_reason: "test".to_string(),
        tokens: 1,
    });
    assert!(job.batch_id.starts_with("batch-"));
    assert_eq!(job.status, BatchStatus::Complete);
    assert_eq!(job.total_items, 2);
    assert_eq!(job.results.len(), 2);
    assert_eq!(job.results[0].content, "echo: Hello");
}

#[test]
#[allow(non_snake_case)]
fn test_BATCH_002_get_job() {
    let store = super::batch::BatchStore::new();
    let job = store.run(vec![], |_| unreachable!());
    let retrieved = store.get(&job.batch_id).expect("should exist");
    assert_eq!(retrieved.batch_id, job.batch_id);
}

#[test]
#[allow(non_snake_case)]
fn test_BATCH_003_list_jobs() {
    let store = super::batch::BatchStore::new();
    store.run(vec![], |_| unreachable!());
    store.run(vec![], |_| unreachable!());
    assert_eq!(store.list().len(), 2);
}

// ============================================================================
// Batch endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BATCH_HDL_001_submit() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "items": [
            {"id": "q1", "messages": [{"role": "user", "content": "What is Rust?"}]},
            {"id": "q2", "messages": [{"role": "user", "content": "What is Python?"}]}
        ]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["status"], "complete");
    assert_eq!(json["total_items"], 2);
    assert_eq!(json["results"].as_array().expect("results").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BATCH_HDL_002_list() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/batch").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}
