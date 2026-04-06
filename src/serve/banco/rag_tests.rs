//! RAG pipeline tests.

// ============================================================================
// RagIndex unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_RAG_001_index_and_search() {
    let index = super::rag::RagIndex::new();
    index.index_document(
        "f1",
        "policy.txt",
        "Refunds are available within 30 days of purchase. Contact support for assistance.",
    );
    // Search uses exact token matching (no stemming), so use "refunds" not "refund"
    let results = index.search("refunds available purchase", 5, 0.0);
    assert!(!results.is_empty(), "should find matching chunk");
    assert!(results[0].score > 0.0);
    assert_eq!(results[0].file, "policy.txt");
}

#[test]
#[allow(non_snake_case)]
fn test_RAG_002_empty_index_returns_empty() {
    let index = super::rag::RagIndex::new();
    let results = index.search("anything", 5, 0.0);
    assert!(results.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_RAG_003_status() {
    let index = super::rag::RagIndex::new();
    assert_eq!(index.status().doc_count, 0);
    assert!(!index.status().indexed);

    index.index_document("f1", "a.txt", "hello world");
    let status = index.status();
    assert_eq!(status.doc_count, 1);
    assert!(status.chunk_count >= 1);
    assert!(status.indexed);
}

#[test]
#[allow(non_snake_case)]
fn test_RAG_004_clear() {
    let index = super::rag::RagIndex::new();
    index.index_document("f1", "a.txt", "hello world");
    assert!(index.status().indexed);
    index.clear();
    assert!(!index.status().indexed);
    assert_eq!(index.status().chunk_count, 0);
}

#[test]
#[allow(non_snake_case)]
fn test_RAG_005_min_score_filters() {
    let index = super::rag::RagIndex::new();
    index.index_document("f1", "a.txt", "machine learning deep learning neural networks");
    let all = index.search("learning", 10, 0.0);
    let filtered = index.search("learning", 10, 999.0);
    assert!(!all.is_empty());
    assert!(filtered.is_empty(), "high min_score should filter everything");
}

#[test]
#[allow(non_snake_case)]
fn test_RAG_006_multiple_documents() {
    let index = super::rag::RagIndex::new();
    index.index_document("f1", "ml.txt", "Machine learning uses data to train models");
    index.index_document("f2", "cooking.txt", "Add salt and pepper to taste");
    let results = index.search("machine learning models", 5, 0.0);
    assert!(!results.is_empty());
    assert_eq!(results[0].file, "ml.txt", "ML doc should rank higher for ML query");
}

#[test]
#[allow(non_snake_case)]
fn test_RAG_007_is_indexed() {
    let index = super::rag::RagIndex::new();
    assert!(!index.is_indexed("f1"));
    index.index_document("f1", "a.txt", "hello");
    assert!(index.is_indexed("f1"));
    assert!(!index.is_indexed("f2"));
}

#[test]
#[allow(non_snake_case)]
fn test_RAG_008_chunking_long_document() {
    let index = super::rag::RagIndex::new();
    // Create a document longer than 512 tokens (~2048 chars)
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    index.index_document("f1", "long.txt", &long_text);
    assert!(index.status().chunk_count > 1, "long doc should be chunked");
}

// ============================================================================
// RAG endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RAG_HDL_001_status_empty() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/rag/status").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["indexed"], false);
    assert_eq!(json["chunk_count"], 0);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RAG_HDL_002_index_and_status() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    // Upload a file first
    state.files.store(
        "doc.txt",
        b"Rust is a systems programming language focused on safety and performance.",
    );

    // Index
    let app = super::router::create_banco_router(state.clone());
    let response = app
        .oneshot(Request::post("/api/v1/rag/index").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["indexed_files"], 1);
    assert!(json["status"]["indexed"].as_bool().expect("indexed"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RAG_HDL_003_chat_with_rag() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    // Upload and index a doc
    state.files.store(
        "rust.txt",
        b"Rust provides memory safety without garbage collection through ownership.",
    );
    // Manually index (since auto-index isn't wired yet)
    state.rag.index_document(
        "manual",
        "rust.txt",
        "Rust provides memory safety without garbage collection through ownership.",
    );

    // Chat with rag: true
    let app = super::router::create_banco_router(state);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "How does Rust handle memory?"}],
        "rag": true
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    // In dry-run mode, prompt_tokens should be larger because RAG context was prepended
    let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().expect("tokens");
    assert!(prompt_tokens > 10, "RAG context should increase prompt size");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RAG_HDL_004_clear_index() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.rag.index_document("f1", "a.txt", "hello world");
    assert!(state.rag.status().indexed);

    let app = super::router::create_banco_router(state.clone());
    let response = app
        .oneshot(Request::delete("/api/v1/rag/index").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NO_CONTENT);
    assert!(!state.rag.status().indexed);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RAG_HDL_005_auto_index_on_upload() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    assert!(!state.rag.status().indexed);

    // Upload a file via JSON endpoint
    let app = super::router::create_banco_router(state.clone());
    let body =
        serde_json::json!({"name": "auto.txt", "content": "Rust ownership borrowing lifetimes"});
    let _response = app
        .oneshot(
            Request::post("/api/v1/data/upload/json")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");

    // RAG index should now have the document
    assert!(state.rag.status().indexed, "upload should auto-index for RAG");
    assert_eq!(state.rag.status().doc_count, 1);

    // Search should find it
    let results = state.rag.search("ownership borrowing", 5, 0.0);
    assert!(!results.is_empty(), "auto-indexed doc should be searchable");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RAG_HDL_006_search_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.rag.index_document("f1", "ml.txt", "machine learning neural networks deep learning");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::get("/api/v1/rag/search?q=neural+networks&top_k=3")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(!json["results"].as_array().expect("results").is_empty());
    assert_eq!(json["query"], "neural networks");
}
