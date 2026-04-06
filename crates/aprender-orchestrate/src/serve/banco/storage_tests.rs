//! Storage and data endpoint tests.

// ============================================================================
// FileStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_STOR_001_store_and_get() {
    let store = super::storage::FileStore::in_memory();
    let info = store.store("test.txt", b"Hello, world!");
    assert!(info.id.starts_with("file-"));
    assert_eq!(info.name, "test.txt");
    assert_eq!(info.size_bytes, 13);
    assert_eq!(info.content_type, "text/plain");
    assert!(!info.content_hash.is_empty());

    let retrieved = store.get(&info.id).expect("should exist");
    assert_eq!(retrieved.name, "test.txt");
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_002_list_files() {
    let store = super::storage::FileStore::in_memory();
    store.store("a.csv", b"a,b,c");
    store.store("b.json", b"{}");
    store.store("c.pdf", b"%PDF");
    let files = store.list();
    assert_eq!(files.len(), 3);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_003_delete() {
    let store = super::storage::FileStore::in_memory();
    let info = store.store("test.txt", b"data");
    assert_eq!(store.len(), 1);
    store.delete(&info.id).expect("delete");
    assert_eq!(store.len(), 0);
    assert!(store.get(&info.id).is_none());
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_004_delete_not_found() {
    let store = super::storage::FileStore::in_memory();
    let result = store.delete("nonexistent");
    assert!(result.is_err());
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_005_content_type_detection() {
    let store = super::storage::FileStore::in_memory();
    assert_eq!(store.store("a.pdf", b"x").content_type, "application/pdf");
    assert_eq!(store.store("b.csv", b"x").content_type, "text/csv");
    assert_eq!(store.store("c.json", b"x").content_type, "application/json");
    assert_eq!(store.store("d.jsonl", b"x").content_type, "application/jsonl");
    assert_eq!(store.store("e.txt", b"x").content_type, "text/plain");
    assert_eq!(store.store("f.bin", b"x").content_type, "application/octet-stream");
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_006_content_hash_deterministic() {
    let store = super::storage::FileStore::in_memory();
    let a = store.store("a.txt", b"same content");
    let b = store.store("b.txt", b"same content");
    assert_eq!(a.content_hash, b.content_hash);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_007d_disk_roundtrip_reload() {
    let dir = std::path::PathBuf::from(format!("/tmp/banco_reload_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Write data to disk
    {
        let store = super::storage::FileStore::with_data_dir(dir.clone());
        store.store("alpha.txt", b"first file");
        store.store("beta.csv", b"a,b,c");
        assert_eq!(store.len(), 2);
    }

    // Create new store from same directory — should load existing files
    {
        let store = super::storage::FileStore::with_data_dir(dir.clone());
        assert_eq!(store.len(), 2, "should reload 2 files from disk");
        let files = store.list();
        let names: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"alpha.txt"));
        assert!(names.contains(&"beta.csv"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_007e_conversation_disk_roundtrip() {
    let dir = std::path::PathBuf::from(format!("/tmp/banco_conv_reload_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Write conversations to disk
    {
        let store = super::conversations::ConversationStore::with_data_dir(dir.clone());
        let id = store.create("test-model");
        store
            .append(&id, crate::serve::templates::ChatMessage::user("Hello from disk"))
            .expect("append");
        assert_eq!(store.len(), 1);
    }

    // Create new store from same directory — should load existing conversations
    {
        let store = super::conversations::ConversationStore::with_data_dir(dir.clone());
        assert_eq!(store.len(), 1, "should reload 1 conversation from disk");
        let list = store.list();
        let conv = store.get(&list[0].id).expect("get");
        assert_eq!(conv.messages.len(), 1);
        assert_eq!(conv.messages[0].content, "Hello from disk");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_007b_disk_persistence() {
    let dir = std::path::PathBuf::from(format!("/tmp/banco_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Store to disk
    let store = super::storage::FileStore::with_data_dir(dir.clone());
    let info = store.store("persist.txt", b"persistent data");

    // Verify file exists on disk
    let upload_path = dir.join("uploads").join(&info.content_hash);
    assert!(upload_path.exists(), "file should exist on disk");

    // Read content back
    let content = store.read_content(&info.id).expect("read");
    assert_eq!(content, b"persistent data");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_007c_audit_log_to_disk() {
    let path = std::path::PathBuf::from(format!("/tmp/banco_audit_{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let log = super::audit::AuditLog::with_file(path.clone());
    log.push(super::audit::AuditEntry {
        ts: "123".to_string(),
        method: "GET".to_string(),
        path: "/health".to_string(),
        status: 200,
        latency_ms: 1,
    });

    // Verify JSONL file was written
    let content = std::fs::read_to_string(&path).expect("read audit file");
    assert!(content.contains("/health"), "audit entry should be in file");
    assert!(content.contains("GET"));

    let _ = std::fs::remove_file(&path);
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_007a_read_content_in_memory() {
    let store = super::storage::FileStore::in_memory();
    let info = store.store("test.txt", b"Hello, world!");
    let content = store.read_content(&info.id).expect("should have content");
    assert_eq!(content, b"Hello, world!");
}

#[test]
#[allow(non_snake_case)]
fn test_STOR_007_different_content_different_hash() {
    let store = super::storage::FileStore::in_memory();
    let a = store.store("a.txt", b"content A");
    let b = store.store("b.txt", b"content B");
    assert_ne!(a.content_hash, b.content_hash);
}

// ============================================================================
// Data endpoint tests (via router oneshot)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_001_upload_json() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "test.csv", "content": "a,b,c\n1,2,3"});
    let response = app
        .oneshot(
            Request::post("/api/v1/data/upload/json")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("file-"));
    assert_eq!(json["name"], "test.csv");
    assert_eq!(json["content_type"], "text/csv");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_002_list_files() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.files.store("a.txt", b"hello");
    state.files.store("b.csv", b"1,2,3");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/data/files").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["files"].as_array().expect("files").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_003_delete_file() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let info = state.files.store("test.txt", b"data");

    let app = super::router::create_banco_router(state.clone());
    let response = app
        .oneshot(
            Request::delete(&format!("/api/v1/data/files/{}", info.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NO_CONTENT);
    assert!(state.files.is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_DATA_HDL_004_delete_not_found() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::delete("/api/v1/data/files/nonexistent").body(Body::empty()).expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

// ============================================================================
// File info endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_STORE_HDL_005_file_info_csv() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let info = state.files.store("data.csv", b"name,age,city\nAlice,30,NYC\nBob,25,LA\n");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::get(&format!("/api/v1/data/files/{}/info", info.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["name"], "data.csv");
    assert_eq!(json["content_type"], "text/csv");
    assert!(!json["preview_lines"].as_array().expect("preview").is_empty());
    // Schema should be detected for CSV
    if let Some(schema) = json["schema"].as_array() {
        assert_eq!(schema.len(), 3); // name, age, city
        assert_eq!(schema[0]["name"], "name");
    }
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_STORE_HDL_006_file_info_txt() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let info = state.files.store("notes.txt", b"Hello world\nLine 2\n");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::get(&format!("/api/v1/data/files/{}/info", info.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["name"], "notes.txt");
    // TXT files don't have schema
    assert!(json["schema"].is_null());
    assert_eq!(json["preview_lines"].as_array().expect("preview").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_STORE_HDL_007_file_info_not_found() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::get("/api/v1/data/files/nonexistent/info").body(Body::empty()).expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}
