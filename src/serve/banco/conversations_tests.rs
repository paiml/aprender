//! Conversation persistence tests.

use crate::serve::templates::ChatMessage;

// ============================================================================
// ConversationStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_CONV_001_create_and_get() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("test-model");
    assert!(id.starts_with("conv-"));
    let conv = store.get(&id).expect("conversation should exist");
    assert_eq!(conv.meta.model, "test-model");
    assert_eq!(conv.meta.title, "New conversation");
    assert!(conv.messages.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_002_append_messages() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    store.append(&id, ChatMessage::user("Hello")).expect("append");
    store.append(&id, ChatMessage::assistant("Hi!")).expect("append");
    let conv = store.get(&id).expect("get");
    assert_eq!(conv.messages.len(), 2);
    assert_eq!(conv.meta.message_count, 2);
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_003_auto_title() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    store
        .append(&id, ChatMessage::user("What is the meaning of life and everything"))
        .expect("append");
    let conv = store.get(&id).expect("get");
    assert_eq!(conv.meta.title, "What is the meaning of...");
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_003b_auto_title_short() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    store.append(&id, ChatMessage::user("Hello")).expect("append");
    let conv = store.get(&id).expect("get");
    assert_eq!(conv.meta.title, "Hello");
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_004_list_conversations() {
    let store = super::conversations::ConversationStore::in_memory();
    store.create("a");
    store.create("b");
    store.create("c");
    let list = store.list();
    assert_eq!(list.len(), 3);
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_005_delete() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("model");
    assert_eq!(store.len(), 1);
    store.delete(&id).expect("delete");
    assert_eq!(store.len(), 0);
    assert!(store.get(&id).is_none());
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_006_delete_not_found() {
    let store = super::conversations::ConversationStore::in_memory();
    let result = store.delete("nonexistent");
    assert!(result.is_err());
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_007_append_not_found() {
    let store = super::conversations::ConversationStore::in_memory();
    let result = store.append("nonexistent", ChatMessage::user("hi"));
    assert!(result.is_err());
}

// ============================================================================
// Conversation HTTP endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_001_create_conversation() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({});
    let response = app
        .oneshot(
            Request::post("/api/v1/conversations")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("conv-"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_002_list_conversations() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.conversations.create("model-a");
    state.conversations.create("model-b");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/conversations").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["conversations"].as_array().expect("array").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_003_get_conversation_not_found() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::get("/api/v1/conversations/nonexistent").body(Body::empty()).expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_004_chat_appends_to_conversation() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let conv_id = state.conversations.create("echo");

    let app = super::router::create_banco_router(state.clone());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "conversation_id": conv_id
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

    // Verify both user message and assistant response were saved
    let conv = state.conversations.get(&conv_id).expect("conversation");
    assert_eq!(conv.messages.len(), 2, "user message + assistant response");
    assert_eq!(conv.messages[0].content, "Hello!");
    assert_eq!(conv.meta.title, "Hello!");
}

// ============================================================================
// Search
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_CONV_010_search_by_content() {
    let store = super::conversations::ConversationStore::in_memory();
    let id1 = store.create("m");
    store.append(&id1, ChatMessage::user("Rust ownership")).expect("append");
    let id2 = store.create("m");
    store.append(&id2, ChatMessage::user("Python decorators")).expect("append");

    let results = store.search("rust");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, id1);

    let results = store.search("python");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, id2);

    let results = store.search("nonexistent");
    assert!(results.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_011_search_by_title() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("m");
    store.append(&id, ChatMessage::user("How does Rust handle memory safety")).expect("append");
    // Title auto-generated from first message
    let results = store.search("memory safety");
    assert_eq!(results.len(), 1);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_007_search_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let id = state.conversations.create("m");
    state.conversations.append(&id, ChatMessage::user("Rust borrow checker")).expect("append");
    state.conversations.create("m");
    state
        .conversations
        .append(&state.conversations.list()[1].id, ChatMessage::user("Python GIL"))
        .expect("append");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::get("/api/v1/conversations/search?q=rust").body(Body::empty()).expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["conversations"].as_array().expect("convs").len(), 1);
}

// ============================================================================
// Rename
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_CONV_012_rename() {
    let store = super::conversations::ConversationStore::in_memory();
    let id = store.create("m");
    store.rename(&id, "New Title").expect("rename");
    assert_eq!(store.get(&id).expect("get").meta.title, "New Title");
}

// ============================================================================
// Export / Import
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_CONV_008_export_all() {
    let store = super::conversations::ConversationStore::in_memory();
    store.create("a");
    store.create("b");
    let _ = store.append(&store.list()[0].id, ChatMessage::user("Hello"));
    let exported = store.export_all();
    assert_eq!(exported.len(), 2);
}

#[test]
#[allow(non_snake_case)]
fn test_CONV_009_import_all() {
    let store = super::conversations::ConversationStore::in_memory();
    store.create("a");
    let exported = store.export_all();

    let store2 = super::conversations::ConversationStore::in_memory();
    assert_eq!(store2.len(), 0);
    let count = store2.import_all(exported);
    assert_eq!(count, 1);
    assert_eq!(store2.len(), 1);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_005_export_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.conversations.create("model-a");
    let _ = state
        .conversations
        .append(&state.conversations.list()[0].id, ChatMessage::user("Hello export"));

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/conversations/export").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: Vec<serde_json::Value> = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json.len(), 1);
    assert!(!json[0]["messages"].as_array().expect("msgs").is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_CONV_HDL_006_import_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    // Export from one state
    let state1 = super::state::BancoStateInner::with_defaults();
    state1.conversations.create("model-a");
    let exported = state1.conversations.export_all();
    let export_json = serde_json::to_vec(&exported).expect("json");

    // Import to another state
    let state2 = super::state::BancoStateInner::with_defaults();
    let app = super::router::create_banco_router(state2.clone());
    let response = app
        .oneshot(
            Request::post("/api/v1/conversations/import")
                .header("content-type", "application/json")
                .body(Body::from(export_json))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["imported"], 1);

    // Verify imported
    assert_eq!(state2.conversations.len(), 1);
}
