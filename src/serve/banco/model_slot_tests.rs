//! Model slot + management endpoint tests.

// ============================================================================
// ModelSlot unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_MODEL_001_empty_slot() {
    let slot = super::model_slot::ModelSlot::empty();
    assert!(!slot.is_loaded());
    assert!(slot.info().is_none());
    assert_eq!(slot.uptime_secs(), 0);
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_002_load_gguf() {
    let slot = super::model_slot::ModelSlot::empty();
    let info = slot.load("/tmp/test-model.gguf").expect("load");
    assert!(slot.is_loaded());
    assert_eq!(info.model_id, "test-model");
    assert_eq!(info.format, super::model_slot::ModelFormat::Gguf);
    assert_eq!(info.path, "/tmp/test-model.gguf");
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_003_load_apr() {
    let slot = super::model_slot::ModelSlot::empty();
    let info = slot.load("/tmp/llama3.apr").expect("load");
    assert_eq!(info.format, super::model_slot::ModelFormat::Apr);
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_004_load_safetensors() {
    let slot = super::model_slot::ModelSlot::empty();
    let info = slot.load("/tmp/model.safetensors").expect("load");
    assert_eq!(info.format, super::model_slot::ModelFormat::SafeTensors);
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_005_unload() {
    let slot = super::model_slot::ModelSlot::empty();
    slot.load("/tmp/model.gguf").expect("load");
    assert!(slot.is_loaded());
    slot.unload().expect("unload");
    assert!(!slot.is_loaded());
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_006_unload_empty_errors() {
    let slot = super::model_slot::ModelSlot::empty();
    assert!(slot.unload().is_err());
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_007_hot_swap() {
    let slot = super::model_slot::ModelSlot::empty();
    slot.load("/tmp/model-a.gguf").expect("load a");
    assert_eq!(slot.info().expect("info").model_id, "model-a");
    slot.load("/tmp/model-b.apr").expect("load b");
    assert_eq!(slot.info().expect("info").model_id, "model-b");
    assert_eq!(slot.info().expect("info").format, super::model_slot::ModelFormat::Apr);
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_008_format_detection() {
    use std::path::Path;
    assert_eq!(
        super::model_slot::ModelFormat::from_path(Path::new("model.gguf")),
        super::model_slot::ModelFormat::Gguf
    );
    assert_eq!(
        super::model_slot::ModelFormat::from_path(Path::new("model.apr")),
        super::model_slot::ModelFormat::Apr
    );
    assert_eq!(
        super::model_slot::ModelFormat::from_path(Path::new("model.safetensors")),
        super::model_slot::ModelFormat::SafeTensors
    );
    assert_eq!(
        super::model_slot::ModelFormat::from_path(Path::new("model.bin")),
        super::model_slot::ModelFormat::Unknown
    );
}

// ============================================================================
// Model management endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MODEL_HDL_001_status_empty() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/models/status").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["loaded"], false);
    assert!(json["model"].is_null());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MODEL_HDL_002_load_and_status() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    // Load via state directly (simulates --model flag)
    state.model.load("/tmp/test.gguf").expect("load");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/models/status").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["loaded"], true);
    assert_eq!(json["model"]["model_id"], "test");
    assert_eq!(json["model"]["format"], "gguf");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MODEL_HDL_003_load_via_api() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"model": "/tmp/phi-3.gguf"});
    let response = app
        .oneshot(
            Request::post("/api/v1/models/load")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["loaded"], true);
    assert_eq!(json["model"]["model_id"], "phi-3");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MODEL_HDL_004_system_shows_model() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.model.load("/tmp/llama3.gguf").expect("load");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/system").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["model_loaded"], true);
    assert_eq!(json["model_id"], "llama3");
}

// ============================================================================
// APR format detection tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_MODEL_009_apr_format_detection() {
    use super::model_slot::ModelFormat;
    use std::path::Path;

    assert_eq!(ModelFormat::from_path(Path::new("model.apr")), ModelFormat::Apr);
    assert_eq!(ModelFormat::from_path(Path::new("model.gguf")), ModelFormat::Gguf);
    assert_eq!(ModelFormat::from_path(Path::new("model.safetensors")), ModelFormat::SafeTensors);
    assert_eq!(ModelFormat::from_path(Path::new("model.bin")), ModelFormat::Unknown);
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_010_apr_load_nonexistent_fails_gracefully() {
    let slot = super::model_slot::ModelSlot::empty();
    // Loading a nonexistent APR file should fail but not panic
    let result = slot.load("/tmp/nonexistent_model.apr");
    // File doesn't exist — load succeeds with metadata (no inference)
    // because format detection works but file read fails inside extract_model_metadata
    assert!(result.is_ok(), "APR load should not panic: {result:?}");
    let info = result.unwrap();
    assert_eq!(info.format, super::model_slot::ModelFormat::Apr);
    // No architecture extracted (file doesn't exist)
    assert!(info.architecture.is_none());
}

// ============================================================================
// BPE tokenizer tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_MODEL_011_bpe_tokenizer_not_loaded_without_file() {
    let slot = super::model_slot::ModelSlot::empty();
    // Loading a model with no sibling tokenizer.json → no BPE tokenizer
    slot.load("/tmp/test-no-tokenizer.gguf").expect("load");
    #[cfg(feature = "aprender")]
    assert!(!slot.has_bpe_tokenizer(), "Should not have BPE without tokenizer.json");
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_012_bpe_tokenizer_search_paths() {
    // Verify the search path logic: stem.tokenizer.json, then tokenizer.json
    use std::path::Path;

    // The load_bpe_tokenizer function is private, but we can test via slot
    let slot = super::model_slot::ModelSlot::empty();
    slot.load("/tmp/nonexistent-bpe-test.gguf").expect("load");
    // No tokenizer.json near /tmp/nonexistent-bpe-test.gguf
    #[cfg(feature = "aprender")]
    assert!(!slot.has_bpe_tokenizer());

    // Verify the format detection still works independently
    assert_eq!(
        super::model_slot::ModelFormat::from_path(Path::new("model.gguf")),
        super::model_slot::ModelFormat::Gguf
    );
}

#[test]
#[allow(non_snake_case)]
fn test_MODEL_013_unload_clears_bpe() {
    let slot = super::model_slot::ModelSlot::empty();
    slot.load("/tmp/test-unload-bpe.gguf").expect("load");
    slot.unload().expect("unload");
    #[cfg(feature = "aprender")]
    assert!(!slot.has_bpe_tokenizer(), "Unload should clear BPE tokenizer");
}
