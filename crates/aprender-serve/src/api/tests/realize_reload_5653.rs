//! #5653: `POST /realize/reload` — every branch of `realize_reload_handler`.
//!
//! The handler validates in a fixed order (registry mode, path, registered id, file on disk)
//! and only then answers 200; each rung below fails exactly one of them.

use axum::http::StatusCode;

use super::native_routes_2376::post;
use crate::api::AppState;
use crate::layers::{Model, ModelConfig};
use crate::registry::ModelRegistry;
use crate::tokenizer::BPETokenizer;

fn registry_state() -> AppState {
    let config = ModelConfig {
        vocab_size: 8,
        hidden_dim: 8,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let vocab: Vec<String> = (0..8)
        .map(|i| if i == 0 { "<unk>".to_string() } else { format!("t{i}") })
        .collect();
    let registry = ModelRegistry::new(4);
    for id in ["default", "known"] {
        let model = Model::new(config.clone()).expect("model");
        let tokenizer = BPETokenizer::new(vocab.clone(), vec![], "<unk>").expect("tokenizer");
        registry.register(id, model, tokenizer).expect("register");
    }
    AppState::with_registry(registry, "default").expect("registry state")
}

fn model_file() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().expect("temp model file")
}

async fn reload(state: AppState, json: &str) -> (StatusCode, String) {
    post(state, "/realize/reload", json).await
}

#[tokio::test]
async fn without_registry_mode_it_is_501_and_names_no_dead_flag() {
    let state = AppState::demo().expect("demo state");
    let (status, body) = reload(state, r#"{"path":"/x"}"#).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "body: {body}");
    assert!(body.contains("multi-model registry mode"), "body: {body}");
    assert!(!body.contains("Start server with"), "must not advise a remedy: {body}");
}

#[tokio::test]
async fn missing_path_is_400_before_the_model_id_is_looked_up() {
    let (status, body) = reload(registry_state(), r#"{"model":"nope"}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert!(body.contains("Model path is required"), "body: {body}");
}

#[tokio::test]
async fn unregistered_model_id_is_404() {
    let file = model_file();
    let json = format!(r#"{{"model":"nope","path":{:?}}}"#, file.path());
    let (status, body) = reload(registry_state(), &json).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body: {body}");
    assert!(body.contains("'nope' not found in registry"), "body: {body}");
}

#[tokio::test]
async fn model_file_that_does_not_exist_is_400() {
    let (status, body) =
        reload(registry_state(), r#"{"model":"known","path":"/nonexistent/m.gguf"}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert!(body.contains("Model file not found: /nonexistent/m.gguf"), "body: {body}");
}

#[tokio::test]
async fn registered_model_and_existing_file_is_200_with_timing() {
    let file = model_file();
    let json = format!(r#"{{"model":"known","path":{:?}}}"#, file.path());
    let (status, body) = reload(registry_state(), &json).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("json body");
    assert_eq!(v["success"], true);
    let message = v["message"].as_str().expect("message");
    assert!(message.contains("'known'"), "message: {message}");
    assert!(message.contains(file.path().to_str().expect("utf8")), "message: {message}");
    assert!(v["reload_time_ms"].is_u64(), "body: {body}");
}

#[tokio::test]
async fn omitted_model_id_reloads_the_default_model() {
    let file = model_file();
    let json = format!(r#"{{"path":{:?}}}"#, file.path());
    let (status, body) = reload(registry_state(), &json).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(body.contains("Model 'default' reload validated"), "body: {body}");
}
