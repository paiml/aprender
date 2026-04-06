//! P2 cross-cutting tests: system prompt presets + Ollama API compat.

// ============================================================================
// System Prompt Presets
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_store_defaults() {
    let store = super::prompts::PromptStore::new();
    let presets = store.list();
    assert!(presets.len() >= 3, "should have built-in presets");
    assert!(store.get("coding").is_some());
    assert!(store.get("concise").is_some());
    assert!(store.get("tutor").is_some());
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_store_create() {
    let store = super::prompts::PromptStore::new();
    let preset = store.create("My Custom", "You are a pirate. Arr!");
    assert!(preset.id.starts_with("preset-"));
    assert_eq!(preset.name, "My Custom");
    assert_eq!(preset.content, "You are a pirate. Arr!");
    assert!(store.get(&preset.id).is_some());
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_store_delete() {
    let store = super::prompts::PromptStore::new();
    assert!(store.delete("coding"));
    assert!(store.get("coding").is_none());
    assert!(!store.delete("nonexistent"));
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_expand_preset_ref() {
    let store = super::prompts::PromptStore::new();
    let expanded = store.expand("@preset:coding");
    assert!(expanded.contains("expert software engineer"));
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_expand_no_ref() {
    let store = super::prompts::PromptStore::new();
    let expanded = store.expand("Just a normal message");
    assert_eq!(expanded, "Just a normal message");
}

#[test]
#[allow(non_snake_case)]
fn test_P2_prompt_expand_unknown_ref() {
    let store = super::prompts::PromptStore::new();
    let expanded = store.expand("@preset:nonexistent");
    assert_eq!(expanded, "@preset:nonexistent");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_prompts_list_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/prompts").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    let presets = json["presets"].as_array().expect("presets array");
    assert!(presets.len() >= 3);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_prompts_create_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "Test", "content": "Be helpful"});
    let response = app
        .oneshot(
            Request::post("/api/v1/prompts")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("preset-"));
    assert_eq!(json["name"], "Test");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_chat_expands_preset_ref() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [
            {"role": "system", "content": "@preset:coding"},
            {"role": "user", "content": "Hello"}
        ]
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
    // The formatted output should contain the expanded preset content
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    // prompt_len should reflect expanded content (> 2 tokens for "Hello" alone)
    let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens");
    assert!(prompt_tokens > 5, "preset should expand, increasing token count");
}

// ============================================================================
// Inference Parameter Tuning
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_get_default_parameters() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/chat/parameters").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!((json["temperature"].as_f64().expect("temp") - 0.7).abs() < 0.01);
    assert!((json["top_p"].as_f64().expect("top_p") - 1.0).abs() < 0.01);
    assert_eq!(json["top_k"].as_u64().expect("top_k"), 40);
    assert_eq!(json["max_tokens"].as_u64().expect("max_tokens"), 256);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_update_parameters() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();

    // Update
    let app = super::router::create_banco_router(state.clone());
    let body = serde_json::json!({
        "temperature": 0.3, "top_p": 0.9, "top_k": 50,
        "repeat_penalty": 1.2, "max_tokens": 512
    });
    let response = app
        .oneshot(
            Request::put("/api/v1/chat/parameters")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!((json["temperature"].as_f64().expect("temp") - 0.3).abs() < 0.01);
    assert_eq!(json["max_tokens"].as_u64().expect("max_tokens"), 512);

    // Verify persists in state
    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/chat/parameters").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!((json["temperature"].as_f64().expect("temp") - 0.3).abs() < 0.01);
}

// ============================================================================
// API Key Auth
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P2_auth_store_local_mode() {
    let store = super::auth::AuthStore::local();
    assert!(!store.requires_auth());
    assert!(store.validate("anything")); // local mode accepts all
    assert_eq!(store.mode(), super::auth::AuthMode::Local);
}

#[test]
#[allow(non_snake_case)]
fn test_P2_auth_store_api_key_mode() {
    let (store, key) = super::auth::AuthStore::api_key_mode();
    assert!(store.requires_auth());
    assert!(key.starts_with("bk_"));
    assert!(store.validate(&key));
    assert!(!store.validate("wrong_key"));
    assert_eq!(store.key_count(), 1);
}

#[test]
#[allow(non_snake_case)]
fn test_P2_key_scope_chat() {
    let scope = super::auth::KeyScope::Chat;
    assert!(scope.allows_path("/api/v1/chat/completions"));
    assert!(scope.allows_path("/health"));
    assert!(scope.allows_path("/api/v1/models"));
    assert!(scope.allows_path("/api/v1/embeddings"));
    assert!(scope.allows_path("/api/v1/prompts"));
    assert!(!scope.allows_path("/api/v1/train/start"));
}

#[test]
#[allow(non_snake_case)]
fn test_P2_key_scope_admin() {
    let scope = super::auth::KeyScope::Admin;
    assert!(scope.allows_path("/anything"));
    assert!(scope.allows_path("/api/v1/models/load"));
    assert!(scope.allows_path("/api/v1/config"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_auth_local_mode_no_header_needed() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    // Default state uses local mode — no auth needed
    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/system").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Ollama API Compat
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_tags() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/tags").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(!json["models"].as_array().expect("models").is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_chat() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hello!"}]
    });
    let response = app
        .oneshot(
            Request::post("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["done"], true);
    assert_eq!(json["message"]["role"], "assistant");
    assert!(json["message"]["content"].as_str().expect("content").contains("No model loaded"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_show() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "llama3"});
    let response = app
        .oneshot(
            Request::post("/api/show")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["modelfile"].as_str().expect("modelfile").contains("llama3"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_generate() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "model": "llama3",
        "prompt": "Why is the sky blue?"
    });
    let response = app
        .oneshot(
            Request::post("/api/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["done"], true);
    assert!(json["response"].as_str().expect("response").contains("No model loaded"));
    assert_eq!(json["model"], "llama3");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_ollama_generate_with_system() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "prompt": "Hello",
        "system": "You are a pirate"
    });
    let response = app
        .oneshot(
            Request::post("/api/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["done"], true);
    // With system prompt, formatted_len should be longer
    let eval_count = json["eval_count"].as_u64().expect("eval_count");
    assert!(eval_count > 0);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P2_config_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/config").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["server"]["port"], 8090);
    assert!(json["inference"]["temperature"].as_f64().is_some());
}
