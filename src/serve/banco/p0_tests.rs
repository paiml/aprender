//! P0 cross-cutting tests: OpenAI SDK compat, config persistence, no-telemetry.

// ============================================================================
// OpenAI SDK Compatibility: lowercase role serialization
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P0_role_serializes_lowercase() {
    use crate::serve::templates::Role;
    let json = serde_json::to_string(&Role::User).expect("serialize");
    assert_eq!(json, r#""user""#, "Role must serialize lowercase for OpenAI SDK compat");
    let json = serde_json::to_string(&Role::Assistant).expect("serialize");
    assert_eq!(json, r#""assistant""#);
    let json = serde_json::to_string(&Role::System).expect("serialize");
    assert_eq!(json, r#""system""#);
}

#[test]
#[allow(non_snake_case)]
fn test_P0_role_deserializes_lowercase() {
    use crate::serve::templates::Role;
    let role: Role = serde_json::from_str(r#""user""#).expect("deserialize");
    assert_eq!(role, Role::User);
    let role: Role = serde_json::from_str(r#""assistant""#).expect("deserialize");
    assert_eq!(role, Role::Assistant);
    let role: Role = serde_json::from_str(r#""system""#).expect("deserialize");
    assert_eq!(role, Role::System);
}

#[test]
#[allow(non_snake_case)]
fn test_P0_openai_sdk_request_format() {
    use super::types::BancoChatRequest;
    // This is exactly what the OpenAI Python SDK sends
    let json = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}"#;
    let req: BancoChatRequest = serde_json::from_str(json).expect("OpenAI SDK format must parse");
    assert_eq!(req.messages[0].role, crate::serve::templates::Role::User);
}

// ============================================================================
// OpenAI SDK Compatibility: /v1/ routes
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P0_v1_models_route() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/v1/models").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P0_v1_chat_completions_route() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// No-Telemetry Guarantee
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P0_system_response_telemetry_false() {
    let state = super::state::BancoStateInner::with_defaults();
    let info = state.system_info();
    assert!(!info.telemetry, "Banco must never report telemetry=true");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P0_system_endpoint_telemetry_field() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/system").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["telemetry"], false, "telemetry must be false in system response");
}

// ============================================================================
// Config Persistence
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P0_config_default() {
    let config = super::config::BancoConfig::default();
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 8090);
    assert!((config.inference.temperature - 0.7).abs() < f32::EPSILON);
    assert!((config.budget.daily_limit_usd - 10.0).abs() < f64::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_P0_config_toml_roundtrip() {
    let config = super::config::BancoConfig::default();
    let toml_str = toml::to_string_pretty(&config).expect("serialize");
    let parsed: super::config::BancoConfig = toml::from_str(&toml_str).expect("deserialize");
    assert_eq!(parsed.server.port, 8090);
    assert!((parsed.inference.temperature - 0.7).abs() < f32::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_P0_config_custom_toml() {
    let toml_str = r#"
[server]
host = "0.0.0.0"
port = 9090
privacy_tier = "sovereign"

[inference]
temperature = 0.3
max_tokens = 512

[budget]
daily_limit_usd = 5.0
"#;
    let config: super::config::BancoConfig = toml::from_str(toml_str).expect("parse");
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 9090);
    assert!(matches!(config.server.privacy_tier, super::config::PrivacyTierConfig::Sovereign));
    assert!((config.inference.temperature - 0.3).abs() < f32::EPSILON);
    assert_eq!(config.inference.max_tokens, 512);
    assert!((config.budget.daily_limit_usd - 5.0).abs() < f64::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_P0_config_privacy_tier_conversion() {
    use crate::serve::backends::PrivacyTier;
    assert_eq!(
        PrivacyTier::from(super::config::PrivacyTierConfig::Sovereign),
        PrivacyTier::Sovereign
    );
    assert_eq!(PrivacyTier::from(super::config::PrivacyTierConfig::Private), PrivacyTier::Private);
    assert_eq!(
        PrivacyTier::from(super::config::PrivacyTierConfig::Standard),
        PrivacyTier::Standard
    );
}

#[test]
#[allow(non_snake_case)]
fn test_P0_config_load_missing_file() {
    // load() returns defaults when file doesn't exist — no panic
    let config = super::config::BancoConfig::load();
    assert_eq!(config.server.port, 8090);
}

// ============================================================================
// P1: Tokenize / Detokenize Endpoints
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_tokenize_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"text": "Hello, world!"});
    let response = app
        .oneshot(
            Request::post("/api/v1/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["count"].as_u64().expect("count") > 0);
    assert!(!json["tokens"].as_array().expect("tokens").is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_detokenize_endpoint() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"tokens": [0, 1, 2, 3]});
    let response = app
        .oneshot(
            Request::post("/api/v1/detokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["text"].as_str().expect("text").contains("4 tokens"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P1_chat_completions_includes_context_window() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"messages": [{"role": "user", "content": "Hi"}]});
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["usage"]["context_window"].as_u64().is_some(), "context_window missing");
    assert!(json["usage"]["context_used_pct"].as_f64().is_some(), "context_used_pct missing");
}

// ============================================================================
// Readiness/liveness probes
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P0_liveness_probe() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/health/live").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P0_readiness_probe_no_model() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/health/ready").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    // No model loaded → not ready
    assert_eq!(response.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
}

// ============================================================================
// Ollama pull/delete compat
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P0_ollama_pull() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "llama3:latest"});
    let response = app
        .oneshot(
            Request::post("/api/pull")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["status"], "success");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_P0_ollama_delete() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"name": "llama3"});
    let response = app
        .oneshot(
            Request::delete("/api/delete")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Scoped API keys
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_P0_scoped_key_chat() {
    use super::auth::{AuthStore, KeyScope};

    let (store, _admin_key) = AuthStore::api_key_mode();
    let chat_key = store.generate_scoped_key(KeyScope::Chat);
    assert!(chat_key.key.starts_with("bk_"));

    // Chat key should allow chat endpoints
    assert!(store.validate_for_path(&chat_key.key, "/api/v1/chat/completions"));
    assert!(store.validate_for_path(&chat_key.key, "/api/v1/models"));

    // Chat key should NOT allow admin endpoints
    assert!(!store.validate_for_path(&chat_key.key, "/api/v1/models/load"));
    assert!(!store.validate_for_path(&chat_key.key, "/api/v1/train/start"));
}

#[test]
#[allow(non_snake_case)]
fn test_P0_scoped_key_train() {
    use super::auth::{AuthStore, KeyScope};

    let (store, _admin_key) = AuthStore::api_key_mode();
    let train_key = store.generate_scoped_key(KeyScope::Train);

    // Train key should allow training endpoints
    assert!(store.validate_for_path(&train_key.key, "/api/v1/train/start"));
    assert!(store.validate_for_path(&train_key.key, "/api/v1/chat/completions"));

    // Train key should NOT allow admin endpoints
    assert!(!store.validate_for_path(&train_key.key, "/api/v1/models/load"));
}

#[test]
#[allow(non_snake_case)]
fn test_P0_scoped_key_admin() {
    use super::auth::{AuthStore, KeyScope};

    let (store, admin_key) = AuthStore::api_key_mode();
    // Admin key allows everything
    assert!(store.validate_for_path(&admin_key, "/api/v1/models/load"));
    assert!(store.validate_for_path(&admin_key, "/api/v1/train/start"));
    assert!(store.validate_for_path(&admin_key, "/api/v1/config"));

    assert_eq!(store.list_keys().len(), 1); // just the generated admin key
    let _ = store.generate_scoped_key(KeyScope::Chat);
    assert_eq!(store.list_keys().len(), 2);
}
