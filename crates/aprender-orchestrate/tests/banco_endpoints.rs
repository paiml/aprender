//! Banco L2 Integration Tests — raw HTTP endpoint validation.
//!
//! Tests non-chat endpoints (system, tools, MCP, metrics, data, config, etc.)
//! against a real Banco TCP server. Complements banco_llm.rs which focuses
//! on chat completions via probar::llm.

#![cfg(feature = "banco")]

use std::time::Duration;

/// Start a Banco server on a random port. Returns (base_url, abort_handle).
async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");

    let state = batuta::serve::banco::state::BancoStateInner::with_defaults();
    let app = batuta::serve::banco::router::create_banco_router(state);

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    (base, handle)
}

// ============================================================================
// System, health, metrics
// ============================================================================

#[tokio::test]
async fn l2_system_info_returns_json() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/system")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["endpoints"].as_u64().unwrap() > 0);
    assert_eq!(json["telemetry"], false);
    handle.abort();
}

#[tokio::test]
async fn l2_system_info_tokenizer_field() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/system")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["tokenizer"].is_null());
    assert_eq!(json["model_loaded"], false);
    assert!(json["version"].is_string());
    handle.abort();
}

#[tokio::test]
async fn l2_health_probes() {
    let (base, handle) = start_server().await;
    let live = reqwest::get(format!("{base}/health/live")).await.unwrap();
    assert_eq!(live.status(), 200);
    let ready = reqwest::get(format!("{base}/health/ready")).await.unwrap();
    assert_eq!(ready.status(), 503); // no model loaded
    handle.abort();
}

#[tokio::test]
async fn l2_prometheus_metrics() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/metrics")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("banco_requests_total"));
    assert!(body.contains("banco_uptime_seconds"));
    handle.abort();
}

// ============================================================================
// Browser, tools, MCP
// ============================================================================

#[tokio::test]
async fn l2_browser_ui_serves_html() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.contains("text/html"));
    let body = resp.text().await.unwrap();
    assert!(body.contains("Banco"));
    assert!(body.contains("/api/v1/chat/completions"));
    handle.abort();
}

#[tokio::test]
async fn l2_tool_calculator() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/tools/execute"))
        .json(&serde_json::json!({
            "id": "l2-test",
            "name": "calculator",
            "arguments": {"expression": "6 * 7"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["content"], "42");
    handle.abort();
}

#[tokio::test]
async fn l2_mcp_initialize() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/mcp"))
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["result"]["serverInfo"]["name"], "banco");
    handle.abort();
}

// ============================================================================
// Data, RAG, tokenize, embeddings
// ============================================================================

#[tokio::test]
async fn l2_upload_and_rag_search() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/api/v1/data/upload/json"))
        .json(&serde_json::json!({
            "name": "test.txt",
            "content": "Banco is a sovereign AI workbench"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp =
        reqwest::get(format!("{base}/api/v1/rag/search?q=sovereign+workbench")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(!json["results"].as_array().unwrap().is_empty());

    handle.abort();
}

#[tokio::test]
async fn l2_tokenize_returns_tokens() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/tokenize"))
        .json(&serde_json::json!({"text": "Hello world"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["count"].as_u64().unwrap() > 0);
    assert!(json["tokens"].is_array());
    handle.abort();
}

#[tokio::test]
async fn l2_detokenize_returns_text() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/detokenize"))
        .json(&serde_json::json!({"tokens": [1, 2, 3]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["text"].is_string());
    handle.abort();
}

#[tokio::test]
async fn l2_embeddings_endpoint() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({
            "model": "local",
            "input": "Test embedding"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["data"].is_array());

    handle.abort();
}

// ============================================================================
// Models, Ollama, config, audit, conversations, batch
// ============================================================================

#[tokio::test]
async fn l2_models_status_no_model() {
    let (base, handle) = start_server().await;

    let resp = reqwest::get(format!("{base}/api/v1/models/status")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["loaded"], false);
    assert!(json["tokenizer"].is_null());

    handle.abort();
}

#[tokio::test]
async fn l2_openai_models_list() {
    let (base, handle) = start_server().await;

    let resp = reqwest::get(format!("{base}/v1/models")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());

    handle.abort();
}

#[tokio::test]
async fn l2_ollama_tags() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/tags")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["models"].is_array());
    handle.abort();
}

#[tokio::test]
async fn l2_conversations_crud() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = reqwest::get(format!("{base}/api/v1/conversations")).await.unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "Test conversation"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_config_get_put() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = reqwest::get(format!("{base}/api/v1/config")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json.is_object());

    let resp = client
        .put(format!("{base}/api/v1/config"))
        .json(&serde_json::json!({"theme": "dark"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_audit_log() {
    let (base, handle) = start_server().await;

    reqwest::get(format!("{base}/api/v1/system")).await.unwrap();

    let resp = reqwest::get(format!("{base}/api/v1/audit")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["entries"].is_array());

    handle.abort();
}

#[tokio::test]
async fn l2_text_completions() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/completions"))
        .json(&serde_json::json!({
            "prompt": "The capital of France is",
            "max_tokens": 16
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["choices"].is_array());
    assert!(!json["choices"][0]["text"].as_str().unwrap().is_empty());
    handle.abort();
}

#[tokio::test]
async fn l2_chat_parameters_get_put() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = reqwest::get(format!("{base}/api/v1/chat/parameters")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["temperature"].is_f64());

    let resp = client
        .put(format!("{base}/api/v1/chat/parameters"))
        .json(&serde_json::json!({"temperature": 0.3, "top_k": 20, "max_tokens": 128}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_batch_completions() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/api/v1/batch"))
        .json(&serde_json::json!({
            "items": [
                {"id": "b1", "messages": [{"role": "user", "content": "Hi"}]},
                {"id": "b2", "messages": [{"role": "user", "content": "Hello"}]}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["results"].is_array());
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);

    handle.abort();
}

// ============================================================================
// Export/import, OpenAI compat, audio formats
// ============================================================================

#[tokio::test]
async fn l2_conversation_export_import_roundtrip() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Create a conversation via chat
    client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "Roundtrip test"}]
        }))
        .send()
        .await
        .unwrap();

    // Export
    let resp = reqwest::get(format!("{base}/api/v1/conversations/export")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let exported: serde_json::Value = resp.json().await.unwrap();
    assert!(exported.is_array());

    // Import back
    let resp = client
        .post(format!("{base}/api/v1/conversations/import"))
        .json(&exported)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["imported"].as_u64().is_some());

    handle.abort();
}

#[tokio::test]
async fn l2_openai_model_by_id_no_model() {
    let (base, handle) = start_server().await;
    // No model loaded → 404 is correct
    let resp = reqwest::get(format!("{base}/v1/models/local")).await.unwrap();
    assert_eq!(resp.status(), 404);
    handle.abort();
}

#[tokio::test]
async fn l2_audio_formats() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/audio/formats")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["formats"].is_array());
    handle.abort();
}

#[tokio::test]
async fn l2_mcp_info() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/mcp/info")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["server"], "banco");
    assert_eq!(json["protocol"], "mcp");
    handle.abort();
}

#[tokio::test]
async fn l2_health_endpoint() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    handle.abort();
}

// ============================================================================
// Model load/unload (Fixes #52)
// ============================================================================

#[tokio::test]
async fn l2_model_load_nonexistent() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/models/load"))
        .json(&serde_json::json!({"model": "/tmp/nonexistent-model.gguf"}))
        .send()
        .await
        .unwrap();
    // Load returns 200 with metadata even for nonexistent files (format detected, size=0)
    assert!(resp.status().is_success() || resp.status() == 500);
    handle.abort();
}

#[tokio::test]
async fn l2_model_unload_without_model() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client.post(format!("{base}/api/v1/models/unload")).send().await.unwrap();
    // Unload without loaded model returns error
    assert_eq!(resp.status(), 400);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["error"].is_object() || json["message"].is_string());
    handle.abort();
}
