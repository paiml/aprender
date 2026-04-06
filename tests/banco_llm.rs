//! Banco L2 Integration Tests — probar::llm against real TCP server.
//!
//! These tests start a real Banco HTTP server, use probar's LlmClient
//! to send actual OpenAI-compatible requests over TCP, and validate
//! responses with LlmAssertion builders.
//!
//! This is the test suite that would have caught "nothing works" in UAT.
//!
//! Non-chat endpoint tests are in banco_endpoints.rs.

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

    // Give server time to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    (base, handle)
}

// ============================================================================
// L2: Chat completion tests via probar LlmClient
// ============================================================================

#[tokio::test]
async fn l2_health_via_llm_client() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let healthy = client.health_check().await.unwrap();
    assert!(healthy, "Server should be healthy");

    handle.abort();
}

#[tokio::test]
async fn l2_chat_completion_valid_structure() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "Hello Banco!".to_string(),
        }],
        temperature: None,
        max_tokens: Some(32),
        stream: None,
    };

    let timed = client.send(&request).await.unwrap();

    // Structural assertions via probar
    let assertions = jugar_probar::llm::LlmAssertion::new().assert_response_valid();

    assert!(
        assertions.run_all_pass(&timed),
        "Response structure invalid: {:?}",
        assertions.run(&timed)
    );

    handle.abort();
}

#[tokio::test]
async fn l2_chat_completion_has_content() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "What is Banco?".to_string(),
        }],
        temperature: None,
        max_tokens: Some(64),
        stream: None,
    };

    let timed = client.send(&request).await.unwrap();
    let content = &timed.response.choices[0].message.content;

    assert!(!content.is_empty(), "Response content is empty");
    // Without a loaded model, should mention "No model loaded" or similar
    // With a model, should have actual generated text
    assert!(content.len() > 5, "Response too short: {content}");

    handle.abort();
}

#[tokio::test]
async fn l2_chat_latency_under_budget() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "hi".to_string(),
        }],
        temperature: None,
        max_tokens: Some(8),
        stream: None,
    };

    let timed = client.send(&request).await.unwrap();

    let assertions =
        jugar_probar::llm::LlmAssertion::new().assert_latency_under(Duration::from_secs(10));

    assert!(
        assertions.run_all_pass(&timed),
        "Latency exceeded 10s: {:?}ms",
        timed.latency.as_millis()
    );

    handle.abort();
}

#[tokio::test]
async fn l2_chat_streaming_sse() {
    let (base, handle) = start_server().await;
    let client = jugar_probar::llm::LlmClient::new(&base, "local");

    let request = jugar_probar::llm::ChatRequest {
        model: "local".to_string(),
        messages: vec![jugar_probar::llm::ChatMessage {
            role: jugar_probar::llm::Role::User,
            content: "Count to 3".to_string(),
        }],
        temperature: None,
        max_tokens: Some(16),
        stream: Some(true),
    };

    let streamed = client.chat_completion_stream(&request).await.unwrap();

    assert!(!streamed.content.is_empty(), "Stream produced no content");
    assert!(streamed.ttft < Duration::from_secs(10), "TTFT too slow: {:?}", streamed.ttft);

    handle.abort();
}
