//! FALSIFY-LOGPROBS-WIRE-4971: `logprobs` / `top_logprobs` on `/v1/chat/completions`
//! (ASOC-INV-021), through the real router on a quantized CPU server.
//!
//! Before #4971 the request type had neither field: serde dropped both, the reply
//! carried no `logprobs`, and a client that asked for them was answered as if it had
//! not. Every test here asserts what that client observes.

use axum::http::StatusCode;
use serde_json::Value;

use super::native_routes_2376::{post, quantized_state};

const CHAT: &str = "/v1/chat/completions";

fn chat_body(extra: &str) -> String {
    format!(
        r#"{{"model":"default","messages":[{{"role":"user","content":"Hi"}}],"max_tokens":4,"temperature":0{extra}}}"#
    )
}

async fn chat_json(extra: &str) -> Value {
    let (status, body) = post(quantized_state(), CHAT, &chat_body(extra)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).expect("chat reply is JSON")
}

fn logprob(entry: &Value) -> f64 {
    entry["logprob"].as_f64().expect("logprob is a number")
}

#[tokio::test]
async fn logprobs_has_one_entry_per_completion_token_with_its_best_alternatives() {
    let reply = chat_json(r#","logprobs":true,"top_logprobs":3"#).await;
    let completion_tokens = reply["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens");
    let content = reply["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("choices[0].logprobs.content is an array");
    assert!(
        completion_tokens > 0,
        "the fixture generated nothing: {reply}"
    );
    assert_eq!(content.len() as u64, completion_tokens, "{reply}");
    for entry in content {
        assert!(entry["token"].is_string(), "{entry}");
        assert!(entry["bytes"].is_array(), "{entry}");
        assert!(logprob(entry) <= 0.0, "{entry}");
        let tops = entry["top_logprobs"].as_array().expect("top_logprobs");
        assert_eq!(tops.len(), 3, "{entry}");
        assert!(
            tops.windows(2).all(|w| logprob(&w[0]) >= logprob(&w[1])),
            "{entry}"
        );
        assert!(tops.iter().all(|t| logprob(t) <= 0.0), "{entry}");
        // Greedy: the chosen token is a best one. Compared by logprob, not by id,
        // because the fixture's logits may tie.
        assert!((logprob(entry) - logprob(&tops[0])).abs() < 1e-4, "{entry}");
    }
}

#[tokio::test]
async fn asking_for_logprobs_changes_no_token_and_not_asking_adds_no_key() {
    let with = chat_json(r#","logprobs":true,"top_logprobs":2"#).await;
    let without = chat_json("").await;
    assert_eq!(
        with["choices"][0]["message"]["content"],
        without["choices"][0]["message"]["content"]
    );
    assert_eq!(
        with["usage"]["completion_tokens"],
        without["usage"]["completion_tokens"]
    );
    let choice = without["choices"][0].as_object().expect("choice object");
    assert!(!choice.contains_key("logprobs"), "{without}");
}

#[tokio::test]
async fn logprobs_without_top_logprobs_carries_empty_alternatives() {
    let reply = chat_json(r#","logprobs":true"#).await;
    let content = reply["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("content");
    assert!(!content.is_empty(), "{reply}");
    assert!(content
        .iter()
        .all(|e| e["top_logprobs"].as_array().is_some_and(Vec::is_empty)));
}

#[tokio::test]
async fn top_logprobs_out_of_range_or_without_logprobs_is_refused_naming_the_field() {
    for extra in [
        r#","logprobs":true,"top_logprobs":21"#,
        r#","logprobs":true,"top_logprobs":-1"#,
        r#","top_logprobs":2"#,
        r#","logprobs":false,"top_logprobs":2"#,
    ] {
        let (status, body) = post(quantized_state(), CHAT, &chat_body(extra)).await;
        assert!(status.is_client_error(), "{extra}: {status} {body}");
        assert!(body.contains("top_logprobs"), "{extra}: {body}");
    }
}

#[tokio::test]
async fn a_streamed_request_for_logprobs_is_refused_not_served_without_them() {
    let (status, body) = post(
        quantized_state(),
        CHAT,
        &chat_body(r#","stream":true,"logprobs":true"#),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert!(body.contains("quantized (streaming)"), "{body}");
}
