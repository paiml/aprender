//! FALSIFY-LOGPROBS-WIRE-4971: `logprobs` / `top_logprobs` on `/v1/chat/completions`
//! (ASOC-INV-021), through the real router on a quantized CPU server.
//!
//! Before #4971 the request type had neither field: serde dropped both, the reply
//! carried no `logprobs`, and a client that asked for them was answered as if it had
//! not. Every test here asserts what that client observes.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::Value;
use tower::util::ServiceExt;

use super::native_routes_2376::{body_string, post, quantized_state};
use crate::api::create_router;

const CHAT: &str = "/v1/chat/completions";

/// `ignore_eos`: the fixture model's greedy first token is a stop token, so
/// without it every reply here is empty and the per-token checks see nothing.
fn chat_body(extra: &str) -> String {
    format!(
        r#"{{"model":"default","messages":[{{"role":"user","content":"Hi"}}],"max_tokens":4,"temperature":0,"ignore_eos":true{extra}}}"#
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

/// The `data:` chunks of an SSE body, without the closing `[DONE]`.
fn stream_chunks(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("an SSE chunk is JSON"))
        .collect()
}

async fn stream_chunks_of(extra: &str) -> Vec<Value> {
    let body = chat_body(&format!(r#","stream":true{extra}"#));
    let (status, body) = post(quantized_state(), CHAT, &body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = stream_chunks(&body);
    assert!(!chunks.is_empty(), "no chunks: {body}");
    chunks
}

#[tokio::test]
async fn a_stream_carries_on_its_chunks_the_entries_of_the_same_reply_unstreamed() {
    let chunks = stream_chunks_of(r#","logprobs":true,"top_logprobs":2"#).await;
    let completion_tokens = chunks
        .iter()
        .find_map(|c| c["usage"]["completion_tokens"].as_u64())
        .expect("the terminal chunk carries usage");
    let streamed: Vec<&Value> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["logprobs"]["content"].as_array())
        .flatten()
        .collect();
    assert!(completion_tokens > 0, "the fixture streamed nothing");
    assert_eq!(streamed.len() as u64, completion_tokens, "{chunks:?}");
    for entry in &streamed {
        let tops = entry["top_logprobs"].as_array().expect("top_logprobs");
        assert_eq!(tops.len(), 2, "{entry}");
        assert!(logprob(entry) <= 0.0, "{entry}");
        assert!((logprob(entry) - logprob(&tops[0])).abs() < 1e-4, "{entry}");
    }
    // Streaming changes no entry: the same request unstreamed has the same ones.
    let reply = chat_json(r#","logprobs":true,"top_logprobs":2"#).await;
    let unstreamed = reply["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("content");
    assert_eq!(streamed.len(), unstreamed.len(), "{reply}");
    for (s, u) in streamed.iter().zip(unstreamed) {
        assert_eq!(s["token"], u["token"], "{s} vs {u}");
        assert!((logprob(s) - logprob(u)).abs() < 1e-4, "{s} vs {u}");
    }
}

#[tokio::test]
async fn a_stream_without_logprobs_has_no_logprobs_key_on_any_chunk() {
    for chunk in stream_chunks_of("").await {
        let choice = chunk["choices"][0].as_object().expect("choice object");
        assert!(!choice.contains_key("logprobs"), "{chunk}");
    }
}

#[tokio::test]
async fn a_traced_request_for_logprobs_is_refused_not_served_without_them() {
    let request = Request::builder()
        .method("POST")
        .uri(CHAT)
        .header("content-type", "application/json")
        .header("X-Trace-Level", "brick")
        .body(Body::from(chat_body(r#","logprobs":true"#)))
        .expect("build request");
    let response = create_router(quantized_state())
        .oneshot(request)
        .await
        .expect("dispatch");
    let status = response.status();
    let body = body_string(response).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert!(body.contains("quantized (traced)"), "{body}");
}
