//! Falsifier: SSE streaming deltas must reassemble into the same text the
//! non-streaming response returns — whitespace included.
//!
//! Dogfooding `cargo install aprender` 0.63.0 from crates.io found that
//! `POST /v1/chat/completions` with `"stream":true` produced
//! `"Thequickbrownfoxjumpsoverthelazydog."` where the non-streaming call on the
//! same server, same model and same prompt returned
//! `"The quick brown fox jumps over the lazy dog."`.
//!
//! Cause: `true_streaming_sse_response` took a `clean: bool` and two of its
//! three call sites passed `true`, which ran `clean_chat_output()` on EVERY
//! SINGLE TOKEN. That function opens with `text.trim_start()` and closes with
//! `.trim()`, so the leading space BPE carries on the token itself (`"Ġquick"`
//! decodes to `" quick"`) was deleted from every delta. Newlines went the same
//! way.
//!
//! This asserts the property that actually matters to a client: concatenating
//! `choices[0].delta.content` across the stream reproduces the full decode
//! byte-for-byte. Reinstating the per-token clean turns it RED.

use crate::tokenizer::BPETokenizer;
use std::sync::Arc;

/// Build a tokenizer whose tokens carry leading whitespace, the way real BPE
/// vocabularies do (`Ġ` in GPT-2/Qwen byte-level BPE decodes to a space).
fn whitespace_bearing_tokenizer() -> (Arc<BPETokenizer>, Vec<u32>, String) {
    let vocab: Vec<String> = vec![
        "The".to_string(),
        " quick".to_string(),
        " brown".to_string(),
        " fox".to_string(),
        "\n".to_string(),
        " jumps".to_string(),
        ".".to_string(),
    ];
    let tokenizer = BPETokenizer::new(vocab.clone(), vec![], "The").expect("build tokenizer");
    let ids: Vec<u32> = (0..vocab.len() as u32).collect();
    let expected: String = vocab.concat();
    (Arc::new(tokenizer), ids, expected)
}

/// Collect the `delta.content` fields out of an SSE body, in order.
fn concat_sse_deltas(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload.trim() == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(text) = value["choices"][0]["delta"]["content"].as_str() {
            out.push_str(text);
        }
    }
    out
}

#[tokio::test]
async fn sse_deltas_reassemble_with_whitespace_intact() {
    let (tokenizer, ids, expected) = whitespace_bearing_tokenizer();

    // Sanity: the tokenizer really does round-trip the whitespace, so a failure
    // below is the streaming path and not the fixture.
    let full_decode = tokenizer.decode(&ids).expect("decode all");
    assert_eq!(
        full_decode, expected,
        "fixture is wrong: tokenizer does not preserve whitespace on a bulk decode"
    );

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(16);
    for id in &ids {
        tx.send(Ok(*id)).await.expect("send token");
    }
    drop(tx);

    let response = crate::api::openai_handlers::true_streaming_sse_response(
        rx,
        tokenizer,
        "chatcmpl-test".to_string(),
        "test-model".to_string(),
        Arc::new(crate::metrics::MetricsCollector::new()),
        std::time::Instant::now(),
        // Budget far above the token count: this stream ends on EOS, not length.
        256,
    );

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read SSE body");
    let body = String::from_utf8(bytes.to_vec()).expect("SSE body is utf-8");

    let streamed = concat_sse_deltas(&body);

    assert_eq!(
        streamed, expected,
        "concatenated SSE deltas must equal the full decode.\n\
         streamed: {streamed:?}\n\
         expected: {expected:?}\n\
         A mismatch here is the v0.63.0 defect: clean_chat_output() applied per \
         token strips the leading space/newline off every delta."
    );

    // Spell the two specific losses out, so a regression names itself.
    assert!(
        streamed.contains(" quick"),
        "leading spaces were stripped from the deltas: {streamed:?}"
    );
    assert!(
        streamed.contains('\n'),
        "newline tokens were stripped from the deltas: {streamed:?}"
    );
}
