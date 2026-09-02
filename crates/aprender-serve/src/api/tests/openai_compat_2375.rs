//! Falsifiers for the OpenAI-compat surface defects found by dogfooding
//! `cargo install aprender` 0.63.0 from crates.io (aprender#2375).
//!
//! Each test asserts a property a CLIENT can observe, not the shape of an
//! internal function:
//!
//! - #3/#5 `POST /v1/completions` accepted `"stream":true` and answered
//!   `content-type: application/json` with a `content-length`. The request type
//!   had no `stream` field at all, so serde dropped the key; the handler's
//!   return type (`Json<CompletionResponse>`) could not have carried an SSE
//!   body even if it had read it.
//! - #6 the streaming chat path emitted `finish_reason: "stop"` in its terminal
//!   chunk even when the generation was cut off at `max_tokens`, while the
//!   NON-streaming response to the same request correctly said `"length"`.
//! - #8 `POST /v1/predict` answered `"No APR model loaded. Use AppState::demo()
//!   or load a .apr model."` on a server that had a model loaded, leaking an
//!   internal Rust constructor to HTTP clients.
//! - #9 `"n": 3` was accepted with HTTP 200 and one choice — the field was
//!   declared and read by nobody.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use std::sync::Arc;
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

/// A router over a REAL demo model, created once for the whole module.
///
/// `create_test_app_shared()` is backed by `demo_mock()` (no model at all), so
/// every generation-dependent assertion here would answer 404 and prove
/// nothing. `AppState::demo()` costs ~0.5s, so it is built once and cloned.
fn demo_app() -> axum::Router {
    static DEMO: std::sync::OnceLock<AppState> = std::sync::OnceLock::new();
    let state = DEMO.get_or_init(|| AppState::demo().expect("demo AppState"));
    create_router(state.clone())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn post(app: axum::Router, uri: &str, body: &str) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request"),
    )
    .await
    .expect("response")
}

fn content_type(response: &axum::response::Response) -> String {
    response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    String::from_utf8(bytes.to_vec()).expect("body is utf-8")
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let text = body_text(response).await;
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("body is JSON ({e}): {text}"))
}

/// The JSON payloads of an SSE body, in order, excluding the `[DONE]` sentinel.
fn sse_payloads(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| payload.trim() != "[DONE]")
        .map(|payload| {
            serde_json::from_str(payload).unwrap_or_else(|e| panic!("SSE frame is JSON ({e})"))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// #2375 findings 3 + 5: /v1/completions must honour stream:true
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_stream_true_is_sse_not_a_buffered_json_body() {
    let response = post(
        demo_app(),
        "/v1/completions",
        r#"{"model":"default","prompt":"token5 token6","max_tokens":4,"stream":true}"#,
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        content_type(&response).starts_with("text/event-stream"),
        "stream:true must be framed as SSE; 0.63.0 answered {:?}",
        content_type(&response)
    );
    // 0.63.0's proof of buffering: a content-length on a streamed response.
    assert!(
        response.headers().get("content-length").is_none(),
        "a streamed body must not be length-prefixed"
    );

    let body = body_text(response).await;
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "an OpenAI stream terminates with the [DONE] sentinel; got:\n{body}"
    );
    let frames = sse_payloads(&body);
    assert!(!frames.is_empty(), "stream carried no data frames:\n{body}");
    for frame in &frames {
        assert_eq!(
            frame["object"], "text_completion",
            "every completion chunk keeps the text_completion envelope"
        );
    }
}

/// The delta-reassembly property: concatenating the streamed `text` deltas must
/// reproduce the completion byte-for-byte, and only the terminal frame may
/// carry a finish reason. Driven with a fixed completion rather than the demo
/// model, whose generated tokens decode to the empty string — an empty text
/// would make the equality vacuous.
#[tokio::test]
async fn completion_sse_frames_reassemble_to_the_completion_text() {
    use crate::api::realize_handlers::{
        completion_sse_response, CompletionChoice, CompletionResponse,
    };
    use crate::api::Usage;

    let text = "The quick brown fox\njumps over the lazy dog.";
    let response = CompletionResponse {
        id: "cmpl-test".to_string(),
        object: "text_completion".to_string(),
        created: 1,
        model: "test-model".to_string(),
        choices: vec![CompletionChoice {
            text: text.to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "length".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 9,
            total_tokens: 10,
        },
    };

    let body = body_text(completion_sse_response(&response)).await;
    let frames = sse_payloads(&body);
    let (last, rest) = frames.split_last().expect("at least a terminal frame");

    let reassembled: String = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(
        reassembled, text,
        "concatenated deltas must equal the completion text"
    );
    assert!(
        rest.len() > 1,
        "a single-frame 'stream' is not a stream: {body}"
    );
    for frame in rest {
        assert!(
            frame["choices"][0]["finish_reason"].is_null(),
            "only the terminal frame carries a finish reason: {frame}"
        );
        assert_eq!(frame["id"], "cmpl-test", "one id for the whole completion");
    }
    assert_eq!(
        last["choices"][0]["finish_reason"], "length",
        "the terminal frame reports the reason the backend computed, not a literal"
    );
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "stream must end with the [DONE] sentinel: {body}"
    );
}

/// The terminal frame — and only the terminal frame — carries a finish reason,
/// and it is the SAME reason the non-streaming response reports (#6 applied to
/// `/v1/completions`).
#[tokio::test]
async fn completions_stream_finish_reason_matches_the_non_streamed_one() {
    const REQUEST: &str = r#"{"model":"default","prompt":"token9","max_tokens":3"#;

    let buffered =
        body_json(post(demo_app(), "/v1/completions", &format!("{REQUEST}}}")).await).await;
    let expected = buffered["choices"][0]["finish_reason"]
        .as_str()
        .expect("non-streaming finish_reason")
        .to_string();

    let streamed = body_text(
        post(
            demo_app(),
            "/v1/completions",
            &format!("{REQUEST},\"stream\":true}}"),
        )
        .await,
    )
    .await;
    let frames = sse_payloads(&streamed);
    let (last, rest) = frames.split_last().expect("at least one frame");

    for frame in rest {
        assert!(
            frame["choices"][0]["finish_reason"].is_null(),
            "only the terminal chunk may carry a finish reason: {frame}"
        );
    }
    assert_eq!(
        last["choices"][0]["finish_reason"],
        expected.as_str(),
        "streaming finish_reason must agree with the non-streaming one ({expected})"
    );
}

/// Control: without `stream`, the response must stay a single JSON body. A fix
/// that made streaming unconditional would pass the tests above and break every
/// non-streaming client.
#[tokio::test]
async fn completions_without_stream_stays_a_single_json_object() {
    let response = post(
        demo_app(),
        "/v1/completions",
        r#"{"model":"default","prompt":"token5","max_tokens":3}"#,
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        content_type(&response).starts_with("application/json"),
        "non-streaming must stay JSON, got {:?}",
        content_type(&response)
    );
    let body = body_text(response).await;
    assert!(
        !body.contains("data: "),
        "non-streaming body must not be SSE-framed: {body}"
    );
    let json: serde_json::Value = serde_json::from_str(&body).expect("single JSON object");
    assert_eq!(json["object"], "text_completion");
    assert!(json["usage"].is_object(), "buffered body carries usage");
}

// ---------------------------------------------------------------------------
// #2375 finding 6: streaming finish_reason must not be a constant
// ---------------------------------------------------------------------------

/// Streaming and non-streaming views of the SAME chat request must agree on why
/// generation ended. 0.63.0 said `"length"` buffered and `"stop"` streamed.
#[tokio::test]
async fn chat_stream_finish_reason_matches_the_non_streamed_one() {
    const REQUEST: &str = r#"{"model":"default","messages":[{"role":"user","content":"token5 token6"}],"max_tokens":3"#;

    let buffered =
        body_json(post(demo_app(), "/v1/chat/completions", &format!("{REQUEST}}}")).await).await;
    let expected = buffered["choices"][0]["finish_reason"]
        .as_str()
        .expect("non-streaming finish_reason")
        .to_string();

    let streamed = body_text(
        post(
            demo_app(),
            "/v1/chat/completions",
            &format!("{REQUEST},\"stream\":true}}"),
        )
        .await,
    )
    .await;
    let frames = sse_payloads(&streamed);
    let last = frames.last().expect("at least one frame");

    assert_eq!(
        last["choices"][0]["finish_reason"],
        expected.as_str(),
        "streaming terminal chunk disagreed with the non-streaming body \
         (expected {expected}); this is the #2375(6) hardcoded literal"
    );
}

/// The `max_tokens` budget must reach the terminal chunk. Fed exactly as many
/// tokens as the budget allows, the stream reports `"length"`; given a budget it
/// never reaches, the same tokens report `"stop"`. A hardcoded literal cannot
/// satisfy both halves.
#[tokio::test]
async fn true_streaming_terminal_chunk_distinguishes_length_from_stop() {
    use crate::tokenizer::BPETokenizer;

    async fn finish_reason_for(max_tokens: usize) -> String {
        let vocab: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let tokenizer = Arc::new(BPETokenizer::new(vocab, vec![], "a").expect("tokenizer"));
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(8);
        for id in 0u32..3 {
            tx.send(Ok(id)).await.expect("send token");
        }
        drop(tx);

        let response = crate::api::openai_handlers::true_streaming_sse_response(
            rx,
            tokenizer,
            "chatcmpl-test".to_string(),
            "test-model".to_string(),
            Arc::new(crate::metrics::MetricsCollector::new()),
            std::time::Instant::now(),
            max_tokens,
            // No prompt was tokenized in this fixture; the terminal chunk's
            // usage reports the truth (0 prompt tokens), not a guess.
            0,
            None,
        );
        let body = body_text(response).await;
        sse_payloads(&body)
            .last()
            .and_then(|frame| frame["choices"][0]["finish_reason"].as_str())
            .map(str::to_string)
            .unwrap_or_default()
    }

    assert_eq!(
        finish_reason_for(3).await,
        "length",
        "3 tokens under a 3-token budget were truncated"
    );
    assert_eq!(
        finish_reason_for(64).await,
        "stop",
        "3 tokens under a 64-token budget ended on their own"
    );
}

// ---------------------------------------------------------------------------
// #2375 finding 9: `n` must not be accepted-and-ignored
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_request_with_n_greater_than_one_is_rejected_not_silently_ignored() {
    let response = post(
        demo_app(),
        "/v1/chat/completions",
        r#"{"model":"default","messages":[{"role":"user","content":"token5"}],"max_tokens":4,"n":3}"#,
    )
    .await;

    assert!(
        response.status().is_client_error(),
        "n:3 must be refused, not answered 200 with one choice (got {})",
        response.status()
    );
    let body = body_text(response).await;
    assert!(
        body.contains("n must be 1"),
        "the refusal must say what is wrong with n: {body}"
    );
}

#[tokio::test]
async fn completions_request_with_n_greater_than_one_is_rejected() {
    let response = post(
        demo_app(),
        "/v1/completions",
        r#"{"model":"default","prompt":"token5","max_tokens":4,"n":5}"#,
    )
    .await;

    assert!(
        response.status().is_client_error(),
        "n:5 must be refused (got {})",
        response.status()
    );
}

/// Control: `n:1` and an absent `n` are still served normally.
#[tokio::test]
async fn n_of_one_and_absent_n_are_both_accepted() {
    for body in [
        r#"{"model":"default","prompt":"token5","max_tokens":3,"n":1}"#,
        r#"{"model":"default","prompt":"token5","max_tokens":3}"#,
    ] {
        let response = post(demo_app(), "/v1/completions", body).await;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "supported request was refused: {body}"
        );
        let json = body_json(response).await;
        assert_eq!(
            json["choices"].as_array().map(Vec::len),
            Some(1),
            "one choice per request"
        );
    }
}

// ---------------------------------------------------------------------------
// #2375 finding 8: /v1/predict must not lie, and must not leak Rust internals
// ---------------------------------------------------------------------------

/// A server with a generative model loaded (`AppState::new`, no APR estimator)
/// is exactly the situation the dogfood run hit: the log said "APR loaded: 291
/// tensors" while `/v1/predict` claimed no model was loaded and told the caller
/// to use an internal Rust constructor.
fn app_with_generative_model_only() -> axum::Router {
    use crate::layers::{Model, ModelConfig};
    use crate::tokenizer::BPETokenizer;

    let model = Model::new(ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
    })
    .expect("demo model");
    let vocab: Vec<String> = (0..100)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("token{i}")
            }
        })
        .collect();
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("tokenizer");
    let state = AppState::new(model, tokenizer);
    assert!(
        state.model_loaded(),
        "fixture must have a model resident, or it does not reproduce the finding"
    );
    create_router(state)
}

#[tokio::test]
async fn predict_without_an_estimator_says_what_is_actually_loaded() {
    let response = post(
        app_with_generative_model_only(),
        "/v1/predict",
        r#"{"features":[1.0,2.0,3.0]}"#,
    )
    .await;

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let error = body_json(response).await["error"]
        .as_str()
        .expect("error string")
        .to_string();

    assert!(
        !error.contains("No APR model loaded"),
        "the server HAS a model loaded; saying otherwise contradicts /health: {error}"
    );
    assert!(
        error.contains("/v1/completions") || error.contains("/v1/chat/completions"),
        "an unusable endpoint must name the one that works: {error}"
    );
}

/// No shipped string may name a Rust constructor the client cannot call.
///
/// This scans EVERY string literal under `src/` (not only lines that happen to
/// mention "error"): the first cut of this guard filtered on the word `error`
/// appearing on the same line, and re-running the #2375(8) mutation showed the
/// filter blind as soon as the literal moved to its own line. A guard that only
/// covers the shape the defect happened to have is theater.
#[test]
fn no_shipped_string_leaks_an_internal_rust_constructor() {
    // Assembled from fragments so this guard does not match its own source.
    let needles: Vec<String> = vec![
        format!("{}{}", "AppState", "::demo()"),
        format!("{}{}", "AppState", "::new()"),
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

    let mut offenders: Vec<String> = Vec::new();
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    collect_rs_files(&root, &mut files);
    assert!(files.len() > 100, "scan found only {} files", files.len());

    // Self-check: the guard must be able to see a literal in the very file the
    // finding was in, so a silent scan failure cannot pass as a clean result.
    let mut scanned_apr_handlers = false;

    for path in files {
        // Only shipped handler strings matter; the test tree may quote the
        // offending message when asserting that it is gone.
        if path.to_string_lossy().contains("/tests/") {
            continue;
        }
        if path.ends_with("apr_handlers.rs") {
            scanned_apr_handlers = true;
        }
        let Ok(source) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (lineno, line) in source.lines().enumerate() {
            // Doc comments explain internals on purpose; only real literals ship.
            if line.trim_start().starts_with("//") {
                continue;
            }
            for text in string_literals(line) {
                for needle in &needles {
                    if text.contains(needle.as_str()) {
                        offenders.push(format!(
                            "{}:{}: {}",
                            path.display(),
                            lineno + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
    }

    assert!(scanned_apr_handlers, "guard never reached apr_handlers.rs");
    assert!(
        offenders.is_empty(),
        "shipped strings must not name internal Rust constructors (#2375 finding 8):\n{}",
        offenders.join("\n")
    );
}

fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Extract double-quoted string literals from a line (good enough for a source
/// guard: it over-approximates, which can only make the guard stricter).
fn string_literals(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    let mut escaped = false;
    for ch in line.chars() {
        match current.as_mut() {
            Some(buf) => {
                if escaped {
                    buf.push(ch);
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    out.push(current.take().unwrap_or_default());
                } else {
                    buf.push(ch);
                }
            },
            None => {
                if ch == '"' {
                    current = Some(String::new());
                }
            },
        }
    }
    if let Some(buf) = current {
        out.push(buf);
    }
    out
}
