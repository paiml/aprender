//! CRUX-C-05 falsification tests — OpenAI SSE streaming on
//! POST /v1/chat/completions (stream=true), terminated by `data: [DONE]`.
//!
//! Contract: `contracts/crux-C-05-v1.yaml` (competitor parity: vLLM
//! openai_compatible_server, canonical reference:
//! https://platform.openai.com/docs/api-reference/chat/streaming).
//!
//! In-process axum via `tower::ServiceExt::oneshot`; no TCP, no network.
//! The canonical `/v1/chat/completions` route dispatches the stream=true
//! path through `registry_fallback` → `pregenerated_sse_response` for the
//! demo `AppState` (no GPU/CUDA/quantized models loaded in tests).

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use realizar::api::{create_router, AppState};
use tower::ServiceExt;

fn router() -> axum::Router {
    let state = AppState::demo().expect("demo state should build");
    create_router(state)
}

fn chat_request(payload: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("request build")
}

async fn collect_bytes(resp: axum::http::Response<Body>) -> Vec<u8> {
    resp.into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes()
        .to_vec()
}

/// Parse an SSE wire body into an ordered list of `data:` payloads
/// (one per event, multi-line values joined by "\n").
///
/// SSE framing: events are `\n\n`-separated; within an event each `data: `
/// line contributes to the event's value, joined with `\n`.
fn parse_sse_data_frames(bytes: &[u8]) -> Vec<String> {
    let body = std::str::from_utf8(bytes).unwrap_or("");
    let mut frames = Vec::new();
    for raw in body.split("\n\n") {
        let mut lines: Vec<&str> = Vec::new();
        for line in raw.split('\n') {
            if let Some(rest) = line.strip_prefix("data: ") {
                lines.push(rest);
            } else if let Some(rest) = line.strip_prefix("data:") {
                // SSE allows `data:` with or without a following space.
                lines.push(rest);
            }
        }
        if !lines.is_empty() {
            frames.push(lines.join("\n"));
        }
    }
    frames
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-05-001: stream=true yields Content-Type: text/event-stream
// terminated by a literal `data: [DONE]` frame (no frames after).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_05_001_content_type_and_done_terminator() {
    let req = chat_request(serde_json::json!({
        "model": "default",
        "messages": [{"role":"user","content":"Count: 1 2 3"}],
        "stream": true,
        "max_tokens": 8,
        "temperature": 0.01
    }));
    let resp = router().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    if status != StatusCode::OK {
        let bytes = collect_bytes(resp).await;
        panic!(
            "FALSIFY-CRUX-C-05-001: stream=true must return 200, got {status}: body={:?}",
            String::from_utf8_lossy(&bytes)
        );
    }
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.to_ascii_lowercase().starts_with("text/event-stream"),
        "FALSIFY-CRUX-C-05-001: Content-Type must start with text/event-stream, got {ct:?}"
    );

    let bytes = collect_bytes(resp).await;
    let frames = parse_sse_data_frames(&bytes);
    assert!(
        !frames.is_empty(),
        "FALSIFY-CRUX-C-05-001: stream must emit at least one SSE frame"
    );
    let last = frames.last().expect("last frame");
    assert_eq!(
        last, "[DONE]",
        "FALSIFY-CRUX-C-05-001: terminal frame must be literal `[DONE]`, got {last:?}"
    );

    // No non-empty frames after [DONE] — [DONE] must be last.
    let done_idx = frames.iter().rposition(|f| f == "[DONE]").unwrap();
    assert_eq!(
        done_idx,
        frames.len() - 1,
        "FALSIFY-CRUX-C-05-001: no frames may appear after [DONE]"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-05-002: every non-terminal frame is a JSON object with
// object == "chat.completion.chunk".
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_05_002_every_chunk_is_chat_completion_chunk() {
    let req = chat_request(serde_json::json!({
        "model": "default",
        "messages": [{"role":"user","content":"hi"}],
        "stream": true,
        "max_tokens": 6,
        "temperature": 0.01
    }));
    let resp = router().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = collect_bytes(resp).await;
    let frames = parse_sse_data_frames(&bytes);
    assert!(!frames.is_empty(), "FALSIFY-CRUX-C-05-002: no SSE frames");

    for (i, f) in frames.iter().enumerate() {
        if f == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(f).unwrap_or_else(|e| {
            panic!("FALSIFY-CRUX-C-05-002: frame {i} is not valid JSON: {f:?} ({e})")
        });
        assert_eq!(
            v["object"].as_str(),
            Some("chat.completion.chunk"),
            "FALSIFY-CRUX-C-05-002: frame {i} object must be literal \
             \"chat.completion.chunk\", got {v}"
        );
        assert!(
            v["choices"].is_array() && !v["choices"].as_array().unwrap().is_empty(),
            "FALSIFY-CRUX-C-05-002: frame {i} must have non-empty choices array"
        );
    }
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-05-003: under deterministic sampling (temperature=0), the
// concatenation of streamed `delta.content` values equals the non-streaming
// `choices[0].message.content` for the same request.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_05_003_delta_concat_equals_nonstream_content() {
    let base = serde_json::json!({
        "model": "default",
        "messages": [{"role":"user","content":"Say hello"}],
        "max_tokens": 12,
        "temperature": 0.01
    });

    // Non-stream
    let mut ns = base.clone();
    ns["stream"] = serde_json::Value::Bool(false);
    let ns_resp = router()
        .oneshot(chat_request(ns))
        .await
        .expect("oneshot ns");
    let ns_status = ns_resp.status();
    if ns_status != StatusCode::OK {
        let bytes = collect_bytes(ns_resp).await;
        panic!(
            "FALSIFY-CRUX-C-05-003: non-stream must return 200, got {ns_status}: body={:?}",
            String::from_utf8_lossy(&bytes)
        );
    }
    let ns_bytes = collect_bytes(ns_resp).await;
    let ns_json: serde_json::Value =
        serde_json::from_slice(&ns_bytes).expect("FALSIFY-CRUX-C-05-003: non-stream json");
    let ns_content = ns_json["choices"][0]["message"]["content"]
        .as_str()
        .expect("FALSIFY-CRUX-C-05-003: non-stream content")
        .to_string();

    // Stream
    let mut s = base;
    s["stream"] = serde_json::Value::Bool(true);
    let s_resp = router().oneshot(chat_request(s)).await.expect("oneshot s");
    assert_eq!(s_resp.status(), StatusCode::OK);
    let s_bytes = collect_bytes(s_resp).await;
    let frames = parse_sse_data_frames(&s_bytes);

    let mut accum = String::new();
    for f in &frames {
        if f == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(f) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
            accum.push_str(content);
        }
    }

    assert_eq!(
        accum, ns_content,
        "FALSIFY-CRUX-C-05-003: Σ delta.content must equal non-stream content under \
         deterministic sampling (temperature=0)"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-05-004: exactly one chunk carries a non-null finish_reason,
// and it is the last JSON chunk before the `[DONE]` terminator.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_05_004_finish_reason_exactly_once_before_done() {
    let req = chat_request(serde_json::json!({
        "model": "default",
        "messages": [{"role":"user","content":"ok"}],
        "stream": true,
        "max_tokens": 6,
        "temperature": 0.01
    }));
    let resp = router().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = collect_bytes(resp).await;
    let frames = parse_sse_data_frames(&bytes);

    let mut nonnull_count = 0usize;
    let mut last_nonnull_idx: Option<usize> = None;
    let mut last_json_idx: Option<usize> = None;
    for (i, f) in frames.iter().enumerate() {
        if f == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(f) {
            Ok(v) => v,
            Err(_) => continue,
        };
        last_json_idx = Some(i);
        let fr = &v["choices"][0]["finish_reason"];
        if !fr.is_null() {
            nonnull_count += 1;
            last_nonnull_idx = Some(i);
            let fr_s = fr
                .as_str()
                .expect("FALSIFY-CRUX-C-05-004: finish_reason must be a string when non-null");
            assert!(
                matches!(fr_s, "stop" | "length" | "content_filter" | "tool_calls"),
                "FALSIFY-CRUX-C-05-004: finish_reason {fr_s:?} not in \
                 {{stop,length,content_filter,tool_calls}}"
            );
        }
    }

    assert_eq!(
        nonnull_count, 1,
        "FALSIFY-CRUX-C-05-004: expected exactly 1 non-null finish_reason, got {nonnull_count}"
    );
    assert_eq!(
        last_nonnull_idx, last_json_idx,
        "FALSIFY-CRUX-C-05-004: non-null finish_reason must be on the last JSON chunk \
         before [DONE] (was at {last_nonnull_idx:?}, last JSON chunk at {last_json_idx:?})"
    );
}
