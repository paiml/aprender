//! PP-27 falsifiers: an SSE stream must DECLARE how it is produced, and the
//! terminal chunk must carry the token counts (and the server's phase timings
//! when it measured them).
//!
//! Why this is a correctness claim and not a nicety: two builders in this crate
//! serve `stream: true`, and they are not the same thing.
//! `true_streaming_sse_response` writes a delta as each token leaves the decode
//! loop; `pregenerated_sse_response` generates the WHOLE completion first and
//! then replays it. A client measuring time-to-first-token and inter-token
//! latency against the second one is measuring the SSE writer, so a receipt
//! built over it records the wrong quantity while looking exactly like a
//! correct one. Before this, neither builder said which it was, and the
//! `--batch` GPU path (a cached model) silently used the replaying one.
//!
//! These drive the REAL router wherever a backend exists on CPU, because the
//! defect lives in what a client receives, not in what a builder returns.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn post_sse(state: AppState, uri: &str, json: &str) -> (StatusCode, String) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(json.to_string()))
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// Every `data:` frame that is not the `[DONE]` sentinel, parsed as JSON.
fn sse_frames(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| payload.trim() != "[DONE]")
        .filter_map(|payload| serde_json::from_str::<serde_json::Value>(payload).ok())
        .collect()
}

const STREAM_BODY: &str = r#"{"model":"default","messages":[{"role":"user","content":"token5 token6"}],"max_tokens":4,"temperature":0.0,"stream":true}"#;

// ---------------------------------------------------------------------------
// Claim 1: a live stream says so, and closes with usage
// ---------------------------------------------------------------------------

/// The CPU quantized deployment — what `apr serve run model.gguf` builds
/// without an accelerator — streams through `true_streaming_sse_response`.
/// Its first chunk must declare `live`, and only its first.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn live_quantized_stream_declares_live_and_carries_usage_on_terminal_chunk() {
    use super::native_routes_2376::quantized_state;

    let (status, body) = post_sse(quantized_state(), "/v1/chat/completions", STREAM_BODY).await;
    assert_eq!(status, StatusCode::OK, "stream request failed: {body}");

    let frames = sse_frames(&body);
    assert!(frames.len() >= 2, "expected an opening and a terminal chunk:\n{body}");

    assert_eq!(
        frames[0]["stream_mode"].as_str(),
        Some("live"),
        "the first chunk of a live stream must declare it; got {}\n{body}",
        frames[0]
    );
    for (i, frame) in frames.iter().enumerate().skip(1) {
        assert!(
            frame["stream_mode"].is_null(),
            "chunk {i} re-declared stream_mode; it belongs on the FIRST chunk only:\n{frame}"
        );
    }

    let terminal = frames.last().expect("terminal chunk");
    assert!(
        !terminal["usage"].is_null(),
        "the terminal chunk must carry usage:\n{terminal}"
    );
    // The deltas a client reassembles ARE the completion, so the count it is
    // told must be the count it received.
    let delta_chunks = frames
        .iter()
        .filter(|f| f["choices"][0]["delta"]["content"].is_string())
        .count();
    assert_eq!(
        terminal["usage"]["completion_tokens"].as_u64(),
        Some(delta_chunks as u64),
        "usage.completion_tokens must equal the deltas actually streamed \
         ({delta_chunks}):\n{terminal}"
    );
    assert_eq!(
        terminal["usage"]["total_tokens"].as_u64(),
        Some(
            terminal["usage"]["prompt_tokens"].as_u64().unwrap_or_default()
                + terminal["usage"]["completion_tokens"].as_u64().unwrap_or_default()
        ),
        "total_tokens must be the sum it claims to be:\n{terminal}"
    );

    // Non-terminal chunks must NOT carry usage: a client that sums them would
    // double-count.
    for (i, frame) in frames.iter().enumerate().take(frames.len() - 1) {
        assert!(
            frame["usage"].is_null(),
            "chunk {i} carried usage; it belongs on the terminal chunk only:\n{frame}"
        );
    }
}

/// The `/v1/chat/completions/stream` sibling route serves the same backend and
/// must make the same declaration — two routes on one server disagreeing about
/// how the stream is produced would be worse than neither declaring.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn the_stream_route_declares_the_same_mode_as_the_stream_flag() {
    use super::native_routes_2376::quantized_state;

    let (_, via_flag) = post_sse(quantized_state(), "/v1/chat/completions", STREAM_BODY).await;
    let (_, via_route) =
        post_sse(quantized_state(), "/v1/chat/completions/stream", STREAM_BODY).await;

    let flag_mode = sse_frames(&via_flag)
        .first()
        .and_then(|f| f["stream_mode"].as_str().map(str::to_string));
    let route_mode = sse_frames(&via_route)
        .first()
        .and_then(|f| f["stream_mode"].as_str().map(str::to_string));
    assert_eq!(flag_mode.as_deref(), Some("live"));
    assert_eq!(
        flag_mode, route_mode,
        "the two chat routes declared different stream modes for the same backend"
    );
}

// ---------------------------------------------------------------------------
// Claim 2: a replayed stream says THAT, and is distinguishable
// ---------------------------------------------------------------------------

/// The dense `Model` backend generates the whole completion, then replays it.
/// It must say `replayed` — this is the case a receipt has to be able to refuse
/// `ttft`/`itl_p95` from.
#[tokio::test]
async fn replayed_stream_declares_replayed() {
    let state = AppState::demo().expect("dense demo AppState");
    let (status, body) = post_sse(state, "/v1/chat/completions", STREAM_BODY).await;
    assert_eq!(status, StatusCode::OK, "stream request failed: {body}");

    let frames = sse_frames(&body);
    assert_eq!(
        frames[0]["stream_mode"].as_str(),
        Some("replayed"),
        "the pre-generated builder must declare `replayed`:\n{}",
        frames[0]
    );
    let terminal = frames.last().expect("terminal chunk");
    assert!(
        !terminal["usage"].is_null(),
        "a replayed stream still owes the client its token counts:\n{terminal}"
    );
    // §3: no phase split exists on this path — generation was over before the
    // first byte was written. Absent, not zero.
    assert!(
        terminal["timings"].is_null(),
        "a replayed stream cannot have measured a prefill phase:\n{terminal}"
    );
}

/// The two modes must be DISTINGUISHABLE on the wire. A declaration that reads
/// the same for both paths would satisfy every assertion above and discharge
/// nothing.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn live_and_replayed_are_different_declarations() {
    use super::native_routes_2376::quantized_state;

    let (_, live) = post_sse(quantized_state(), "/v1/chat/completions", STREAM_BODY).await;
    let (_, replayed) = post_sse(
        AppState::demo().expect("dense demo AppState"),
        "/v1/chat/completions",
        STREAM_BODY,
    )
    .await;

    let live_mode = sse_frames(&live)[0]["stream_mode"].as_str().map(str::to_string);
    let replayed_mode = sse_frames(&replayed)[0]["stream_mode"]
        .as_str()
        .map(str::to_string);
    assert_eq!(live_mode.as_deref(), Some("live"));
    assert_eq!(replayed_mode.as_deref(), Some("replayed"));
    assert_ne!(
        live_mode, replayed_mode,
        "the two SSE mechanisms must not declare the same mode"
    );
}

// ---------------------------------------------------------------------------
// Claim 3: `stream_options` is accepted, and does not gate the emission
// ---------------------------------------------------------------------------

/// An OpenAI client that sends `stream_options: {"include_usage": true}` must
/// not be rejected as malformed...
#[tokio::test]
async fn stream_options_include_usage_is_accepted() {
    let state = AppState::demo().expect("dense demo AppState");
    let body = r#"{"model":"default","messages":[{"role":"user","content":"token5"}],"max_tokens":2,"temperature":0.0,"stream":true,"stream_options":{"include_usage":true}}"#;
    let (status, response) = post_sse(state, "/v1/chat/completions", body).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "stream_options must be accepted, got {status}: {response}"
    );
    let frames = sse_frames(&response);
    assert!(
        !frames.last().expect("terminal chunk")["usage"].is_null(),
        "usage must be present when the client asked for it"
    );
}

/// ...and a client that does NOT send it still gets usage. PP-27 needs the
/// counts on every run: a harness cannot retro-fit an opt-in flag onto a band
/// that already happened, and llama-server emits its `timings` unconditionally
/// for the same reason.
#[tokio::test]
async fn usage_is_emitted_without_the_opt_in() {
    let state = AppState::demo().expect("dense demo AppState");
    let (_, response) = post_sse(state, "/v1/chat/completions", STREAM_BODY).await;
    let frames = sse_frames(&response);
    assert!(
        !frames.last().expect("terminal chunk")["usage"].is_null(),
        "usage must be emitted regardless of stream_options"
    );
}

// ---------------------------------------------------------------------------
// Claim 4: §3 timings are measured or absent — never zero
// ---------------------------------------------------------------------------

/// A backend that did not separate prefill from decode reports NO `timings`
/// key. `0.0` would enter `prefill_ratio` as a measurement.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn timings_absent_is_null_not_zero() {
    use super::native_routes_2376::quantized_state;

    let nonstream = r#"{"model":"default","messages":[{"role":"user","content":"token5 token6"}],"max_tokens":4,"temperature":0.0}"#;
    let (status, body) = post_sse(quantized_state(), "/v1/chat/completions", nonstream).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json body");
    assert!(
        parsed["timings"].is_null(),
        "the CPU quantized backend does not measure a phase split; it must report \
         no timings rather than zeros:\n{parsed}"
    );
    assert!(
        !body.contains("\"prompt_ms\":0"),
        "a zero prefill duration must never reach the wire:\n{body}"
    );

    // Streaming form of the same claim.
    let (_, stream_body) = post_sse(quantized_state(), "/v1/chat/completions", STREAM_BODY).await;
    let frames = sse_frames(&stream_body);
    assert!(
        frames.last().expect("terminal chunk")["timings"].is_null(),
        "an unmeasured phase split must be absent on the terminal chunk too"
    );
}

/// When a backend DOES measure, the block appears with llama.cpp's key names,
/// so one client parser serves both lanes.
///
/// Driven through `build_chat_response` — the single function every
/// non-streaming backend returns through — because no CPU backend in this crate
/// measures a phase split, and asserting the shape on a fabricated CUDA run
/// would prove less than nothing.
#[tokio::test]
async fn nonstream_response_carries_timings_when_measured() {
    use crate::api::PhaseTimings;

    let measured = PhaseTimings {
        prefill_ms: Some(40.0),
        decode_ms: Some(200.0),
    };
    let timings = measured
        .to_timings(512, 128)
        .expect("both phases measured, so a wire block is representable");

    let response = crate::api::openai_handlers::build_chat_response(
        "chatcmpl-test".to_string(),
        "test-model".to_string(),
        "hello".to_string(),
        512,
        128,
        128,
        None,
        None,
        std::time::Duration::from_millis(240),
        None,
        None,
        Some(timings),
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let parsed: serde_json::Value =
        serde_json::from_slice(&bytes).expect("json body");

    let t = &parsed["timings"];
    assert!(!t.is_null(), "measured timings must reach the wire:\n{parsed}");
    // llama.cpp's key names, verbatim.
    assert_eq!(t["prompt_n"].as_u64(), Some(512));
    assert_eq!(t["prompt_ms"].as_f64(), Some(40.0));
    assert_eq!(t["predicted_n"].as_u64(), Some(128));
    assert_eq!(t["predicted_ms"].as_f64(), Some(200.0));
    // 512 tokens in 40 ms = 12 800 tok/s; 128 in 200 ms = 640 tok/s.
    assert!(
        (t["prompt_per_second"].as_f64().expect("prompt rate") - 12_800.0).abs() < 1e-6,
        "prompt_per_second must be prompt_n/prompt_ms, got {t}"
    );
    assert!(
        (t["predicted_per_second"].as_f64().expect("decode rate") - 640.0).abs() < 1e-6,
        "predicted_per_second must be predicted_n/predicted_ms, got {t}"
    );
    assert!(
        t["clock"].as_str().is_some_and(|c| c.contains("Instant")),
        "the block must state which clock produced it, got {t}"
    );

    // usage and timings must AGREE about the prompt: a receipt divides one by
    // the other.
    assert_eq!(
        parsed["usage"]["prompt_tokens"].as_u64(),
        t["prompt_n"].as_u64(),
        "timings.prompt_n and usage.prompt_tokens must be the same number"
    );
}

// ---------------------------------------------------------------------------
// The conversion rule itself
// ---------------------------------------------------------------------------

#[cfg(test)]
mod phase_timings_rules {
    use crate::api::{PhaseTimings, Timings};

    /// One measured phase is not a phase split. Filling the other with `0.0`
    /// would put a fabricated numerator into a gated ratio.
    #[test]
    fn a_half_measured_split_produces_no_wire_block() {
        assert!(PhaseTimings {
            prefill_ms: Some(40.0),
            decode_ms: None,
        }
        .to_timings(512, 128)
        .is_none());
        assert!(PhaseTimings {
            prefill_ms: None,
            decode_ms: Some(200.0),
        }
        .to_timings(512, 128)
        .is_none());
        assert!(PhaseTimings::default().to_timings(512, 128).is_none());
        assert!(PhaseTimings {
            prefill_ms: Some(40.0),
            decode_ms: Some(200.0),
        }
        .to_timings(512, 128)
        .is_some());
    }

    /// A rate over a zero-length interval is undefined; the key is omitted
    /// rather than reported as `0.0`, which would read as "infinitely slow".
    #[test]
    fn a_zero_duration_yields_no_rate() {
        let t = Timings::from_phases(512, 0.0, 128, 200.0);
        assert!(t.prompt_per_second.is_none());
        assert!(t.predicted_per_second.is_some());
        let json = serde_json::to_value(&t).expect("serialize");
        assert!(
            json.get("prompt_per_second").is_none(),
            "an undefined rate must not appear as a key: {json}"
        );
        assert_eq!(json["prompt_ms"].as_f64(), Some(0.0));
    }

    /// The rate is per SECOND from a duration in MILLISECONDS — a factor-1000
    /// slip here would make every prefill ratio look like a thousandfold win.
    #[test]
    fn the_rate_is_per_second_not_per_millisecond() {
        let t = Timings::from_phases(1000, 1000.0, 10, 100.0);
        assert_eq!(t.prompt_per_second, Some(1000.0));
        assert_eq!(t.predicted_per_second, Some(100.0));
    }
}
